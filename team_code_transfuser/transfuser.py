import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_, to_2tuple
from einops import rearrange
from RSMambaFusion import MambaBlock
from DeformableAtt import MultiHead
class TransfuserBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, use_velocity=True):
        super().__init__()
        self.config = config       

        if(config.use_point_pillars == True):
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len

        if(self.config.use_target_point_image == True):
            in_channels += 1

        self.mamba = MambaBlock(patch_size=4, 
                                in_chans=3, 
                                num_classess = self.config.num_classes, 
                                depths=[2, 2, 9, 2], 
                                dims=[96, 192, 384, 768], 
                                # =========================
                                ssm_d_state=16,
                                ssm_ratio=2.0,
                                ssm_dt_rank="auto",
                                ssm_act_layer="silu",        
                                ssm_conv=3,
                                ssm_conv_bias=True,
                                ssm_drop_rate=0.0, 
                                ssm_init="v0",
                                forward_type="v2",
                                # =========================
                                mlp_ratio=4.0,
                                mlp_act_layer="gelu",
                                mlp_drop_rate=0.0,
                                # =========================
                                drop_path_rate=0.1, 
                                patch_norm=True, 
                                norm_layer="LN",use_checkpoint=False)
        

        self.avg_pool_img = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool_lidar = nn.AdaptiveAvgPool2d((1,1))
        # FPN fusion
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        # top down
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))

    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        
        return p2, p3, p4, p5
        
    def forward(self, image, lidar, velocity):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        image_features, lidar_features = self.mamba(image, lidar, velocity)       

        x4 = lidar_features
        image_features_grid = image_features  # For auxilliary information

        image_features = self.avg_pool_img(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.avfl_pool_lidar(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)

        fused_features = image_features + lidar_features

        features = self.top_down(x4)
        return features, image_features_grid, fused_features

class SegDecoder(nn.Module):
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.num_class = config.num_class

        self.deconv1 = nn.Sequential(
                    nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv2 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv3 = nn.Sequential(
                    nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.config.deconv_channel_num_3, self.num_class, 3, 1, 1),
                    )

    def forward(self, x):
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)

        return x

# we will change depth decoder by our deformable attention
# the final input need to return (B, H * scale_factor_1 * scale_factor_2, W * scale_factor_1 * scale_factor_2)
# the latent dim need to be change (512 is final input of transfuser above, that mean we need to change by num_class or setting up as above)
class DepthDecoder(nn.Module):
    def __init__(
            self,
            config,           
    ):
        super().__init__()
        self.config = config
        self.latent_dim = self.config.num_classes

        self.deformable = MultiHead(sdim = self.config.deconv_channel_num_3,n_heads = 8, in_channels = self.latent_dim, channels= 8, ffw_dims= 128)
        self.deconv1 = nn.Seuquential(
            nn.Conv2d(self.config.num_classes, self.config.num_classes, 3,1,1),
            nn.ReLU(True),
            nn.Conv2d(self.config.num_classes,1,3,1,1)
        )
    def forward(self, x):
        x = self.deformable(x)
        x = self.deconv1(x)
        x = torch.sigmoid(x).squeeze(1)

        return x
# class DepthDecoder(nn.Module):
#     def __init__(self, config, latent_dim=512):
#        # 512 because this taking input from last embedding from transfuserbackbone, which now is change into self.config.num_classes
#         super().__init__()
#         self.config = config
#         self.latent_dim = latent_dim

#         self.deconv1 = nn.Sequential(
#                     nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
#                     nn.ReLU(True),
#                     nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
#                     nn.ReLU(True),
#                     )
#         self.deconv2 = nn.Sequential(
#                     nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
#                     nn.ReLU(True),
#                     nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
#                     nn.ReLU(True),
#                     )
#         self.deconv3 = nn.Sequential(
#                     nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
#                     nn.ReLU(True),
#                     nn.Conv2d(self.config.deconv_channel_num_3, 1, 3, 1, 1),
#                     )

#     def forward(self, x):
#         x = self.deconv1(x)
#         x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
#         x = self.deconv2(x)
#         x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
#         x = self.deconv3(x) # it return  (B, 1, H, W)
#         x = torch.sigmoid(x).squeeze(1) # (B, H, W)
        
#         return x

# we will do GPT echange by MambaBlock
# task here, we have to take pretrain-model with weighted parameter due to attack occur inside RSM_SS, loading and test for predicting waypoint and other task
        
class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, architecture, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = None
        # Delete parts of the networks we don't want
        if (architecture.startswith('regnet')): # Rename modules so we can use the same code
            self.features.conv1 = self.features.stem.conv
            self.features.bn1  = self.features.stem.bn
            self.features.act1 = nn.Sequential() #The Relu is part of the batch norm here.
            self.features.maxpool =  nn.Sequential()
            self.features.layer1 =self.features.s1
            self.features.layer2 =self.features.s2
            self.features.layer3 =self.features.s3
            self.features.layer4 =self.features.s4
            self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.features.head = nn.Sequential()

        elif (architecture.startswith('convnext')):
            self.features.conv1 = self.features.stem._modules['0']
            self.features.bn1 = self.features.stem._modules['1']
            self.features.act1 = nn.Sequential()  # Don't see any activatin function after the stem. Need to verify
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.stages._modules['0']
            self.features.layer2 = self.features.stages._modules['1']
            self.features.layer3 = self.features.stages._modules['2']
            self.features.layer4 = self.features.stages._modules['3']
            self.features.global_pool = self.features.head
            self.features.global_pool.flatten = nn.Sequential()
            self.features.global_pool.fc = nn.Sequential()
            self.features.head = nn.Sequential()
            # ConvNext don't have the 0th entry that res nets use.
            self.features.feature_info.append(self.features.feature_info[3])
            self.features.feature_info[3] = self.features.feature_info[2]
            self.features.feature_info[2] = self.features.feature_info[1]
            self.features.feature_info[1] = self.features.feature_info[0]

            #This layer norm is not pretrained anymore but that shouldn't matter since it is the last layer in the network.
            _tmp = self.features.global_pool.norm
            self.features.global_pool.norm = nn.LayerNorm((self.config.perception_output_features,1,1), _tmp.eps, _tmp.elementwise_affine)


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        in_channels: input channels
    """

    def __init__(self, architecture, in_channels=2):
        super().__init__()

        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        if (architecture.startswith('regnet')): # Rename modules so we can use the same code
            self._model.conv1 = self._model.stem.conv
            self._model.bn1  = self._model.stem.bn
            self._model.act1 = nn.Sequential()
            self._model.maxpool =  nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()

        elif (architecture.startswith('convnext')):
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules['0']
            self._model.layer2 = self._model.stages._modules['1']
            self._model.layer3 = self._model.stages._modules['2']
            self._model.layer4 = self._model.stages._modules['3']
            self._model.global_pool = self._model.head
            self._model.global_pool.flatten = nn.Sequential()
            self._model.global_pool.fc = nn.Sequential()
            self._model.head = nn.Sequential()
            _tmp = self._model.global_pool.norm
            self._model.global_pool.norm = nn.LayerNorm((self.config.perception_output_features,1,1), _tmp.eps, _tmp.elementwise_affine)

        # Change the first conv layer so that it matches the amount of channels in the LiDAR
        # Timm might be able to do this automatically
        _tmp = self._model.conv1
        use_bias = (_tmp.bias != None)
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=use_bias)
        # Need to delete the old conv_layer to avoid unused parameters
        del _tmp
        del self._model.stem.conv
        torch.cuda.empty_cache()
        if(use_bias):
            self._model.conv1.bias = _tmp.bias

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
# our code start from here
