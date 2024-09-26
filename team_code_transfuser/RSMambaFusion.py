import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_, to_2tuple
from einops import rearrange
import numpy as np
from torch.utils.checkpoint import checkpoint
from convmodule import ConvModule
from torch.nn import SyncBatchNorm as _SyncBatchNorm
from functools import partial
from typing import Callable, Any

from RSMamba import OSSBlock, OSSM, RSM_SS

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


# make this task for connection to lidarnet of transfuser.
class MambaBlock(nn.Module):
    def __init__(
        self, 
        # =============================
        # mamba
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
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
        norm_layer="LN",
        use_checkpoint=False,  
        **kwargs,     
    ):
        
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims*2**i_layer) for i_layer in range(self.num_layers)]

    # mamba opponent
    # output shape: (B, num_classes, H, W)
        self.model = RSM_SS(
            patch_size=patch_size,
            in_chans=in_chans, 
            num_classes=num_classes, 
            depths=depths, 
            dims=dims, 
            # =========================
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,        
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate, 
            ssm_init=ssm_init,
            forward_type=forward_type,
            # =========================
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            # =========================
            drop_path_rate=drop_path_rate, 
            patch_norm=patch_norm, 
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            )
    #patch_embeding
    @staticmethod
    def patch_embedding(self, image_tensor, lidar_tensor, velocity, patch_norm, in_channels = 3, embed_dim = 96, patch_size = 4, norm_layer = nn.LayerNorm):
        assert patch_size ==4
        B = lidar_tensor.shape[0]
        image_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size = 3, stride = 2, padding = 1), #(B,embed_dim//2, H/2, W/2)
            (Permute(0,2,3,1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim//2) if patch_norm else nn.Identity()),
            (Permute(0,3,1,2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size = 3, stride = 2, padding = 1), # (B, embed_dim, H/4, W/4)
            Permute(0,2,3,1), # (B, H/4, W,4, embed_dim)
            (norm_layer(embed_dim) if patch_norm else nn.Identity())
        )
        lidar_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size = 3, stride = 2, padding = 1), #(B,embed_dim//2, H/2, W/2)
            (Permute(0,2,3,1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim//2) if patch_norm else nn.Identity()),
            (Permute(0,3,1,2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size = 3, stride = 2, padding = 1), # (B, embed_dim, H/4, W/4)
            Permute(0,2,3,1), # (B, H/4, W,4, embed_dim)
            (norm_layer(embed_dim) if patch_norm else nn.Identity())
        )
        
        velocity_embedding = nn.Linear(1, embed_dim)

        image_embed = image_embedding(image_tensor)
        lidar_embed = lidar_embedding(lidar_tensor)
        velocity_embed = velocity_embedding(velocity)
        # reshape to (B, -1, 96)
        image_embed = image_embed.view(B, -1, embed_dim).contiguous()
        lidar_embed = lidar_embed.view(B, -1, embed_dim).contiguous()
        
        #integrate into token_embedding
        token_embeddings = torch.cat((image_embed, lidar_embed), dim = 1) + velocity_embed.unsqueeze(1)

        token_embeddings = token_embeddings.reshape(B, 144,224,embed_dim).permute(0,3,1,2).contiguous()


        return token_embeddings
    
    def patch_departing(self, token_embeddings, image_tensor, lidar_tensor):

        B1, C, H1, W1 = image_tensor.shape
        assert B1 == B*4

        B, C, H2, W2 = lidar_tensor.shape
        B, C1, _, _ = token_embeddings.shape

        image_decode = nn.Sequential(
            nn.Upsample(scale_factor= 4, mode = 'nearest'),            
            nn.Conv2d(C1,C,kernel_size= 1, padding= 0, bias = False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace = True),
        )

        token_embeddings = token_embeddings.permute(0,2,3,1).view(B,-1,self.num_classes).contiguous().view(B, 4*H1/4*W1/4 + H2/4*W2/4, self.num_classes)
        image_tensor_out = token_embeddings[:, :4*H1/4*W1/4, :].view(B*4, H1/4, W1/4, self.num_classes).permute(0,3,1,2).contiguous()
        lidar_tensor_out = token_embeddings[:, 4*H1/4*W1/4:, :].view(B, H2/4, W2/4, self.num_classes).permute(0,3,1,2).contiguous()

        image_tensor_out = image_decode(image_tensor_out)
        lidar_tensor_out = image_decode(lidar_tensor_out)
        

        return image_tensor_out, lidar_tensor

    def forward(self, image_tensor, lidar_tensor,velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        token_embeddings = self.patch_embedding(image_tensor, lidar_tensor, velocity)
        # the wrong point here is we already patch_embedding in rsmamba, that mean we only need to merge image and lidar and put into model, patch embeding might be use latter in model
        # patch departing can be use latter one, but need only change (no need 96 dim to 96 dim)

        output = self.model(token_embeddings)

        image_tensor_out, lidar_tensor_out = self.patch_departing(output, image_tensor, lidar_tensor)


        return image_tensor_out, lidar_tensor_out # (B1, num_classes, H1, W1) (B, num_classes, H2, W2)

        
