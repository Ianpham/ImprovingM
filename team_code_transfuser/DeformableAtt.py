import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_, to_2tuple
from einops import rearrange
import numpy as np
from torch.utils.checkpoint import checkpoint
from convmodule import ConvModule
from torch.nn import SyncBatchNorm as _SyncBatchNorm
from einops import rearrange

    
# the idea here that we can go with window attention shifting to deformable window attention, attached it model into swin v2 to experimental
# clarify impact of all feature map or window sliding and extract deformable point (which might be suitbale for multiple task/depth and object inside the picture)
# consider deformable between q,k,v not on same size or position of window_size 
# consider the size of window, in which small of window size, which  require small receptive field where we reduce deformable point or vice versa.

# follow same pattern or make it chaos to increase ability of model

# remember that we can use current Dat++ for depth estimation due to level of detail
# however when to come segmentation and object detection, consider to reduce window size based on what requires

# apply window attention in deformable, remove relative_position_index
# we dont need to inherit window attetion to deformableattention due to it matching different style of relative position
class DeformableAtt(nn.Module):
    def __init__(self,
                 dim,
                #  q_size,
                 n_heads,               
                 patch_size,
                 window_size,
                 stride,
                 attn_drop,
                 proj_drop,                 
                 offset_range_factor,
                 log_cpb = True, 
                 use_conv_patches = False):
        
        super().__init__()
        self.dim = dim

        # layer shall be start from dims_stem to dims
        # equal to dim_stem=96, dims=[96, 192, 384, 768]
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, patch_size // 2, 1),
            LayerNormProxy(dim)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim, patch_size, patch_size, 0),
            LayerNormProxy(dim)
        )
        # consider that image_size will be divide by patch_size
        # attention components
        self.n_heads = n_heads
        # self.n_groups = n_groups
        self.proj_q = nn.Conv2d(dim, dim, 1,1,0) # dim in this case must be C, or if it is embeded, then x should be treated other way to make sure this can be right in input and output
        self.proj_k = nn.Conv2d(dim, dim, 1,1,0)
        self.proj_v = nn.Conv2d(dim, dim, 1,1,0)
        self.n_head_channels = self.dim //self.n_heads
        # self.n_group_channels = self.dim // self.n_groups
        # self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.attn_drop = nn.Dropout(p = attn_drop)
        self.proj_drop = nn.Dropout(p = proj_drop)

        # window components
        self.window_size = window_size
        # let say the flow in final 
        # (B, C, H, W) -> (B, C, H // self.window_size, window_size, W// self.window_size, window_size)
        # -> (B, C*  H // self.window_size * W // self.window_size, self. window_size , self.window_size) (still in X)
        #  for offset 
        # -> (B * n_groups, C // n_groups * H//self.window_size *W//self.window size, kk, kk//stride)
        # C // n_groups * H//self.window_size *W//self.window size = self.n_group_channels
        # in this case self.rpe_table will be with dimension (C*  H // self.window_size * W // self.window_size)
        # deformable components
        # is dim of window attention same with deformable attention
        # deformable components
        # self.ksize = kk # the kk size of offset we can do with window_size as well
        pad = self.window_size //2 if self.window_size != stride else 0
        self.stride = stride # consider to make exchange on stride using window size, conduct the change through each window
        # or stride must be small than window_size due to x_sampled process
        self.pad  = pad
        # conditional position encoding
        self.log_cpb = log_cpb
        self.offset_range_factor = offset_range_factor

        # q_size input on q
        # self.q_size = q_size # fmap -> img_size = 224 # we need it here, there will be no other thing like dwc_pe, fixed_pe or log+cpb, else from line 307
        # No: use_pe, dwc_pe, fixed_pe, no_off, use_conv_patch
        # self.q_h, self.q_w = self.q_size
        # self.kv_h, self.kv_w = self.q_h // self.stride, self.q_w //  self.stride # why do we need q_size here, it only support self.fixed_pe = True
        # due the paper, The last row reports the results of our proposed
        # deformable relative positional bias, showing strong overall
        # performance on various tasks, indicating that DeformRPB is
        # the best suitable approach to exploiting geometric informa-
        # tion in the scheme of deformable attention.
        # represent rpe table, need to consider rpe table constructed from multiple small in window square

                    
    @torch.no_grad()
    def _get_ref_point(self, h, w, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype= dtype, device=device),
            torch.linspace(0.5, w - 0.5, B, dtype= dtype, device=device),
            indexing= 'ij',
        )

        ref = torch.stack([ref_y, ref_x], dim = -1)
        ref[..., 1].div_(w - 1).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(h -1).mul_(2.0).sub_(1.0)

        ref = ref.unsqueeze(0).repeat(B*self.n_groups, -1,-1,-1)

    @torch.no_grad()
    def _get_q_grids(self, h, w, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, h , h, dtype= dtype, device=device),
            torch.linspace(0, w , B, dtype= dtype, device=device),
            indexing= 'ij',
        )

        ref = torch.stack([ref_y, ref_x], dim = -1)
        ref[..., 1].div_(w - 1).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(h -1).mul_(2.0).sub_(1.0)

        ref = ref.unsqueeze(0).repeat(B*self.n_groups, -1,-1,-1)       

    def forward(self,x, mask = None):
        B, C, H, W =  x.shape # c is dim here after patch embeding
        dtype, device = x.dtype, x.device
        
        # deformable part # consider that part
        # let say dim = 1024, n_group at least 14^2, for this kind of mindset will not be pratical
        self.n_groups = H // self.window_size * W//self.window_size # number windows in frame of picture (equal 1024 square, impossible)
        self.n_group_channels = self.dim // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups # next task is considering the n_groups (we are in very number while the original are just [2,4,8,16])
        if self.log_cpb:
            self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
        self.conv_offset = nn.ModuleList(
        nn.Conv2d(self.n_group_channels, self.n_group_channels, kernel_size= self.window_size,stride = self.stride, padding_mode= self.pad, groups = self.n_group_channels),
        LayerNormProxy(self.n_group_channels),
        nn.GELU(),
        nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        x = self.patch_embedding(x)	 
        q = self.proj_q(x)
        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', c = self.n_group_channels, g = self.n_groups)
        offset = self.conv_offset(q_off).contiguous() # B*g, 2, H #change by stride as factor ratio
        Hk, Wk = offset.size(2), offset.size(3) # Hk = H//self.ksize, wk = W//self.ksize
        n_sample = Hk*Wk # thinkback and make sure n_sample shall be size of window_size square

        if self.offset_range_factor >= 0:
            offset_range = torch.tensor([1.0/(Hk - 1), 1.0/(Wk - 1)], device = device).reshape(1,2,1,1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        
        offset = rearrange(offset, 'b c h w -> b h w c')
        reference = self._get_ref_point(Hk, Wk, B, dtype, device)
        if self.offset_range_factor >=0:
            pos = offset + reference
        else:
            pos  = (offset + reference).clamp_(-1., 1.)
        # sampling
        x_sampled = F.grid_sample(
            input = x.reshape(B*self.n_groups, self.n_group_channels, H, W),
            grid = pos[..., (1,0)], # y, x -> x, y
            mode = 'bilinear', align_corners= True) # (B*g, Cg, Hg, Wg)       
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)           
	
        # attention matrix
        q = q.reshape(B*self.n_heads, self.n_head_channels, H*W)
        k = self.proj_k(x_sampled).reshape(B*self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B*self.n_heads, self.n_head_channels, n_sample)
        
        attn = torch.einsum('b c m, b c n -> b m n', q,k)
        attn = attn.mul(self.scale)   
        if self.log_cpb:
            q_grid = self._get_q_grid(H,W,B, dtype, device)     
            displacement = (q_grid.reshape(*self.n_groups, H*W,2).unsqueeze(2)- pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(4.0) # d_y, d_x [-8, +8])
            displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) +1.0)/np.log2(8.0)
            attn_bias = self.rpe_table(displacement)
            attn = attn + rearrange(attn_bias, 'b m n h -> (b h) m n ', h = self.n_group_heads)

        if mask is not None:
        # attn : (b * nW) h w w
        # mask : nW ww ww
            nW,w, w, _ = mask.size()
            attn = rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.n_heads, w1=w, w2=w) + mask.reshape(1, nW, 1, w, w)
            attn = rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')        
        attn = F.softmax(attn, dim = 2)       	
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)       
        
        out = out.reshape(B,C,H,W)
        y = self.proj_drop(self.proj_out(out))
        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
    


        
class ShiftedWindow(DeformableWindow):
    def __init__(self, dim,q_size, n_heads, patch_size, window_size, stride, attn_drop, proj_drop, shift_size):
        super().__init__(dim,q_size, n_heads, patch_size, window_size, stride, attn_drop, proj_drop)
        # DeformableWindow(self,
        #          dim,
        #          q_size,
        #          n_heads,
        #          patch_size
        #          window_size,
        #          stride,
        #          attn_drop,
        #          proj_drop,
        #          no_off, dwc_pe, use_pe, fixed_pe, log_cpb,
        #          offset_range_factor,
        #          use_conv_patches)
        self.fmap_size = to_2tuple(q_size)
        self.shift_size = shift_size
        
        assert 0 <= self.shift_size <= self.window_size

        img_mask = torch.zeros(*self.q_size)
        h_slices = (slice(0,-self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0,-self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        
        cnt = 0

        for h in h_slices:
            for w in w_slices:
                img_mask[h,w] = cnt
                cnt += 1

        mask_windows = rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1 = self.window_size, w1 = self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0))
        self.attn_mask = attn_mask

    def forward(self, x):
        # convert x to window shape here 
        B, C, H, W = x.shape

        x = x.view(B, C, self.window_size, ...)
        shifted_x = torch.roll(x, shifts = (-self.window_size, -self.window_size), dim = (2,3))
        # that mean when we do forward in deformable window, the shifted x, shall be related in dimension.
        # mean input size shall be contain window. change from this part to control dim inside
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x= torch.roll(sw_x, shifts= (self.window_size, self.window_size), dim = (2,3))

        return x, None, None
    
class SingleShiftedWindow(nn.Module):
    def __init__(self, dim, q_size, n_heads, patch_size, window_sizes,stride,shift_size, n_ffw = 2,
                 proj_drop = 0.,
                 ffw_drop = 0.,
                 attn_drop = 0.,
                 drop_path = 0.,
                 act_func = nn.GELU(),
                 norm_func = LayerNormProxy(),
                ):
        #dim,q_size, n_heads, patch_size, window_size, stride, attn_drop, proj_drop, shift_size
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_size = q_size
        self.n_ffw = n_ffw
        self.window_sizes = window_sizes
        self.norm1 = norm_func(dim)
        self.norm2 = norm_func(dim)
        self.attn = ShiftedWindow(dim, q_size, n_heads, window_sizes, ffw_drop, attn_drop, proj_drop, act_func)
         # construct FFN layer
         # we will set dim = dim*2 with H, W decrease by two for each round
        in_channels = self.dim
        for _ in range(self.n_ffw -1):
            layers = nn.Sequential(nn.Linear(in_channels, ffw_dims,bias = True),
                                 nn.ReLU(),
                                 nn.Dropout(ffw_drop),
                                 )
            in_channels = ffw_dims
        layers.append(nn.Linear(ffw_dims, dim))
        layers.append(nn.Dropout(ffw_drop))
        self.layers = nn.Sequential(*layers)
        self.drop_layer = DropPath(drop_path)
        self.layer_scale = LayerScale(dim)
        self.ffn = nn.Sequential(self.layers,
                                 self.layer_scale,
                                 self.drop_layer)
    def forward(self, x):
       identity = x
       x = self.norm(x)
       x = self.attn(x)
       x = x + identity
       identity = x
       x = self.norm2(x)
       x = self.ffn(x)
       x = x + identity
       return x
        
# let summarize here, we dont/need need singled shifted window, second, we need local window, third, the final FPN need to be consider whether we need multishifted windown or multihead
class MultiShiftedWindow(nn.Module):
    def __init__(self, 
                 dim,
                 n_heads,
                 ffw_dims,
                 n_ffw = 2,
                 window_sizes = [5,7,9,12],                
                 proj_drop = 0.,
                 ffw_drop = 0.,
                 attn_drop = 0.,
                 drop_path = 0.,
                 layer_scale_init_value=0.,
                 ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.ffw_dims = ffw_dims
        self.n_ffw = n_ffw
        self.window_sizes = window_sizes
        self.attns = nn.ModuleList()
        # please set each layers for later extract with lidar and image (we will have 4 layers here)
        for window_size in window_sizes:
            # please check ffw_dims is img_dim or fmap_dims in this situation (this fmap_size = 224)
            self.attns.append(ShiftedWindow(dim,ffw_dims, n_heads,window_size, shift_size= 0, attn_drop= attn_drop, proj_drop= proj_drop))
            self.attns.append(ShiftedWindow(dim,ffw_dims, n_heads,window_size, shift_size= window_size//2, attn_drop= attn_drop, proj_drop= proj_drop))
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.reduce = nn.Linear(dim*2*len(window_size), dim, bias = False)

        # construct FFN layer
        in_channels = self.dim
        for _ in range(self.n_ffw -1):
            layers = nn.Sequential(nn.Linear(in_channels, ffw_dims,bias = True),
                                 nn.ReLU(),
                                 nn.Dropout(ffw_drop),
                                 )
            in_channels = ffw_dims
        layers.append(nn.Linear(ffw_dims, dim))
        layers.append(nn.Dropout(ffw_drop))
        self.layers = nn.Sequential(*layers)
        self.drop_layer = DropPath(drop_path)
        self.layer_scale = LayerScale(dim)
        self.ffn = nn.Sequential(self.layers,
                                 self.layer_scale,
                                 self.drop_layer)

    def forward(self,x, hw_shape):
        identity = x
        x = self.norm1(x)
        x_w = [checkpoint(attn, x, hw_shape) + identity for attn in self.attns]
        x = torch.cat(x_w, dim = -1)
        x = self.reduce(x)
        identity = x
        x = self.norm2(x)
        x = self.ffn(x)
        out = identity + x
        return out

class MultiHead(nn.Module):
    # using for depth and semantic decoder => this night, learn about decoder transfomer
    def __init__(self,
                 dim,
                 n_heads,
                 in_channels,
                 channels,
                 ffw_dims,
                 n_ffw = 2,
                 window_sizes = [5,7,9,12],
                 align_corners=False,
                 n_lateral_layers = 2,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.channels = channels
        self.n_ffw = n_ffw
        self.window_sizes = window_sizes
        self.ffw_dims = ffw_dims
        self.n_lateral_layers = n_lateral_layers

        assert len(self.in_channels) == self.n_lateral_layers
        self.align_corner = align_corners
        
        self.lateral_convs = nn.ModuleList()
        self.single_swins = nn.ModuleList()
        self.l_conv4 = ConvModule(self.in_channels[-1], self.channels,
                                kk_size = 1,
                                norm_func = _SyncBatchNorm,
                                act_func= nn.ReLU(),
                                requred_grad = True,
                                inplace = False)
        for in_channel in self.in_channels[:-1]:
            l_conv = ConvModule(
                in_channel,
                self.channels,
                kk_size= 1,
                norm_func = _SyncBatchNorm,
                act_func= nn.ReLU(),
                requred_grad = True,
                in_place = True,
            )
            self.lateral_convs.append(l_conv)
            single_window = SingleShiftedWindow(self.channels, self.n_heads, self.channels, n_ffw = 2, window_sizes = self.window_sizes[:-1],)
            self.fpn_swins.append(single_window)

        self.mutil_swin = MultiShiftedWindow(
            self.channels,
            n_heads,
            self.ffw_dims,
            self.n_ffw,
            self.window_sizes
        )
        self.out = nn.Conv2d(self.channels, self.channels, kernel_size = 1)

    def transform_input(self, input):
        input = [input[i] for i in self.n_lateral_layers]
        upsample_input = [F.interpolate(x, input[0].shape[2:], mode = 'bilinear', align_corners = self.align_corner) for x in input]

        inputs =  torch.cat(upsample_input, dim = 1)

        return inputs
    def forward(self, x):
        # building multi head for decoder
        x = self.transform_input(x)
        # build lateral layers
        laterals = [
            l(x[i]) for i,l in enumerate(self.lateral_convs) 
        ]
        laterals. append(self.l_conv4(x[-1]))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] += F.interpolate(laterals[i], size = prev_shape, mode = 'bilinear', align_corners = self.align_corner)

        #build fpn output
        fpn_outs = []
        for i in range(used_backbone_levels -1):
            B, C, H, W = laterals[i].shape
            x = laterals[i].view(B,C,H*W).transpose(2,1)
            x = checkpoint(self.single_swins[i], x, (H,W))
            x = x.permute(0,2,1).view(B, self.channels,H, W)
            fpn_outs.append(x)
        fpn_outs.append(laterals[-1])
        # fuse lateral features
        for i in range(used_backbone_levels - 10,-1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], fpn_outs[0].shape[2:], mode = 'bilinear', align_corners = self.align_corner)

        fpn_out = torch.stack(fpn_outs, dim = 0).sum(dim = 0)
        # apply to head
        B, C, H, W = fpn_out.shape
        fpn_out = fpn_out.view(B,C,H*W).transpose(2,1)
        output = self.mutil_swin(fpn_out, (H,W))
        output = output.permute(0,2,1).view(B, self.channels, H, W)
        output = self.out(output)
        return output

# task, connect as lateral for small window to big window.
class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self,x):
        x = rearrange(x,'b c h m  -> b h m c')
        x = self.norm(x)
        x = rearrange(x,'b h m c -> b c h m')

        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class LayerScale(nn.Module):
    def __init__(self,in_channels, inplace: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.inplace = inplace
        self.scale = nn.init.xavier_normal_(nn.Parameter(torch.ones(in_channels)))
    def forward(self,x):
        if self.inplace:
            return x.mul_(self.scale)
        else:
            return x*self.scale

























        