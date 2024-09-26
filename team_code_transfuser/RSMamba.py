#borrow from https://github.com/MzeroMiko/VMamba


import os
import time
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from batchnorm import SynchronizedBatchNorm2d

# import selective scan, 
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
try:
    import selective_scan_cuda_core
except Exception as e:
    ...

try:
    import selective_scan_cuda
except Exception as e:
    ...

# fvcore flops
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
class SelectiveScanMamba(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        # all in float
        # if u.stride(-1) != 1:
        #     u = u.contiguous()
        # if delta.stride(-1) != 1:
        #     delta = delta.contiguous()
        # if D is not None and D.stride(-1) != 1:
        #     D = D.contiguous()
        # if B.stride(-1) != 1:
        #     B = B.contiguous()
        # if C.stride(-1) != 1:
        #     C = C.contiguous()
        # if B.dim() == 3:
        #     B = B.unsqueeze(dim=1)
        #     ctx.squeeze_B = True
        # if C.dim() == 3:
        #     C = C.unsqueeze(dim=1)
        #     ctx.squeeze_C = True
        
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        # dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        # dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanFake(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        x = delta
        out = u
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias = u * 0, delta * 0, A * 0, B * 0, C * 0, C * 0, (D * 0 if D else None), (delta_bias * 0 if delta_bias else None)
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)
def gather_by_angle(tensor, angle):
    B, C, H, W = tensor.size()
    rad_angle = math.radians(angle)

    # step sizes in x and y
    step_x = math.cos(rad_angle)
    step_y = math.sin(rad_angle)
    # create grid of indice
    indices_x = torch.arange(0,W, device = tensor.device)
    indices_y = torch.arange(0,H, device = tensor.device)
    grid_x, grid_y = torch.meshgrid(indices_x, indices_y, indexing = 'xy')
    
    # starting position
    start_x = (grid_y * step_x).round().long()
    start_y = (grid_y * step_y).round().long()

    # create the gathering indices
    gather_indices = (start_x.unsqueeze(-1) + indices_x).clamp(0,W-1)
    gather_indices = gather_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

    # gatehr the elements along the specified angle
    gathered = tensor.gather(3, gather_indices)

    return gathered.transpose(-1,-2).reshape(B,C,H*W)

def scatter_by_angle(tensor_flat, original_shape, angle):
    B, C, H, W = original_shape
    rad_angle = math.radians(angle)
    
    # Compute the step sizes in the x and y directions based on the angle
    step_x = math.cos(rad_angle)
    step_y = math.sin(rad_angle)
    
    # Create a grid of indices for scattering
    indices_x = torch.arange(0, W, device=tensor_flat.device)
    indices_y = torch.arange(0, H, device=tensor_flat.device)
    grid_x, grid_y = torch.meshgrid(indices_x, indices_y, indexing='xy')
    
    # Compute the starting positions for each row
    start_x = (grid_y * step_x).round().long()
    start_y = (grid_y * step_y).round().long()
    
    # Create the scattering indices
    scatter_indices = (start_x.unsqueeze(-1) + indices_x).clamp(0, W-1)
    scatter_indices = scatter_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    
    # Create an empty tensor to store the scattered result
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    
    # Scatter the flattened tensor back to the original shape
    result_tensor.scatter_(3, scatter_indices, tensor_flat.reshape(B, C, H, W))
    
    return result_tensor

class SelectiveDirection(nn.Module):
    def __init__(
        self,
        in_channels,
        n_groups,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_groups = n_groups
        self.n_group_channels = self.in_channels // self.n_groups
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1, groups=self.n_groups),
            nn.GELU(),
            nn.Conv2d(self.in_channels, 2, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Divide the image into 9 sub-sections
        sub_h, sub_w = H // 3, W // 3
        sub_sections = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue  # Skip the center sub-section
                sub_section = x[:, :, i*sub_h:(i+1)*sub_h, j*sub_w:(j+1)*sub_w]
                sub_sections.append(sub_section)

        # Compute attention points for each sub-section
        points = []
        for sub_section in sub_sections:
            sub_section_points = self.conv_offset(sub_section)
            sub_section_points = sub_section_points.mean(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
            points.append(sub_section_points)

        points = torch.cat(points, dim=1)
        points = points.reshape(B, 8, 2, 1, 1)
        print(points.shape)
        # Compute angles from the center point to the learned points
        angles = self.create_angle(points)

        return points, angles

    def create_angle(self, points):
        B, _, _, _, _ = points.shape
        center_x, center_y = 0.5, 0.5  # Assuming normalized coordinates

        angles = []
        for i in range(B):
            batch_points = points[i].squeeze()  # Shape: (8, 2)
            batch_angles = []
            for j in range(8):
                point_x, point_y = batch_points[j]
                angle = math.atan2(point_y - center_y, point_x - center_x)
                batch_angles.append(angle)
            angles.append(batch_angles)

        angles = torch.tensor(angles, device=points.device)
        angles = angles.reshape(B, 8, 1, 1, 1)

        return angles

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 8, C, H*W))
        angles = SelectiveDirection(in_channels=C, n_groups=8)(x)[1]
        for i in range(8):
            xs[:, i, :, :] = gather_by_angle(x,angles[i])
        return xs
    @staticmethod
    def backward(ctx, ys:torch.Tensor):
        B, C, H, W = ctx.shape        
        angles = SelectiveDirection(in_channels=C, n_groups=8)(ys)[1]
        for i in range(8):
            ys += scatter_by_angle(ys[:,i], (B, C, H, W), angles[i])
        return ys

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H,W)
        ys = ys.view(B, K, D, -1)
        # recheck it
        angles = SelectiveDirection(in_channels=D, n_groups=8)(ys)[1]
        for i in range(8):
            ys += scatter_by_angle(ys[:,i], (B, D, H, W), angles[i])
        
        return ys.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):

        H, W = ctx.shape
        B, C, L = x.shape

        xs = x.new_empty((B, 8, C, L))
        angles = SelectiveDirection(in_channels=C, n_groups=8)(x)[1]
        for i in range(8):
            xs[:, i, :, :] = gather_by_angle(x,angles[i])

        return xs.view(B,8,C,H,W)

class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C, H*W))
        angles = SelectiveDirection(in_channels=C, n_groups=2)(x)[1]
        for i in range(2):
            xs[:, i, :, :] = gather_by_angle(x, angles[i])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        angles = SelectiveDirection(in_channels=C, n_groups=2)(ys)[1]
        for i in range(2):
            ys += scatter_by_angle(ys[:, i], (B, C, H, W), angles[i])
        return ys

class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        angles = SelectiveDirection(in_channels=D, n_groups=2)(ys)[1]
        for i in range(2):
            ys += scatter_by_angle(ys[:, i], (B, D, H, W), angles[i])
        return ys.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        angles = SelectiveDirection(in_channels=C, n_groups=2)(x)[1]
        for i in range(2):
            xs[:, i, :, :] = gather_by_angle(x, angles[i])
        return xs.view(B, 2, C, H, W)

class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1).contiguous()
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        y = ys.sum(dim=1).view(B, C, H, W)
        return y


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        y = ys.sum(dim=1).view(B, D, H * W)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.view(B, 1, C, L).repeat(1, 4, 1, 1).contiguous().view(B, 4, C, H, W)
        return xs

def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    out_norm_shape="v0",
    # ==============================
    to_dtype=True, # True: final out to dtype
    force_fp32=False, # True: input fp32
    # ==============================
    nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
    # ==============================
    SelectiveScan=None,
    CrossScan=CrossScan,
    CrossMerge=CrossMerge,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
        
    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    xs = CrossScan.apply(x)
    
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    # ZSJ 这里把矩阵拆分成不同方向的序列，并进行扫描
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    # ZSJ 这里把处理之后的序列融合起来，并还原回原来的矩阵形式
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]: # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
    else: # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class OSSM(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v0=self.forward_corev0,
            # v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=partial(
                cross_selective_scan, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            )),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=partial(
                cross_selective_scan, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            )),
            # ===============================
            fake=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanFake),
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
        )
        if forward_type.startswith("debug"):
            from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations, cross_selective_scanv2
            FORWARD_TYPES.update(dict(
                debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
                debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
                debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16, self),
                debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs, self),
                debugforward_core_mambassm_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
                debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm, self),
                debugforward_core_sscore_fusecscm_fwdnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fwdnrow, self),
                debugforward_core_sscore_fusecscm_bwdnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_bwdnrow, self),
                debugforward_core_sscore_fusecscm_fbnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fbnrow, self),
                debugforward_core_ssoflex_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm, self),
                debugforward_core_ssoflex_fusecscm_i16o32=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
                debugscan_sharessm=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scanv2),
            ))
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        # ZSJ k_group 指的是扫描的方向
        # k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1
        k_group = 8 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
    
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        # ZSJ 这里进行data expand操作，也就是把相同的数据在不同方向展开成一维，并拼接起来,但是这个函数只用在旧版本
        # 把横向和竖向拼接在K维度
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # torch.flip把横向和竖向两个方向都进行反向操作
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)
    
    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scan, force_fp32=None):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        # ZSJ V2版本使用的mamba，要改扫描方向在这里改
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
        )
        return x
    
    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if with_dconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=with_dconv)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        try:
            from ss2d_ablations import SS2DDev
            _OSSM = SS2DDev if forward_type.startswith("dev") else OSSM
        except:
            _OSSM = OSSM

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _OSSM(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path(self.norm(self.op(input)))
            else:
                x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class Decoder_Block(nn.Module):
    """Basic block in decoder."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output



class RSM_SS(nn.Module):
    def __init__(
        self, 
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
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = self._make_patch_embed_v2
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = self._make_downsample_v3

        # self.encoder_layers = [nn.ModuleList()] * self.num_layers
        self.encoder_layers = []
        self.decoder_layers = []

        for i_layer in range(self.num_layers):
            # downsample = _make_downsample(
            #     self.dims[i_layer], 
            #     self.dims[i_layer + 1], 
            #     norm_layer=norm_layer,
            # ) if (i_layer < self.num_layers - 1) else nn.Identity()

            downsample = _make_downsample(
                self.dims[i_layer - 1], 
                self.dims[i_layer], 
                norm_layer=norm_layer,
            ) if (i_layer != 0) else nn.Identity()  # ZSJ 修改为i_layer != 0，也就是第一层不下采样，和论文的图保持一致，也方便我取出每个尺度处理好的特征

            self.encoder_layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))
            if i_layer != 0:
                self.decoder_layers.append(Decoder_Block(in_channel=self.dims[i_layer], out_channel=self.dims[i_layer-1]))

        self.encoder_block1, self.encoder_block2, self.encoder_block3, self.encoder_block4 = self.encoder_layers
        self.deocder_block1, self.deocder_block2, self.deocder_block3 = self.decoder_layers

        
        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[0]//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dims[0]//2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.dims[0]//2, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_seg = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

        self.apply(self._init_weights)

        #S2FPN 
        self.fab = nn.Sequential(
            conv_block(
                self.dims[3],
                self.dims[3]//2,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                groups = self.dims[3]//2,
                dilation = 1,
                bn_act = True
            ),
            nn.Dropout(p = 0.15)
        )
        self.cfgb = nn.Sequential(
        conv_block(self.dims[3],
                    self.dims[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    group=self.dims[3],
                    dilation=1,
                    bn_act=True),
        nn.Dropout(p=0.15))
        # global feature
        self.gfu3 = GlobalFeatureUpsample(self.dims[2], self.dims[3], self.dims[2])
        self.gfu2 = GlobalFeatureUpsample(self.dims[2], self.dims[3], self.dims[2])
        self.gfu2 = GlobalFeatureUpsample(self.dims[2], self.dims[3], self.dims[2])

        # attention pyramid
        self.apf2 = PyramidFusionNet(self.dims[3], self.dims[3], self.dims[2])
        self.apf3 = PyramidFusionNet(self.dims[2], self.dims[2], self.dims[1])
        self.apf4 = PyramidFusionNet(self.dims[1], self.dims[1], self.dims[0])

        # seghead
        self.seghead = SegHead(self.dims[0], num_classes) # consider in both case num_classes and 3 for specific purpose
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(OSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            # ZSJ 把downsample放到前面来，方便我取出encoder中每个尺度处理好的图像，而不是刚刚下采样完的图像
            downsample=downsample,
            blocks=nn.Sequential(*blocks,),
        ))


    def forward(self, x1: torch.Tensor):
        B, C, H, W = x1.shape
        x1 = self.patch_embed(x1)

        x1_1 = self.encoder_block1(x1)  # (B, h/4, w/4, dims[0])
        x1_2 = self.encoder_block2(x1_1) # (B,h/8, w/8, dims[1])
        x1_3 = self.encoder_block3(x1_2) # (B, h/16, w/16, dims[2])
        x1_4 = self.encoder_block4(x1_3)  # (B, h/32, w/32, dims[1])

        x1_1 = rearrange(x1_1, "b h w c -> b c h w").contiguous() # (B, dims[0], h/4, w/4)
        x1_2 = rearrange(x1_2, "b h w c -> b c h w").contiguous() # (B, dims[1], h/8, w/8)
        x1_3 = rearrange(x1_3, "b h w c -> b c h w").contiguous() # (B, dims[2], h/16, w/16)
        x1_4 = rearrange(x1_4, "b h w c -> b c h w").contiguous() # (B, dims[3], h/32, w/32)

        CFGB = self.cfgb(x1_4) # (B, dims[3], h/64, w/64)

        ADF1 = self.apf2(CFGB, x1_4) # (B, dims[3], h/32, w/32)
        ADF2 = self.apf3(ADF1, x1_3) #  (B, dims[2], h/16, w/16) 
        ADF3 = self.apf4(ADF2, x1_2) # (B, dims[1], h/8, w/8)

        FAB = self.fab(x1_4) # (B, dims[3]//2, h/64, w/64) = dims[2]

        dec4 = self.gfu3(ADF1,FAB) #(B, dims[2], h/32, w/32)
        dec3 = self.gfu2(ADF2,dec4) # (B, dims[1], h/16, w/16)
        dec2 = self.gfu(ADF3, dec3) # (B, dims[0], h/8, w/8)

        seghead = F.interpolate(seghead, (H, W), mode="bilinear", align_corners=True)

        seghead = self.seghead(dec2)

        return seghead

class conv_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation = (1,1),
        group = 1,
        bn_act = False,
        bias = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels,kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = group, bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PRelu(out_channels)
        self.use_bn_act = bn_act

    def forward(self,x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)
class DropPath(nn.Module):
    def __init__(
        self,
        drop_path = None
    ):
        super().__init__()
        self.drop_path = drop_path

    def forward(self,x):
        return self.drop_path(x, self.drop_path, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features,out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.dconv1 = conv_block(in_features, in_features//2, kernel_size=3, stride=1, padding=2, dilation=2, bn_act=True)
        self.dconv2 = conv_block(in_features, in_features//2, kernel_size=3, stride=1, padding=4, dilation=4, bn_act=True)
        self.fuse = conv_block(in_features, out_features, 1,1,0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        d1 = self.dconv1(x)
        d2 = self.dconv2(x)
        dd = torch.cat([d1,d2],1)
        x = self.fuse(dd)
        x = torch.sigmoid(x)
        x = self.drop(x)
        return x
    
class ScaleAwareStripAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        drop_rate = 0.15
    ):
        super().__init__()
        self.conv_sh = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1, padding = 0)
        self.bn_sh1 = nn.BatchNorm2d(in_channels)
        self.bn_sh2 = nn.BatchNorm2d(in_channels)

        self.augment_conv = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1, padding = 0)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride =1, padding = 0)
        self.conv_res = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1, padding = 0)

        self.drop = drop_rate
        self.fuse = conv_block(in_channels, in_channels, kernel_size = 1, stride = 1, padding = 0, bn_act = False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        b,c,h,w = x.shape

        mxpool = F.max_pool2d(x, [h,1])
        mxpool = F.conv2d(mxpool, self.conv_sh.weight, padding = 0, dilation = 1)
        mxpool = self.bn_sh1(mxpool)
        mxpool_v = mxpool.view(b,c,-1).permute(0,2,1)

        avgpool = F.conv2d(x, self.conv_sh.weight, padding = 0, dilation = 1)
        avgpool = self.bn_sh2(avgpool)
        avgpool_v = avgpool.view(b,c,-1)

        attn = torch.bmm(mxpool_v, avgpool_v)
        attn = torch.softmax(attn, dim = 1)

        v = F.avg_pool2d(x, [h,1])
        v = self.conv_v(v)
        v = v.view(b,c,-1)
        attn = torch.bmm(v, attn)
        attn = self.augment_conv(attn)

        attn1, attn2 = attn[:,0,:,:].unsqueeze(1), attn[:,1,:,:].unsqueeze(1)
        fusion = attn1*avgpool + attn2*mxpool

        out = F.dropout(self.fuse(fusion),p = self.drop, training = self.training)
        out = F.relu(self.gamma * out + (1-self.gamma)*x)
        out = self.fuse_out(out)

        return out
        

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim, 
        key_dim,
        num_heads,
        mlp_ratio = 4.,
        attn_ratio = 2.,
        drop = 0.,
        drop_path = 0.
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn = ScaleAwareStripAttention(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features = dim, out_features = dim, drop = drop)

    def forward(self,x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))

        return x

class ScaleAwareBlock(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio, attn_ratio, num_layers):
        super().__init__()
        self.tr = nn.Sequential(*(AttentionBlock(dim, key_dim, num_heads, mlp_ratio, attn_ratio) for _ in num_layers))

    def forward(self, x):
        return self.tr(x)

class ChannelWise(nn.Module):
    def __init__(self, channels, reduction = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            conv_block(channels, channels//reduction, 1,1, padding = 0, bias = False), nn.Relu(inplace = True),
            conv_block(channels//reduction, channels, 1,1, padding = 0, bias = False), nn.Sigmoid()
        )

    def forward(self,x):
        y = self.avg_pool(x)
        y = self.conv_pool(y)
        
        return x*y
    
class PyramidFusionNet(nn.Module):
    def __init__(
        self,
        channels_high,
        channels_low,
        channels_out,
        classes
    ):
        super().__init__()
        
        self.lateral_low = conv_block(channels_low, channels_high, 1,1,bn_act = True, padding = 0)

        self.conv_low = conv_block(channels_high, channels_out,3,1,bn_act = True, padding = 1)
        self.conv_high = conv_block(channels_high, channels_out, 3,1,bn_act = True, padding = 1)
        self.sa = ScaleAwareBlock(
            channels_out,
            key_dim = 16,
            num_heads = 8,
            mlp_ratio = 1,
            attn_ratio = 1,
            num_layers = 1
        )
        self.ca = ChannelWise(channels_out)

        self.FRB = nn.Sequential(
            conv_block(2*channels_high, channels_out, 1,1, bn_act = True, padding = 0),
            conv_block(channels_out, channels_out,3,1, bn_act = True, padding = 1)
        )

        self.adf = conv_block(channels_out, channels_out, 3,1, padding = 1, group = 1, bn_act =True)

    def forward(self,x_high, x_low):
        _,_,h,w = x_low.shape

        lat_low = self.lateral_low(x_low)

        high_up = F.interpolate(x_high, size = lat_low.shape[2:], mode = 'bilinear', align_corners = False)

        concate = torch.cat([lat_low, high_up], dim = 1)
        concate = self.FRB(concate)

        conv_low = self.conv_low(lat_low)
        conv_high = self.conv_high(high_up)

        sa = self.sa(concate)
        ca = self.ca(concate)

        mul1 = torch.mul(sa,conv_high)
        mul2 = torch.mul(ca,conv_low)

        attn_out = mul1 + mul2

        ADF = self.adf(attn_out)

        return ADF

class GlobalFeatureUpsample(nn.Module):
    def __init__(self, low_channels, in_channels, out_channels):
        super(GlobalFeatureUpsample, self).__init__()

        self.conv1 = conv_block(low_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            nn.ReLU(inplace=True))
        self.conv3 = conv_block(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bn_act=True)

        self.s1 = conv_block(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.s2 = nn.Sequential(
            conv_block(out_channels//2, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            SynchronizedBatchNorm2d(out_channels),
            nn.Sigmoid())

        self.fuse = conv_block(2*out_channels, out_channels, kernel_size=3, stride=1, padding=1, bn_act=True)

    def forward(self, x_gui, y_high):
        h, w = x_gui.size(2), x_gui.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_gui = self.conv1(x_gui)
        y_up = self.conv2(y_up)
        fuse = y_up + x_gui
        fuse = self.conv3(fuse)
        s1,s2 = torch.chunk(fuse,2,dim=1)
        s1 = self.s1(s1)
        s2 = self.s2(s2)

        ml1 = s1 * y_up
        ml2 = s2 * x_gui
        out = torch.cat([ml1,ml2],1)
        out = self.fuse(out)

        return out

class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegHead, self).__init__()
        self.fc = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.fc(x)