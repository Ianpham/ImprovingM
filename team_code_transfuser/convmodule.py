#short version in implemented from https://arxiv.org/abs/2305.11624,applied only for 2d in this project
# all rights reserved
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn import SyncBatchNorm as _SyncBatchNorm

import torch
import torch.nn as nn

from functools import partial
from typing import Union, Optional

def efficient_conv(bn: _BatchNorm, conv = nn.modules.conv._ConvNd, x: torch.Tensor):
    # get param
    weight_on_the_fly   = conv.weight
    bias_on_the_fly     = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_var)
    bn_weight           = bn.weight if bn.weight is not None else torch.ones_like(bn.running_var)
    bn_bias             = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_var)

    # compute weight coeff
    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape([-1] + [1]*(len(conv.weight.shape) - 1)) #[C_out, 1, 1, 1]  
    coeff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff # [C_out, 1, 1, 1]

    # weight and bias on the fly
    weight_on_the_fly = weight_on_the_fly * coeff_on_the_fly # [C_out, C_in, k, k]
    bias_on_the_fly = (bias_on_the_fly - bn.running_mean) * coeff_on_the_fly.flatten() + bn_bias # [C_out]

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)

class ConvModule(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kk_size: Union[int,tuple[int, int]],
                 act_func = Optional(nn.RelU(), nn.LeakyReLU),
                 norm_func = Optional(nn.BatchNorm2d, _BatchNorm, _InstanceNorm,_SyncBatchNorm),
                 order: tuple = ('conv', 'norm', 'act'),                               
                 inplace = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kk_size = kk_size        
        self.inplace = inplace
        self.order = order
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kk_size, 1,0)
        self.norm = norm_func(self.out_channels)
        self.act_func = act_func
        self.init_weights()
        
    def init_weights(self):
        if self.act_func == nn.ReLU():
            return nn.init.kaiming_normal_(mode = 'fan_out', a = 0, nonlinearity='relu')
        if self.act_func == nn.Tanh() or nn.Sigmoid():
            return nn.init.xavier_normal_(gain = 1)

    def forward(self, x):
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer_index + 1 < len(self.order) and self.order[layer_index + 1] == 'norm':
                if layer == 'conv':
                    self.conv.forward = partial(efficient_conv, self.norm, self.conv)
                    layer_index += 1
                    x = self.conv(x)
                    del self.conv.forward
                else:
                    x = self.conv(x)
            elif layer == 'norm':
                x = self.norm(x)
            elif layer == 'act':
                x = self.act_func(x)
            layer_index += 1
        return x

            


