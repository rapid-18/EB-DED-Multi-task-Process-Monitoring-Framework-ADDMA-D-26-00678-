import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
import torch.nn.functional as F
class Pag(nn.Module):
    #pixel attention guided
    def __init__(self,channel_s,channel_t,ratio=2,with_channel=True,add=True,kernel=1,
                 dilation=1,depthwise=False,with_weight=False,As=True,At=True,**kwargs):
        super().__init__()
        self.As = As
        self.At = At
        padding = dilation*(kernel-1)//2
        # self.xs_conv=ConvModule(channel_s,channel_s,kernel_size=1,stride=1,padding=0)
        channels = int(channel_t*ratio)
        if kernel == 1:
            depthwise = False 
        groups_s = channel_s if depthwise else 1
        group_t = channel_t if depthwise else 1
        self.f_t = ConvModule(
            channel_t, channels, kernel_size=kernel,padding=padding,dilation=dilation,groups=group_t,act_cfg=None)
        self.f_s = ConvModule(
            channel_s, channels, kernel_size=kernel,padding=padding,dilation=dilation,groups=groups_s,act_cfg=None)
        self.with_channel = with_channel
        self.up_conv = ConvModule(channel_t,channel_s,1,1)
        self.add = add
        if with_channel:
            self.up = ConvModule(
                channels, channel_s, 1, act_cfg=None)
        self.with_weight=with_weight
        if self.add and self.with_weight:
            self.weight = nn.Parameter(torch.ones(1))

    def forward(self,x_s,x_t):
        x_t_1 = self.f_t(x_t)
        x_s_1 = self.f_s(x_s)
        if self.with_channel:
            sigma = torch.sigmoid(self.up(x_t_1 * x_s_1))
        else:
            sigma = torch.sigmoid(torch.sum(x_t_1 * x_s_1, dim=1).unsqueeze(1))
        if self.As and self.At:
            out = sigma * self.up_conv(x_t) + (1 - sigma) * x_s
        elif self.As:
            out = (1-sigma)*x_s
        else:
            out = sigma * self.up_conv(x_t)
        if self.add:
            if self.with_weight:
                out = self.weight[0]*out
            out = out + x_s
        return out
class MSPag(nn.Module):
    #Multi-Scale-Pag
    def __init__(self, channel_s, channel_t, ratio=2,use_soft_max=True,add=True,fixed_weight=False,As=True,At=True, 
                    add_fixed=False,use_activate=True,kernels=[1, 3, 3], dilations=[1,1,2,],with_channel=True,
                    depthwise=False,adaptive_weight=False,**kwargs):
        super().__init__()
        self.num_branches = len(kernels)
        if len(dilations)<len(kernels):
            new_dilations = [dilations[0] for i in range(len(kernels))]
            dilations = new_dilations
        assert(len(dilations)==len(kernels))
        self.pag_branches = nn.ModuleList()
        for i, kernel in enumerate(kernels):
            self.pag_branches.append(
                Pag(
                    channel_s=channel_s,
                    channel_t=channel_t,
                    kernel_size=kernel,
                    dilation=dilations[i],
                    ratio=ratio,
                    with_channel=with_channel,
                    depthwise=depthwise,
                    add=False,
                    As=As,
                    At=At
                )
            )
        self.fixed_weight = fixed_weight
        self.adaptive_weight = adaptive_weight
        if not fixed_weight:
            if not self.adaptive_weight:
                self.branch_weights = nn.Parameter(torch.ones(self.num_branches) / self.num_branches)
            else:
                self.branch_weights = Adaptive_Weight(in_channels=channel_t*self.num_branches,num_branches=self.num_branches,reduce_ratio=2)
        self.use_soft_max=use_soft_max
        self.add=add
        self.add_fixed=add_fixed
        self.use_activate=use_activate

    def forward(self, x_s, x_t):
        branch_outputs = []
        
        # 计算每个分支的输出
        for branch in self.pag_branches:
            branch_output = branch(x_s, x_t)
            branch_outputs.append(branch_output)
        
        
        # 计算分支权重
        if not self.fixed_weight:
            if not self.adaptive_weight:
                if not self.use_activate:
                    weights = self.branch_weights
                else:
                    if self.use_soft_max:
                        weights = torch.softmax(self.branch_weights, dim=0)
                    else:
                        weights = torch.sigmoid(self.branch_weights)
                    # weights = self.branch_weights
                if self.add_fixed:
                    weights = weights + 1/(1*self.num_branches)
            else:
                aggregated = torch.cat(branch_outputs,dim=1) #b,2c,h,w
                weights = self.branch_weights(x_t) #b,2,1,1
                B,C,H,W=aggregated.shape
                aggregated = aggregated.reshape(B,self.num_branches,-1,H,W)
                aggregated = weights * aggregated
                aggregated = torch.sum(aggregated,dim=1)
        else:
            weights = [1/self.num_branches for i in range(self.num_branches)]
        # 加权聚合
        if not self.adaptive_weight or self.fixed_weight:
            aggregated = None
            for i, output in enumerate(branch_outputs):
                if aggregated is None:
                    aggregated = weights[i] * output
                else:
                    aggregated = aggregated + weights[i] * output
        
        # 最终与原输入相加
        if self.add:
            return aggregated + x_s
        else:
            return aggregated
