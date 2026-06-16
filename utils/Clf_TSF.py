import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
import torch.nn.functional as F
from .se_layer import SELayer

class DPFA_C3D(nn.Module):
    def __init__(self,channel_s,channel_t,time_step,reduce_ratio=8,pool_size=1,avg_pool=True,max_pool=True,include_first=False,
                 Parallel_att=True,temporal_first=True,temporal_att=True,channel_att=True,time_conv_reduce=1,
                 max_avg_cat=False,threeD_agg=True,threeD_agg_first=False,last_conv=False,temporal_pool=False,**kwargs):
        super().__init__( **kwargs)
        self.pool_size=pool_size
        assert(avg_pool or max_pool)
        self.include_first = include_first
        self.avg_pool = avg_pool
        self.max_pool = max_pool
        self.temporal_att = temporal_att
        self.channel_att = channel_att
        self.max_avg_cat=max_avg_cat
        self.temporal_pool = temporal_pool
        time_step_mul=1 if self.temporal_pool else time_step+1
        self.threeD_agg = threeD_agg
        self.threeD_agg_first = threeD_agg_first
        self.use_last_conv=last_conv
        self.channel_up = (channel_t!=channel_s)
        self.time_conv_reduce = time_conv_reduce
        if self.channel_up:
            self.up_conv = ConvModule(channel_t,channel_s,1,1)
            channel_t = channel_s
        if self.temporal_pool and self.threeD_agg_first:
            self.avg_pool = nn.AdaptiveAvgPool3d((1,pool_size,pool_size))
            self.max_pool = nn.AdaptiveMaxPool3d((1,pool_size,pool_size))
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d((time_step+1,pool_size,pool_size))
            self.max_pool = nn.AdaptiveMaxPool3d((time_step+1,pool_size,pool_size))
        if temporal_att:
            self.temporal_attention = nn.Sequential(
                nn.AdaptiveAvgPool3d((time_step+1, 1, 1)),  # [n, c, t, 1, 1]
                nn.Conv3d(channel_t, channel_t//reduce_ratio, 1),
                nn.BatchNorm3d(channel_t//reduce_ratio),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel_t//reduce_ratio, channel_t, 1),
                nn.BatchNorm3d(channel_t),
                nn.Sigmoid()
            )
        if channel_att:
                self.channel_atttion = SELayer(channel_t,ratio=reduce_ratio,retun_weight=True)

        if self.use_last_conv:
            if not self.threeD_agg_first and self.threeD_agg:
                last_conv_channel = time_step_mul*(channel_t//time_conv_reduce)
            elif self.threeD_agg_first and threeD_agg:
                last_conv_channel = time_step_mul*(channel_t//time_conv_reduce)*(1+max_avg_cat)
            else:
                last_conv_channel = time_step_mul*channel_t*(1+max_avg_cat)
            self.last_conv = ConvModule(last_conv_channel,last_conv_channel,1,1)
        if self.threeD_agg:
            self.time_conv = nn.Sequential(nn.Conv3d(
                in_channels=channel_t*(1+self.max_avg_cat),
                out_channels=channel_t//time_conv_reduce,
                kernel_size=(3, 1, 1),  # 时间维度3个步长，空间1x1
                padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(channel_t//time_conv_reduce),
            nn.ReLU()
            )
            self.out_channels=(channel_t//time_conv_reduce)*(pool_size**2)*time_step_mul
        else:
            self.out_channels=channel_t*(pool_size**2)*time_step_mul
        self.Parallel_att = Parallel_att
        if not self.Parallel_att:
            self.temporal_first = temporal_first
            assert(temporal_att and channel_att)
        

    def forward(self,x_s:torch.Tensor,x_t:torch.Tensor):
        #x_s:[B,C,H,W] x_t[B,T,C,H,W]
        #channel weight
        if not self.include_first:
            x_t = x_t[:,:-1,:,:,:]
        if self.channel_up:
            B,T,C,H,W = x_t.shape
            x_t = x_t.reshape(B*T,C,H,W)
            x_t = self.up_conv(x_t)
            x_t = x_t.reshape(B,T,-1,H,W)
        # if self.channel_up:
        #     x_s = self.down_conv(x_s)
        B,T,C,H,W = x_t.shape
        xs = x_s.unsqueeze(dim=1)
        x = torch.cat([xs,x_t],dim=1) #b,t+1,c,h,w

        if self.Parallel_att:
            if self.temporal_att:
                x1 = x.permute(0,2,1,3,4)#b,c,t+1,h,w
                temporal_weights = self.temporal_attention(x1) #b,c,t,1,1
            if self.channel_att:
                x1 = x.reshape(B*(T+1),C,H,W)
                channel_weight = self.channel_atttion(x1)    #b*T,c,1,1
                channel_weight = channel_weight.reshape(B,T+1,C,1,1).permute(0,2,1,3,4)#b,c,t,1,1
            if self.channel_att and self.temporal_att:
                weight = channel_weight*temporal_weights
            elif self.channel_att:
                weight = channel_weight
            elif self.temporal_att:
                weight = temporal_weights
            else:
                weight = 1
            x = x.permute(0,2,1,3,4) #b,c,t+1,h,w
            x = weight*x

        else:
            if self.temporal_first:
                x1 = x.permute(0,2,1,3,4)#b,c,t+1,h,w
                temporal_weights = self.temporal_attention(x1) #b,c,t,1,1
                # print(x1.shape)
                # print(temporal_weights.shape)
                x = temporal_weights*x1 #b,c,t,1,1
                x1 = x.reshape(B*(T+1),C,H,W)
                channel_weight = self.channel_atttion(x1)  #b*T,c,1,1
                x = (channel_weight*x1).reshape(B,T+1,C,H,W).permute(0,2,1,3,4)
            else:
                x1 = x.reshape(B*(T+1),C,H,W)
                channel_weight = self.channel_atttion(x1)  #b*T,c,1,1
                x = (channel_weight*x1).reshape(B,T+1,C,H,W).permute(0,2,1,3,4)
                temporal_weights = self.temporal_attention(x) #b,c,t,1,1
                x = temporal_weights*x #b,c,t,1,1
        if self.threeD_agg_first and self.threeD_agg:
            x = self.time_conv(x)
        if self.avg_pool:
            avg_x = self.avg_pool(x)#[b,c,t,ps,ps]
        if self.max_pool:
            max_x = self.max_pool(x)#[b,c,t,ps,ps]
        if self.avg_pool and self.max_pool:
            if self.max_avg_cat:
                x = torch.cat([avg_x,max_x],dim=1)
            else:
                x = avg_x+max_x
        elif self.avg_pool:
            x = avg_x
        else:
            x = max_x

        if self.threeD_agg and not self.threeD_agg_first:
            x = self.time_conv(x)
            if self.temporal_pool:
                x1 = torch.mean(x,dim=2)
                x2,_ = torch.max(x,dim=2)
                x = x1+x2
        if self.use_last_conv:
            if not self.temporal_pool:
                x = x.reshape(B,(T+1)*(C//self.time_conv_reduce),self.pool_size,self.pool_size)
            else:
                x = x.reshape(B,-1,self.pool_size,self.pool_size)
            x = self.last_conv(x)

        x = x.reshape(B,-1)
        return x
