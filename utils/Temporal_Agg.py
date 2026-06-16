import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
import torch.nn.functional as F
from mmseg.utils import ConfigType, SampleList
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer

class DP_Agg(nn.Module):
    def __init__(self,in_channels,avg_pool=True,max_pool=True,num_convs=1,pool_cat=False,include_first=False,**kwargs):
        super().__init__()
        self.avg_pool = avg_pool
        self.max_pool = max_pool
        convs = [ConvModule(in_channels,in_channels,1,1) for i in range(num_convs)]
        self.convs = nn.Sequential(*convs)
        self.pool_cat = pool_cat
        if self.pool_cat:
            self.cat_conv = ConvModule(2*in_channels,in_channels,1,1)
        self.include_first = include_first
        
    def forward(self,x):
        if not self.include_first:
            x = x [:,:-1,:,:,:]
        #x:[b,t,c,h,w]
        if self.avg_pool:
            avg_x = torch.mean(x,dim=1)
        if self.max_pool:
            max_x,_ = torch.max(x,dim=1)
        if self.avg_pool and self.max_pool:
            if self.pool_cat:
                x = torch.cat([avg_x,max_x],dim=1)
                x = self.cat_conv(x)
            else:
                x = max_x+avg_x
        elif self.max_pool:
            x = max_x
        else:
            x = avg_x
        return self.convs(x)

