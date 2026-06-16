import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmseg.registry import MODELS
import torch.nn.functional as F
from mmcv.cnn import DepthwiseSeparableConvModule,ConvModule
from ..resnet import ResNet
from ..resnet import ResNet,ResNetV1c,ResNetV1d
from .unet_backbone import Basic_backbone
from mmengine import Config
def _convert_to_2d(x: torch.Tensor,left_first=True) -> torch.Tensor:
    """(N, C, T, H, W) -> (N x T, C, H, W)"""
    if not left_first:
        x = x[:,:,:-1,:,:]
    x = x.permute((0, 2, 1, 3, 4))
    x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
    return x

def _convert_to_3d(x: torch.Tensor,batches) -> torch.Tensor:
    """(N x T, C, H, W) -> (N, T, C, H, W)"""
    x = x.reshape(batches, -1, x.shape[1], x.shape[2], x.shape[3])
    # x = x.permute((0, 2, 1, 3, 4))
    return x

@MODELS.register_module()
class DualBranch_Backbone(BaseModel):
    backbone_config={
        "resnet":ResNet,
        'resnetV1c':ResNetV1c,
        'resnetV1d':ResNetV1d,
        'basic':Basic_backbone,
    }
    def __init__(self,
                pretrained=None,
                pretrained_config=None, 
                frozen = False,                 
                left_first=True,#True:空间 False：时间
                from_AED=False,
                backbone_name ='basic',
                backbone_config=dict(
                in_channels=1,
                base_channels=48,
                num_stages=5,
                strides=(1, 1, 1, 1,1),
                enc_num_convs=(2, 2, 2, 2,2),
                downsamples=(True, True, True,True),
                enc_dilations=(1, 1, 1, 1,1),
                ),
                aux_backbone_config=dict(
                in_channels=1,
                base_channels=16,
                num_stages=5,
                strides=(1, 1, 1, 1,1),
                enc_num_convs=(2, 2, 2, 2,2),
                downsamples=(True, True, True,True),
                enc_dilations=(1, 1, 1, 1,1),
                )
                 ):
        super().__init__()
        self.backbone = self.backbone_config[backbone_name](**backbone_config)
        self.aux_backbone=self.backbone_config[backbone_name](**aux_backbone_config)
        self.left_first=left_first
        if pretrained is not None and pretrained_config is not None:
            pretrained_model = MODELS.build(Config.fromfile(pretrained_config).model)
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            pretrained_model.load_state_dict(state_dict, strict=True)
            # 提取 Encoder
            if not from_AED:
                self.backbone = pretrained_model.backbone.backbone.backbone
                self.aux_backbone = pretrained_model.backbone.backbone.aux_backbone
            else:
                self.backbone = pretrained_model.backbone.backbone
                self.aux_backbone = pretrained_model.backbone.aux_backbone
            del pretrained_model
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.aux_backbone.parameters():
                param.requires_grad = False

    def forward(self,x,**kwargs):
        batches = x.shape[0]
        x_t = _convert_to_2d(x,left_first=self.left_first)
        x_t = list(self.aux_backbone(x_t))
        for i in range(len(x_t)):
            x_t[i] = _convert_to_3d(x_t[i],batches)
        x_s = list(self.backbone(x[:,:,-1,:,:]))
        return x_s,x_t

@MODELS.register_module()
class SingleStreamBackbone(BaseModel):
    backbone_config={
        "resnet":ResNet,
        'resnetV1c':ResNetV1c,
        'resnetV1d':ResNetV1d,
        'basic':Basic_backbone,
    }
    def __init__(self,                  
                 space_stream=True,#True:空间 False：时间
                 backbone_name ='resnet',
                 backbone_config=dict(
                    in_channels=1,
                    depth=18,
                    base_channels=64,
                    num_stages=4,
                    strides=(1,2,2,2),
                 ),
                 **kwargs,
                 ):
        super().__init__()
        self.backbone = self.backbone_config[backbone_name](**backbone_config)
        self.stream=space_stream

    def forward(self, x,pretrainning=True):
        if len(x.shape)==4:
            x = x.unsqueeze(dim=1)
        if pretrainning:
            if self.stream:
                x_t = None
                x_s = self.forward_s(x)
            else:
                x_s = None
                x_t = self.forward_t(x)
        else:
            x_s,x_t = self.forward_st(x)
        return x_s,x_t
    
    def forward_t(self,x):
        batches = x.shape[0]
        x_t = _convert_to_2d(x,left_first=False)
        x_t = list(self.backbone(x_t))
        for i in range(len(x_t)):
            x_t[i] = _convert_to_3d(x_t[i],batches)
        return x_t
    
    def forward_s(self,x):
        cur_frame=x[:,:,-1,:,:]
        x_s = list(self.backbone(cur_frame))
        return x_s

    def forward_st(self,x):
        batches = x.shape[0]
        x = _convert_to_2d(x,left_first=True)
        x = list(self.backbone(x))
        for i in range(len(x)):
            x[i] = _convert_to_3d(x[i],batches)
        x_s = [xt[:,-1,:,:,:] for xt in x]
        x_t = [xt[:,:-1,:,:,:] for xt in x]
        return x_s,x_t
