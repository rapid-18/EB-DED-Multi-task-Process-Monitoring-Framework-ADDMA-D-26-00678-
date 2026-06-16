import torch
import torch.nn as nn
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F
from mmseg.utils import ConfigType, SampleList
from typing import List, Tuple
from ..utils.Clf_TSF import*
@MODELS.register_module()
class Base_Clf_Head(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 num_classes=2,
                 reduce_ratio=3,
                 dropout_ratio=0.1,
                 mode='2d',
                 base_channels=48,
                 **kwargs,
                 ):
        super().__init__(in_channels=in_channels,channels=num_classes,num_classes=num_classes,**kwargs)
        self.pool = nn.AdaptiveMaxPool2d((1,1)) if mode == '2d' else nn.AdaptiveAvgPool3d((1,1,1))
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fc1 = nn.Linear(in_channels, in_channels//reduce_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//reduce_ratio, num_classes)

    def forward(self, x,vis=False, **kwargs):
        if isinstance(x,tuple):
            x,_ = x
        if isinstance(x,List):
            x = x[-1]
        # [B, C, T, H, W] → 全局池化 → [B, C*T]
        # print(x.shape)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ac1(x)
        if vis:
            return x
        x = self.dropout(x)
        return self.fc2(x)  # 直接输出回归值 [B, num_classes]
    
    def predict_by_feat(self, seg_logits: torch.Tensor,
                        batch_img_metas: List[dict]) -> torch.Tensor:
        return seg_logits
    
    def loss_by_feat(self, seg_logits: torch.Tensor,
                     batch_data_samples: SampleList) -> dict:
        seg_label = torch.stack([sample.clf_label for sample in batch_data_samples],dim=0)
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        # seg_label = seg_label.squeeze(1) #label:[B,H,W]

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,)
        # fused_logits = torch.mean(torch.stack(seg_logits), dim=0)
        # loss['acc_seg'] = accuracy(
        #     seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

@MODELS.register_module()
class DualStream_Clf_Head(Base_Clf_Head):
    st_fuse_settings = {
        'DPFA_C3D':DPFA_C3D,
    }
    def __init__(self,
                 in_channels,
                 time_step,
                 include_first=False,
                 in_channels_aux=768,
                 st_fuse='DPFA_C3D',
                 st_fuse_config={},
                 base_channels=48,
                 last_fuse='cat',
                 reduce_ratio=4,
                 num_classes=2,
                 tsm_agg = False,
                 **kwargs,
                 ):
        super().__init__(in_channels=in_channels,num_classes=num_classes,**kwargs)
        if tsm_agg:
            self.agg = TSM_agg(channels=in_channels_aux,include_first=include_first)
        if tsm_agg:
            include_first = True
        self.st_fuse=self.st_fuse_settings[st_fuse](channel_s=in_channels,channel_t=in_channels_aux,time_step=time_step-1,include_first=include_first,**st_fuse_config)
        self.last_fuse=last_fuse
        self.tsm_agg = tsm_agg

        channel = in_channels+self.st_fuse.out_channels if last_fuse=='cat' else in_channels
        self.fc1=nn.Linear(channel, channel//reduce_ratio)
        self.fc2 = nn.Linear(channel//reduce_ratio, num_classes)

    def forward(self, enc_outs,vis=False, **kwargs):
        enc_outs_s,enc_outs_t = enc_outs
        if self.tsm_agg:
            xt = self.agg(enc_outs_t[-1])
        else:
            xt = enc_outs_t[-1]
        fused_outs = self.st_fuse(enc_outs_s[-1],xt)
        # [B, C, T, H, W] → 全局池化 → [B, C*T]
        x = enc_outs_s[-1]
        x = self.pool(x)
        x = self.flatten(x)
        if self.last_fuse=='cat':
            x = torch.cat([x,fused_outs],dim=1)
        elif self.last_fuse =='add':
            x = x+fused_outs
        else:
            x = x
        x = self.fc1(x)
        x = self.ac1(x)
        if vis:
            return x
        x = self.dropout(x)
        return self.fc2(x)  # 直接输出回归值 [B, num_classes]
