import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
from torch import Tensor
from typing import List, Tuple
from mmseg.utils import ConfigType, SampleList
from ..utils.st_FuseBlock import*
from ..utils.st_FuseBlock import _convert_to_3d
from ..utils.up_conv_block import RefinementModule
from ..utils.Seg_st_FuseBlock import*
from ..utils.Temporal_agg import*
from ..utils.LowLevel_Fuse import*
def CheckSize(shape,num_stages,backbone_type='resnet'):
        h,w = shape
        if backbone_type=='resnet':
            h = (h+1)//2
            w = (w+1)//2
        stage_size = [(h,w)]
        add = 1 if backbone_type=='resnet' else 0
        add_stages = 0 if backbone_type=='resnet' else -1
        for i in range(num_stages+add_stages):
            h = (h+add)//2
            w = (w+add)//2
            stage_size.append((h,w))
        return stage_size

def CheckUpParams(shape,stage,stage_size):
        H,W = shape
        h,w = stage_size
        #i:下采样次数
        stride = 2**(stage+1)
        kernel = 2*stride
        padding = (kernel-H+stride*(h-1))//2
        return kernel,stride,padding
    
class CED(nn.Module):
    def __init__(self,
        base_channels=64,
        scale_factor=2,
        num_stages=4,
        output_shape=(420,420),
        backbone_type='resnet',
    ):
        super().__init__()
        self.num_stages = num_stages
        self.output_shape=output_shape
        self.scale_factor = scale_factor
        #确定各级的尺寸
        self.stage_size = CheckSize(output_shape,num_stages,backbone_type)
        self.backbone_type=backbone_type
        #最底层的上采样
        self.bridge = nn.Sequential(nn.Upsample(size=self.stage_size[-2],mode='bilinear',),nn.Conv2d(base_channels*2**(num_stages-1),base_channels*2**(num_stages-1),1,1),
                                    nn.BatchNorm2d(base_channels*2**(num_stages-1)),nn.ReLU())
        self.Up_Fuse = nn.ModuleList()
        for i in range(num_stages):
            in_channel_decoder=base_channels*2**i
            if i==0:
                in_channel_encoder = base_channels
                s_size = output_shape
            else:
                in_channel_encoder =  base_channels*2**(i-1)
                s_size = self.stage_size[i-1] if self.backbone_type=='resnet' else self.stage_size[i-2]
            if backbone_type=='resnet':
                self.Up_Fuse.append(RefinementModule(in_channel_encoder,in_channel_decoder,in_channel_encoder,s_size,scale_factor))
            else:
                if i!=0:
                    scale_factor=1 if i==1 else self.scale_factor
                    s_size=output_shape if i==1 else s_size
                    self.Up_Fuse.append(RefinementModule(in_channel_encoder,in_channel_decoder,in_channel_encoder,s_size,scale_factor))
        self.num_stages = num_stages-1 if backbone_type!='resnet' else num_stages
        self.final_conv = nn.Sequential(nn.Conv2d(base_channels,1,1,1),nn.BatchNorm2d(1),nn.Sigmoid())

    def forward(self,enc_outs):
        x = enc_outs[-1]
        x = self.bridge(x)
        for i in reversed(range(self.num_stages)):
            x = self.Up_Fuse[i](enc_outs[i],x)
        x = self.final_conv(x)
        return [x] #返回list为[特征尺度从低到高，融合后的最终预测]

@MODELS.register_module()
class DualStream_EdgeDetectHead(BaseDecodeHead):
    BlockType = {
        'CED':CED,
    }

    st_fuse_settings = {
        'MSPag':MSPag
    }

    temporal_agg_settings={
        'DP':DP_Agg,
    }

    def __init__(self,
                output_shape=(416,416),
                base_channels=48,
                base_channels_aux=32,
                num_stages=4,
                time_step=4,
                st_fuse=['MSPag'],
                st_fuse_config=[{}],
                fuse_indices=[(1,2,3,4)],
                BlockType='CED',
                BlockConfig:dict={},
                backbone_type='basic',
                num_convs=1,
                temporal_agg='DP,
                agg_config=dict(avg_pool=True,max_pool=True,num_convs=1,pool_cat=False),
                **kwargs
                ):
        super().__init__(num_classes=1,in_channels=base_channels,channels=base_channels,**kwargs)
        #时空融合选项
        st_fuse=[st_fuse] if isinstance(st_fuse,str) else st_fuse
        fuse_indices=[fuse_indices] if isinstance(fuse_indices,tuple) else fuse_indices
        st_fuse_config=[st_fuse_config] if isinstance(st_fuse_config,dict) else st_fuse_config
        self.st_fuse = nn.ModuleList()
        self.fuse_indices=fuse_indices
        indice_temp=[]
        self.temporal_agg=True if temporal_agg is not None else False
        self.agg_modules = nn.ModuleList()
        assert len(self.fuse_indices)==len(st_fuse) and len(self.fuse_indices)==len(st_fuse_config)
        for k,fuse_indices in enumerate(self.fuse_indices):
            for i,indice in enumerate(fuse_indices):
                indice_temp.append(indice)
                if indice ==0 and backbone_type=='resnet':
                    in_channel = base_channels
                    in_channel_t = base_channels_aux
                else:
                    in_channel = base_channels*2**(indice-1) if backbone_type=='resnet' else base_channels*2**indice
                    in_channel_t = base_channels_aux*2**(indice-1) if backbone_type=='resnet' else base_channels_aux*2**indice
                self.st_fuse.append(self.st_fuse_settings[st_fuse[k]](channel_s=in_channel,channel_t=in_channel_t,time_step=time_step-1,out_channel=in_channel,**st_fuse_config[k]))
                if self.temporal_agg:
                    self.agg_modules.append(self.temporal_agg_settings[temporal_agg](in_channels=in_channel_t,**agg_config))
        self.fuse_indices = indice_temp
        self.Block = self.BlockType[BlockType](output_shape=output_shape,base_channels=base_channels,backbone_type=backbone_type,
                                               num_stages=num_stages,**BlockConfig)

        
    def forward(self,enc_outs,**kwargs):
        enc_outs_s,enc_outs_t = enc_outs
        for i,indice in enumerate(self.fuse_indices):
            if self.temporal_agg:
                enc_outs_t[indice] = self.agg_modules[i](enc_outs_t[indice])
            enc_outs_s[indice] = self.st_fuse[i](enc_outs_s[indice],enc_outs_t[indice])
        enc_outs = self.Block(enc_outs_s)
        return enc_outs #返回结果为k×[N,H,W]的List（RCF形式的多尺度模型）
    
    def predict_by_feat(self, seg_logits: List[Tensor],
                        batch_img_metas: List[dict]) -> Tensor:
        #平均所有输出
        if len(seg_logits)>1:
            seg_logits = torch.mean(torch.stack(seg_logits), dim=0)
        else:
            seg_logits = seg_logits[-1]
        return seg_logits
    
    def loss_by_feat(self, seg_logits: List[Tensor],
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
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
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        # fused_logits = torch.mean(torch.stack(seg_logits), dim=0)
        # loss['acc_seg'] = accuracy(
        #     seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss
