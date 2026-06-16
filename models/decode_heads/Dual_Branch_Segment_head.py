import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
import torch.nn.functional as F
from mmseg.utils import ConfigType, SampleList
from ..utils import UpConvBlock, Upsample, PixelShuffleUpconv
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from ..backbones.unet import BasicConvBlock
from .decode_head import BaseDecodeHead
from ..utils.Segment_TSF import*
from ..utils.Temporal_agg import*
class Unet_decoder(nn.Module):
    def __init__(self,
                 out_indices = (0,),
                 output_shape:tuple[int]=(420,420),
                 base_channels=32,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 backbone_type='resnet',
                 norm_eval=False,):
        super().__init__()
        self.out_indices=out_indices
        self.out_shape = output_shape
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels
        self.decoder = nn.ModuleList()
        stage_size = CheckSize(output_shape,num_stages)
        up_sample_cfgs = [dict(type=upsample_cfg['type']) for s in stage_size]
        for i in range(num_stages):
            if i==0 and backbone_type=='resnet':
                skip_channels = base_channels
            else:
                skip_channels = base_channels* 2**(i-1)
            if i!=0 or backbone_type=='resnet':
                upsample = downsamples[i-1]
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2**i,
                        skip_channels=skip_channels,
                        out_channels=skip_channels,
                        num_convs=dec_num_convs[i-1],
                        stride=1,
                        dilation=dec_dilations[i-1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=up_sample_cfgs[i-1] if upsample else None,
                        dcn=None,
                        plugins=None))

        
    def forward(self, enc_outs,**kwargs):
        dec_outs = []
        x = enc_outs[-1]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            if i in self.out_indices:
                dec_outs.append(x)
        dec_outs[-1] = x
        if len(dec_outs)==1:
            return x
        else:
            return list(reversed(dec_outs))        

@MODELS.register_module()
class DualBranch_SegmentHead(BaseDecodeHead):
    BlockType = {
        'Unet':Unet_decoder,
    }

    st_fuse_settings = {
        'MSPag':MSPag
    }
    temporal_agg_settings={
        'DP':DP_Agg,
    }

    def __init__(self,
                in_channels=768,
                output_shape=(416,416),
                base_channels=48,
                base_channels_aux=16,
                num_stages=4,
                time_step=4,
                st_fuse=['Pag'],
                st_fuse_config=[{'ratio':2}],
                fuse_indices=[(1,2)],
                BlockType='Unet',
                BlockConfig={},
                backbone_type='basic',
                temporal_agg=DP,
                agg_config=dict(avg_pool=True,max_pool=True,num_convs=1,pool_cat=False,include_first=True),
                num_convs=1,
                **kwargs
                ):
        super().__init__(in_channels=in_channels,channels=base_channels,**kwargs)
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
                else:
                    in_channel = base_channels*2**(indice-1) if backbone_type=='resnet' else base_channels*2**indice
                    in_channel_t = base_channels_aux*2**(indice-1) if backbone_type=='resnet' else base_channels_aux*2**indice
                self.st_fuse.append(self.st_fuse_settings[st_fuse[k]](channel_s=in_channel,channel_t=in_channel_t,time_step=time_step-1,**st_fuse_config[k]))
                if self.temporal_agg:
                    self.agg_modules.append(self.temporal_agg_settings[temporal_agg](in_channels=in_channel_t,**agg_config))
        self.fuse_indices = indice_temp
        self.Block = self.BlockType[BlockType](output_shape=output_shape,base_channels=base_channels,backbone_type=backbone_type,
                                               num_stages=num_stages,**BlockConfig)
        convs = [ConvModule(base_channels,base_channels,3,1,1) for i in range(num_convs)]
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        
    def forward(self,enc_outs,**kwargs):
        enc_outs_s,enc_outs_t = enc_outs
        agg = [None for i in range(len(enc_outs_t))]
        # for l,s in enumerate(enc_outs_s):
        #     print(l,s.shape)
        for i,indice in enumerate(self.fuse_indices):
            if self.temporal_agg:
                agg[indice] = self.agg_modules[i](enc_outs_t[indice])
            enc_outs_s[indice] = self.st_fuse[i](enc_outs_s[indice],agg[indice])
        x = self.Block(enc_outs_s)
        if isinstance(x,list):
            x = x[0]
        out = self.convs(x)
        out = self.cls_seg(x)
        return out
