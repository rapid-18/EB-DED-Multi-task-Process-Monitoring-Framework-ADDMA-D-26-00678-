import torch
import torch.nn as nn
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from typing import List, Tuple

    
@MODELS.register_module()
class Decoder_Head(BaseDecodeHead):
    def __init__(self,
                in_channels: int = 128,          # 潜在空间维度
                hidden_channels:list=[64,128],
                least_size=(1,1),
                latent_dim=1024,
                projection_size=(4,4),
                output_channels: int = 1,       # 输出通道数（如RGB图像）
                upsample_type:int=0, #0:插值 1:卷积
                output_shape: tuple = (420, 420),  # 输出图像尺寸 (H, W)
                loss_decode=dict(type='MSELoss', loss_weight=1.0)
                ):
        super().__init__(in_channels=in_channels,channels=in_channels,num_classes=1,loss_decode=loss_decode)
        self.last_pool_s = nn.AdaptiveAvgPool2d(least_size)
        self.latent_fc = nn.Sequential(
            nn.Linear(self.in_channels*least_size[0]*least_size[1], latent_dim),nn.ReLU())  # 假设最后特征图尺寸16x16
        
        self.channels = list(reversed(hidden_channels))
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, self.channels[0] * projection_size[0] * projection_size[1]),  # 投影到 (base_channels, 4, 4)
            nn.ReLU(inplace=True)
        )
        self.projection_size=projection_size
        

        self.deconvs = nn.ModuleList()
        if upsample_type == 1:
            for i in range(len(self.channels)-1):
                self.deconvs.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i+1],
                    kernel_size=5,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(self.channels[i+1]),
                nn.ReLU(inplace=True)))
        else:
            for i in range(len(self.channels)-1):
                self.deconvs.append(nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',),
                ConvModule(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i+1],
                    kernel_size=3,
                    stride=1,
                    padding=1
                    ),
                nn.BatchNorm2d(self.channels[i+1]),
                nn.ReLU(inplace=True)))

        self.last_conv = nn.Sequential(
            nn.Upsample(
                size=output_shape,
                mode='bilinear',),
            ConvModule(
                in_channels=self.channels[-1],
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1
                ),
            nn.BatchNorm2d(output_channels),
            nn.Sigmoid())
        
    def forward(self,x,**kwargs):
        x = self.last_pool_s(x)
        x = torch.flatten(x,1)
        x = self.latent_fc(x)
        x = self.projection(x)
        x = x.view(-1, self.channels[0], self.projection_size[0],self.projection_size[1])
        for layer in self.deconvs:
            x = layer(x)
        x = self.last_conv(x)
        return x
    
    def predict_by_feat(self, seg_logits: torch.Tensor,
                        batch_img_metas: List[dict]) -> torch.Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        return seg_logits
    
    def loss_by_feat(self, seg_logits: torch.Tensor,
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


        

