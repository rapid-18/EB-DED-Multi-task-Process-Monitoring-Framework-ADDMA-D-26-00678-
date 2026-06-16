# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer
import torch.nn.functional as F

class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""
        x = self.upsample(x)
        if skip.shape[-2]>x.shape[-2]:
            skip = skip[:,:,:x.shape[-2],:x.shape[-1]]
            # x = F.upsample_bilinear(x,skip.shape[-2:])
        elif skip.shape[-2]<x.shape[-2]:
            x = x[:, :, :skip.shape[-2], :skip.shape[-1]]
            # skip = F.upsample_bilinear(skip,x.shape[-2:])
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out
    
class PixelShuffleUpconv(nn.Module):
    """CED网络中的核心精炼模块 (Refinement Module)
    负责融合来自前向传播路径（encoder）和后向精炼路径（decoder）的特征图，
    并使用亚像素卷积进行上采样。
    """
    def __init__(self, in_channels_encoder, in_channels_decoder, out_channels, out_shape,scale_factor=2,num_convs=2):
        """
        Args:
            in_channels_encoder: 来自前向传播路径（横向连接）的特征图通道数
            in_channels_decoder: 来自后向精炼路径（上一层）的特征图通道数
            out_channels: 精炼模块输出的特征图通道数
            scale_factor: 上采样因子，通常为2
        """
        super(PixelShuffleUpconv, self).__init__()
        self.scale_factor = scale_factor
        self.out_shape=out_shape
        # 0. 上采样: 使用亚像素卷积 (PixelShuffle) 进行上采样
        # 首先用一个卷积层将通道数扩展为 out_channels * (scale_factor ** 2)
        self.conv_for_shuffle = nn.Conv2d(in_channels_decoder, in_channels_decoder * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        #1.降维
        self.conv = ConvModule(in_channels_decoder,in_channels_encoder,1,1)
        # 2. 融合: 将降维后的两个特征图在通道维度上拼接，然后用3x3卷积进行融合和进一步处理
        fuse = []
        for i in range(num_convs):
            in_channel = 2*in_channels_encoder if i == 0 else out_channels
            fuse.append(ConvModule(in_channel, out_channels, kernel_size=3, padding=1))
        self.fuse = nn.Sequential(*fuse)
        # 然后使用PixelShuffle将空间分辨率扩大scale_factor倍，通道数恢复为out_channels

    def forward(self, feat_encoder:torch.Tensor, feat_decoder:torch.Tensor):
        """
        Args:
            feat_encoder: 来自前向传播路径（横向连接）的特征图
            feat_decoder: 来自后向精炼路径（上一层）的特征图，需要先进行上采样或直接使用

        Returns:
            上采样并融合后的特征图
        """
        # 对两个输入特征图进行降维
        _, _, h, w = feat_decoder.shape
        # 计算为了能被 scale_factor 整除，需要在右边和下边填充的像素数
        pad_r = (self.scale_factor - w % self.scale_factor) % self.scale_factor
        pad_b = (self.scale_factor - h % self.scale_factor) % self.scale_factor
        if pad_r > 0 or pad_b > 0:
            feat_decoder = F.pad(feat_decoder, (0, pad_r, 0, pad_b), mode='replicate')
        feat_decoder = self.conv_for_shuffle(feat_decoder)
        upsampled = self.pixel_shuffle(feat_decoder)
        upsampled = upsampled[:, :, :self.out_shape[0], :self.out_shape[1]]
        upsampled = self.conv(upsampled)
        # 在通道维度上拼接
        fused = torch.cat([feat_encoder, upsampled], dim=1)
        # 用3x3卷积进行融合
        fused = self.fuse(fused)
        return fused
    
class RefinementModule(nn.Module):
    """CED网络中的核心精炼模块 (Refinement Module)
    负责融合来自前向传播路径（encoder）和后向精炼路径（decoder）的特征图，
    并使用亚像素卷积进行上采样。
    """
    def __init__(self, in_channels_encoder, in_channels_decoder, out_channels, out_shape,scale_factor=2):
        """
        Args:
            in_channels_encoder: 来自前向传播路径（横向连接）的特征图通道数
            in_channels_decoder: 来自后向精炼路径（上一层）的特征图通道数
            out_channels: 精炼模块输出的特征图通道数
            scale_factor: 上采样因子，通常为2
        """
        super(RefinementModule, self).__init__()
        self.scale_factor = scale_factor
        self.out_shape=out_shape

        # 1. 降维: 分别对来自encoder和decoder的特征图进行1x1卷积降维，以减少参数量
        # 论文中提到将通道数减少到一个较小的值 (kh', ku')
        self.reduce_encoder = ConvModule(in_channels_encoder, out_channels // 2, kernel_size=1)
        self.reduce_decoder = ConvModule(in_channels_decoder, out_channels // 2, kernel_size=1)

        # 2. 融合: 将降维后的两个特征图在通道维度上拼接，然后用3x3卷积进行融合和进一步处理
        self.fuse = ConvModule(out_channels,out_channels, kernel_size=3, padding=1)

        # 3. 上采样: 使用亚像素卷积 (PixelShuffle) 进行上采样
        # 首先用一个卷积层将通道数扩展为 out_channels * (scale_factor ** 2)
        self.conv_for_shuffle = nn.Conv2d(out_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        # 然后使用PixelShuffle将空间分辨率扩大scale_factor倍，通道数恢复为out_channels
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, feat_encoder, feat_decoder):
        """
        Args:
            feat_encoder: 来自前向传播路径（横向连接）的特征图
            feat_decoder: 来自后向精炼路径（上一层）的特征图，需要先进行上采样或直接使用

        Returns:
            上采样并融合后的特征图
        """
        # 对两个输入特征图进行降维
        feat_encoder_reduced = self.reduce_encoder(feat_encoder)
        feat_decoder_reduced = self.reduce_decoder(feat_decoder)

        # 在通道维度上拼接
        fused = torch.cat([feat_encoder_reduced, feat_decoder_reduced], dim=1)
        # 用3x3卷积进行融合
        fused = self.fuse(fused)
        # 1. 计算 PixelShuffle 前需要的填充
        _, _, h, w = fused.shape
        # 计算为了能被 scale_factor 整除，需要在右边和下边填充的像素数
        pad_r = (self.scale_factor - w % self.scale_factor) % self.scale_factor
        pad_b = (self.scale_factor - h % self.scale_factor) % self.scale_factor
        # 2. 应用填充 (只在右和下填充，避免改变内容位置)
        if pad_r > 0 or pad_b > 0:
            fused = F.pad(fused, (0, pad_r, 0, pad_b), mode='replicate')
        # 准备进行亚像素卷积
        # 先通过卷积层扩展通道
        fused = self.conv_for_shuffle(fused)
        # 使用PixelShuffle进行上采样
        upsampled = self.pixel_shuffle(fused)
        if upsampled.shape[-1]<self.out_shape[-1]:
            pad_a = self.out_shape[-2] - upsampled.shape[-2]
            pad_b = self.out_shape[-1] - upsampled.shape[-1]
            upsampled = F.pad(upsampled, (0,pad_a , 0, pad_b), mode='replicate')
        else:
            upsampled = upsampled[:, :, :self.out_shape[0], :self.out_shape[1]]
        return upsampled
    
class PixelShuffleUpconv(nn.Module):
    """CED网络中的核心精炼模块 (Refinement Module)
    负责融合来自前向传播路径（encoder）和后向精炼路径（decoder）的特征图，
    并使用亚像素卷积进行上采样。
    """
    def __init__(self, in_channels_encoder, in_channels_decoder, out_channels, out_shape,scale_factor=2,num_convs=2):
        """
        Args:
            in_channels_encoder: 来自前向传播路径（横向连接）的特征图通道数
            in_channels_decoder: 来自后向精炼路径（上一层）的特征图通道数
            out_channels: 精炼模块输出的特征图通道数
            scale_factor: 上采样因子，通常为2
        """
        super(PixelShuffleUpconv, self).__init__()
        self.scale_factor = scale_factor
        self.out_shape=out_shape
        # 0. 上采样: 使用亚像素卷积 (PixelShuffle) 进行上采样
        # 首先用一个卷积层将通道数扩展为 out_channels * (scale_factor ** 2)
        self.conv_for_shuffle = nn.Conv2d(in_channels_decoder, in_channels_decoder * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        #1.降维
        self.conv = ConvModule(in_channels_decoder,in_channels_encoder,1,1)
        # 2. 融合: 将降维后的两个特征图在通道维度上拼接，然后用3x3卷积进行融合和进一步处理
        fuse = []
        for i in range(num_convs):
            in_channel = 2*in_channels_encoder if i == 0 else out_channels
            fuse.append(ConvModule(in_channel, out_channels, kernel_size=3, padding=1))
        self.fuse = nn.Sequential(*fuse)
        # 然后使用PixelShuffle将空间分辨率扩大scale_factor倍，通道数恢复为out_channels

    def forward(self, feat_encoder:torch.Tensor, feat_decoder:torch.Tensor):
        """
        Args:
            feat_encoder: 来自前向传播路径（横向连接）的特征图
            feat_decoder: 来自后向精炼路径（上一层）的特征图，需要先进行上采样或直接使用

        Returns:
            上采样并融合后的特征图
        """
        # 对两个输入特征图进行降维
        _, _, h, w = feat_decoder.shape
        # 计算为了能被 scale_factor 整除，需要在右边和下边填充的像素数
        pad_r = (self.scale_factor - w % self.scale_factor) % self.scale_factor
        pad_b = (self.scale_factor - h % self.scale_factor) % self.scale_factor
        if pad_r > 0 or pad_b > 0:
            feat_decoder = F.pad(feat_decoder, (0, pad_r, 0, pad_b), mode='replicate')
        feat_decoder = self.conv_for_shuffle(feat_decoder)
        upsampled = self.pixel_shuffle(feat_decoder)
        upsampled = upsampled[:, :, :self.out_shape[0], :self.out_shape[1]]
        upsampled = self.conv(upsampled)
        # 在通道维度上拼接
        _,_,h1,w1 = upsampled.shape
        _,_,h0,w0 = feat_encoder.shape
        pad_w = w0-w1
        pad_h = h0-h1
        if pad_w > 0 and pad_h > 0:
            upsampled = F.pad(upsampled, (0, pad_h, 0, pad_w), mode='replicate')
        elif pad_h<0 and pad_w<0:
            upsampled = upsampled[:,:,:h0,:w0]
        fused = torch.cat([feat_encoder, upsampled], dim=1)
        # 用3x3卷积进行融合
        fused = self.fuse(fused)
        return fused

