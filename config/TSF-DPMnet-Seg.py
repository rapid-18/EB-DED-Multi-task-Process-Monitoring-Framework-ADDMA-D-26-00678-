num_stages=5
base_channel=48
base_channel_aux=16
# 模型配置（输入通道=1）
model = dict(
    type='EncoderDecoder',
    test_cfg=dict(mode='whole'),
    backbone=dict(
        type='Encoder',
        pretraining_task=False,
        backbone_config=dict(
            type='DualBranch_Backbone',
            backbone_name='basic',
            left_first=True,
            backbone_config=dict(
                in_channels=1,
                base_channels=48,
                num_stages=num_stages,
                strides=tuple((1 for i in range(num_stages))),
                enc_num_convs=tuple((2 for i in range(num_stages))),
                downsamples=tuple((True for i in range(num_stages-1))),
                enc_dilations=tuple((1 for i in range(num_stages))),
            ),
        aux_backbone_config=dict(
                in_channels=1,
                base_channels=base_channel_aux,
                num_stages=num_stages,
                strides=tuple((1 for i in range(num_stages))),
                enc_num_convs=tuple((2 for i in range(num_stages))),
                downsamples=tuple((True for i in range(num_stages-1))),
                enc_dilations=tuple((1 for i in range(num_stages))),
        ),
            ),
        fuse_config=None,
    ),
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[0],
        std=[255],
        size=(size,size)
    ),
    decode_head=dict(
        type='DualBracnh_SegmentHead',
        base_channels=base_channel,
        base_channels_aux=base_channel_aux,
        output_shape=(size,size),
        BlockType='Unet',
        time_step=timestep,
        BlockConfig=dict(dec_num_convs=tuple((2 for i in range(num_stages)))),
        num_convs=1,
        dropout_ratio=0.1,
        fuse_indices = [(0,),(1,2,3,4)],
        st_fuse=[
            'MSPag',
            ],
        st_fuse_config=[
                        dict(ratio=2,use_soft_max=False,add=True,kernels=[1,3],fixed_weight=True,add_fixed=False, 
                        use_activate=True,dilations=[1,1],with_channel=True,depthwise=False,As=True,At=False),
                        dict(ratio=ratio,use_soft_max=use_softmax,add=add,kernels=kernels,fixed_weight=Fixed_weight,add_fixed=add_Fixed, 
                             use_activate=use_activate,dilations=diliations,with_channel=with_channels,adaptive_weight=adaptive_weight,depthwise=False,As=As,At=At),
                         ],
        temporal_agg='DP',
        agg_config=dict(avg_pool=True,max_pool=True,num_convs=1,pool_cat=False,include_first=True),
        num_stages=num_stages,
        num_classes=3,
        backbone_type='basic',
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=2.0)
            # dict(type='WeightedDiceLoss', use_sigmoid=False, loss_weight=2.0,dice_weight=[0.3,1.0,1.0])
            ]
    )
)
