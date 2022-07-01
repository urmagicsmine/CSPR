# model settings
conv_cfg = dict(type = 'Conv3d')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
#norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    is_3d_pred=True,
    pretrained='/lung_general_data/pretrained_model/mr3d/mr3d34_ms640_36.9-2747d8e1.pth',
    backbone=dict(
        type='ResNet3D',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        norm_eval=False,
        frozen_stages=-1,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        dilations=(1, 2, 4, 8), #(1, 12, 24, 36)
        dropout_ratio=0.1,
        num_classes=3,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
