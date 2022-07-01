# model settings
#norm_cfg = dict(type='SyncBN', requires_grad=True)
#norm_cfg=dict(type='BN3d') # TODO: Need to confirm
#conv_cfg = dict(type = 'Conv3d')
model = dict(
    type='EncoderDecoder',
    is_3d_pred=True,
    pretrained='/data3/lizihao/code/mmclassification/pretrained_models/res3d18_imagenet_BN-64e74d35.pth',
    backbone=dict(
        type='ResNet3D',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=2,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='ComboLoss', use_sigmoid=False, loss_weight=(1.0, 0.5))),
            #type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='ComboLoss', use_sigmoid=False, loss_weight=(1.0 * 0.4, 0.5 * 0.4))))
            #type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
