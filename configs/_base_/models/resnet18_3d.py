# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet3D',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
