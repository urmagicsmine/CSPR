fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='ImageClassifier',
    pretrained='pretrained_model/res3d18_imagenet_BN_73.07-5ff2ecb7.pth',
    backbone=dict(
        type='ResNet3D',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        depth_stride=True,      # consistent with ACS
        stem_stride=True,
        strides=(1, 2, 2, 2),   # consistent with ACS
        in_channels=1,
        conv_cfg=dict(type='Conv3d'),
        #norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        norm_cfg=dict(type='BN3d'),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling', use_3d_gap=True),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
    ))
# dataset settings
dataset_type = 'LIDCDataset'
img_norm_cfg = dict(
    mean=[114.495]*3, std=[57.63]*3, to_rgb=True)

train_pipeline = [
    # 1. random crop 2. rotation 3. reflection(flip by 3 axis)
    dict(type='LoadTensorFromFile'),
    dict(type='TensorNormCropRotateFlip', crop_size=48, move=5, train=True),
    dict(type='ToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadTensorFromFile'),
    dict(type='TensorNormCropRotateFlip', crop_size=48, move=5, train=False),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/lidc/',
        ann_file='train_test_split.csv',
        sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/lidc/',
        ann_file='train_test_split.csv',
        sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/lidc/nodule',
        ann_file='train_test_split.csv',
        sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=test_pipeline))

evaluation = dict(interval=2, metric='all',
        metric_options=dict(topk=(1, 2)) )

# checkpoint saving
checkpoint_config = dict(interval=50)
# yapf:disable
log_config = dict(
    interval=7,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[30, 60, 90], warmup='constant', warmup_iters=50)
runner = dict(type='EpochBasedRunner', max_epochs=100)
