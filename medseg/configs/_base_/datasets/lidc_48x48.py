# dataset settings
dataset_type = 'LidcDataset'
data_root = 'data/LIDC/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (48, 48, 48)
samples_per_gpu = 1

train_pipeline = [
    dict(type='LoadPairDataFromFile'),
    dict(type='RandomCenterCrop', crop_size=crop_size, move=5),
    dict(type='RandomRotate3D'),
    dict(type='RandomZFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', is_3d_input=True),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadPairDataFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        zflip=False,
        flip=False,
        transforms=[
            #dict(type='Resize', keep_ratio=True),
            dict(type='RandomZFlip'),
            dict(type='RandomFlip'),
            dict(type='RandomCenterCrop', crop_size=crop_size, move=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='train_split.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        crop_size=(48, 48, 48),
        disp_per_case=False,
        split='test_split.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        crop_size=(48, 48, 48),
        disp_per_case=False,
        split='test_split.txt',
        pipeline=test_pipeline))
