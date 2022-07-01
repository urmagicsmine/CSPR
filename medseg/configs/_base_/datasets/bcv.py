# dataset settings
dataset_type = 'BCVDataset'
data_root = 'data/BCV/'

#crop_size = (448, 448)
#crop_size = (96, 96)
crop_size = (64, 64)
input_depth = 32

train_pipeline = [
    dict(type='LoadTensorFromFile'),
    #dict(type='TensorNormCropRotateFlip', size=48, train=True, copy_channels=False),
    dict(type='RandomZCrop', crop_depth=input_depth, cat_max_ratio=0.95),
    dict(type='RandomZFlip', prob=0.5),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.75, 1.25)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0), #255 may incur error
    dict(type='DefaultFormatBundle', is_3d_input=True),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadTensorFromFile'),
    #dict(type='DefaultFormatBundle'),
    #dict(type='Collect', keys=['img',]),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        zflip=False,
        flip=False,
        transforms=[
            #dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
            #dict(type='TensorNormCropRotateFlip', size=48, train=False, copy_channels=False),
            dict(type='Resize', keep_ratio=True),
            #dict(type='RandomZFlip'),
            #dict(type='RandomFlip'),
            #dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
            dict(type='DefaultFormatBundle', is_3d_input=True),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=False,
        ann_file='all_files.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        ann_file='all_files.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        ann_file='all_files.txt',
        pipeline=test_pipeline))

