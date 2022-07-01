# dataset settings
dataset_type = 'BCVDataset'
data_root = 'data/BCV/'

#crop_size = (448, 448)
#crop_size = (96, 96)
#crop_size = (64, 64)
#input_depth = 32

size=64

train_pipeline = [
    dict(type='LoadTensorFromFile'),
    dict(type='TensorNormCropRotateFlip', size=size, mean=83., std=137., train=True, copy_channels=False),
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
            dict(type='TensorNormCropRotateFlip', size=size, mean=83., std=137., train=False, copy_channels=False),
            #dict(type='DefaultFormatBundle', is_3d_input=True),
            dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
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
        size=(size,size,size),
        ann_file='all_files.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        size=(size,size,size),
        ann_file='all_files.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        size=(size,size,size),
        ann_file='all_files.txt',
        pipeline=test_pipeline))

