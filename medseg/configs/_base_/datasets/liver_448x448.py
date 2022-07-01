# dataset settings
dataset_type = 'NiiLiverDataset'
data_root = 'data/LITS/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (448, 448)
input_depth = 32
target_slice_spacings = None #[1.] # None
samples_per_gpu = 1


train_pipeline = [
    dict(type='LoadVolumeFromFile', window_width=450, window_center=25, ct_type='nii'),
    dict(type='LoadVolumeAnnotations'),
    dict(type='NormZData', target_slice_spacings=target_slice_spacings),
    dict(type='RandomZCrop', crop_depth=input_depth, cat_max_ratio=0.95),
    dict(type='RandomZFlip', prob=0.5),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.75, 1.25)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0), #255 may incur error
    dict(type='DefaultFormatBundle', is_3d_input=True),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadVolumeFromFile', window_width=450, window_center=25),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        target_slice_spacings=target_slice_spacings,
        zflip=False,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='NormZData'),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomZFlip'),
            dict(type='RandomFlip'),
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
        split='train_nii.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        disp_per_case=True, 
        split='valid_nii.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        disp_per_case=True, 
        split='test_nii.txt',
        pipeline=test_pipeline))
