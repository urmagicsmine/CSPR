_base_ = [
    '../_base_/default_runtime.py',
]

# model settings
conv_cfg = dict(type = 'Conv3d')
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    is_3d_pred=True,
    pretrained='pretrained_model/res3d18_imagenet_BN_73.07-5ff2ecb7.pth',
    backbone=dict(
        type='ResNet3D',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        norm_eval=False,
        frozen_stages=-1,
        style='pytorch',
        contract_dilation=True,
        force_downsample=False,
        unet_mode=True),
    neck=dict(   
        type='UNetNeck', 
        in_channels=[64, 64, 128, 256, 512],
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg),    
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=16,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='ComboLoss', use_sigmoid=False, loss_weight=(1.0, 0.5), num_classes = 2, smooth=0.001)))

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='slide3d', crop_size=(64, 512, 512), stride=(24, 512, 512))

## Dataset
# dataset settings
dataset_type = 'OLiverRsDataset'
data_root = 'data/LITS/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
input_depth = 64
target_slice_spacings =[1.] #[1.] # None
samples_per_gpu = 2

train_pipeline = [
    dict(type='LoadImageFromFile', slice_input=True, num_slice=input_depth),
    dict(type='LoadAnnotations', slice_input=True, num_slice=input_depth),
    dict(type='RandomZFlip', prob=0.5),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
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
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        num_slices=64,
        split='liver_norm1_all_dir_pa_range_pa500epochs.txt',
        pipeline=train_pipeline),
    val=dict(
        type='NiiOLiverDataset',
        data_root=data_root,
        disp_per_case=True,
        split='liver_valid_nii.txt',
        pipeline=test_pipeline),
    test=dict(
        type='NiiOLiverDataset',
        data_root=data_root,
        disp_per_case=True,
        split='liver_test_nii.txt',
        pipeline=test_pipeline))

##Runtime

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
num_epochs = 1000
step1_epochs = 500
step2_epochs = 750
num_gpu = 8
total_samples = 105
save_times = 10
# TODO Neet to Redefine in liver_448x448.py
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False) 
max_iters = int((total_samples * num_epochs) / (samples_per_gpu * num_gpu))
runner = dict(type='IterBasedRunner', max_iters=max_iters)
checkpoint_config = dict(by_epoch=False, interval= max_iters // (save_times))
evaluation = dict(interval=max_iters//save_times, metric='mDice3D')

