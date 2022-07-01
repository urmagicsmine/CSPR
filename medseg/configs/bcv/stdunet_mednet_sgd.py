_base_ = [
    '../_base_/default_runtime.py',
]

num_classes=14

# model settings
conv_cfg = dict(type = 'Conv3d')
#norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    is_3d_pred=True,
    pretrained='pretrained_model/MedicalNet/resnet_18_23dataset_medicalnet.pth',
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
        kinetics_resnet=True,
        force_downsample=False,
        unet_mode=True,
        stdunet_mode=True),
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
        num_classes=num_classes,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            #type='ComboLoss', use_sigmoid=False, loss_weight=(1.0, 0.5), num_classes = num_classes, smooth=0.001)))
            type='DiceCELoss',
            lambda_ce=0.3)))

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='slide3d', crop_size=(32, 256, 256), stride=(12, 128, 128))
#test_cfg = dict(mode='slide3d', crop_size=(32, 512, 512), stride=(12, 512, 512))

## Dataset
# dataset settings
dataset_type = 'BCVDataset'
data_root = 'data/BCV/'
img_norm_cfg = dict(
    mean=[83.], std=[137.], to_rgb=True)
crop_size = (256, 256)
input_depth = 32
target_slice_spacings =[1.] #[1.] # None
samples_per_gpu = 1

train_pipeline = [
    dict(type='LoadTensorFromFile'),
    #dict(type='LoadAnnotations', slice_input=True, num_slice=input_depth),
    dict(type='RandomZCrop', crop_depth=input_depth, cat_max_ratio=0.95),
    dict(type='RandomZFlip', prob=0.5),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', num_slice=input_depth, **img_norm_cfg), #num_slice must match channel of input
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0), #255 may incur error
    dict(type='DefaultFormatBundle', is_3d_input=True),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadTensorFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        target_slice_spacings=target_slice_spacings,
        zflip=False,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            #dict(type='NormZData'),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomZFlip'),
            dict(type='RandomFlip'),
            dict(type='Normalize', num_slice=input_depth, **img_norm_cfg), #num_slice must match channel of input
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
        #num_slices=64,
        ann_file='all_files.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        disp_per_case=True,
        ann_file='all_files.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        disp_per_case=True,
        ann_file='all_files.txt',
        pipeline=test_pipeline))

##Runtime

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
#optimizer_config = dict()
num_epochs = 13000 # 10x more epochs for bcv dataset, cause bcv only has 21 samples
step1_epochs = 500
step2_epochs = 750
num_gpu = 8
total_samples = 24
save_times = 10
# TODO Neet to Redefine in liver_448x448.py
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False) 
#lr_config = dict(
#    policy='step',
#    step=[int((total_samples * step1_epochs) / (samples_per_gpu * num_gpu)), int((total_samples * step2_epochs) / (samples_per_gpu * num_gpu))]) 
max_iters = int((total_samples * num_epochs) / (samples_per_gpu * num_gpu))
runner = dict(type='IterBasedRunner', max_iters=max_iters)
checkpoint_config = dict(by_epoch=False, interval= max_iters // (save_times))
evaluation = dict(interval=max_iters//save_times, metric='mDice3D')
evaluation = dict(interval=1000, metric='mDice3D')

log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
