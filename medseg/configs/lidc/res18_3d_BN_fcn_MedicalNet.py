_base_ = [
    #'../_base_/models/fcn3d_r18-d8.py',
    '../_base_/datasets/lidc_48x48.py',
    #'../_base_/default_runtime.py',
    #'../_base_/schedules/schedule_40k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
#norm_cfg=dict(type='BN3d')
#norm_cfg=dict(type='GN', num_groups=16)
conv_cfg = dict(type = 'Conv3d')
model = dict(
    type='EncoderDecoder',
    is_3d_pred=True,
    pretrained='pretrained_model/MedicalNet/resnet_18_23dataset.pth',
    backbone=dict(
        type='ResNet3DNoStemStride',
        depth=18,
        num_stages=4,
        unet_mode=True,           # Add the stem features to outputs list.
        out_indices=(0, 1, 2, 3), # indices of res stage 0~3
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 1, 2),
        #avg_down=True,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        norm_eval=False,
        shortcut_type='A',
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='LIDCFCNHead',
        in_channels=(64, 128, 512),
        channels=512,
        input_transform='multiple_select',
        in_index=(0, 2, 4),      # features of stem, res stage1, res stage 3, seperately
        #dropout_ratio=0.0,
        num_classes=2,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            #type='ComboLoss', use_sigmoid=False, loss_weight=(1.0, 0.5))))
            type='CombineLoss', use_sigmoid=False, loss_weight=(1.0, 0.3))))
            #type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
#optimizer_config = dict()
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
num_epochs = 200
num_gpu = 8
total_samples = 2142 # testing 526
save_times = 10
sample_per_gpu = 4
data = dict(samples_per_gpu=4)
max_iters = int((total_samples * num_epochs) / (sample_per_gpu * num_gpu))
runner = dict(type='IterBasedRunner', max_iters=max_iters)
checkpoint_config = dict(by_epoch=False, interval= max_iters // save_times)
evaluation = dict(interval=max_iters // save_times, metric='mDice3D')
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)

# runtime settings
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
