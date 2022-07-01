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
    pretrained='pretrained_models/ModelGenesis/Genesis_Chest_CT.pt',
    backbone=dict(
        type='UNet3D',
        num_classes=2,
        ),
    decode_head=dict(
        in_channels=1,
        channels=1,
        num_classes=2,
        conv_cfg=conv_cfg,
        type='PseudoHead',
        #align_corners=False,
        loss_decode=dict(
            type='CombineLoss', use_sigmoid=False, loss_weight=(1.0, 0.3)))
        )

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
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
