#num_epochs = 80
#num_gpu = 8
#total_samples = 105 * 27
save_times = 10
max_iters=80000
#sample_per_gpu = 2

# optimizer
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
#optimizer_config = dict()
#optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=max_iters)
checkpoint_config = dict(by_epoch=False, interval= max_iters // save_times)
evaluation = dict(interval=max_iters // save_times, metric='mDice3D')

