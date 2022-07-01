# data split of BCV is as same as that in UNETR
# see mmseg/datasets/bcv.py

# training bash
# num of gpus
GPU=4
bash ./tools/dist_train.sh stdunet_25d_sgd.py $GPU
bash ./tools/dist_train.sh stdunet_acs_sgd.py $GPU
bash ./tools/dist_train.sh stdunet_i3d_sgd.py $GPU
bash ./tools/dist_train.sh stdunet_kinetics_sgd.py $GPU
bash ./tools/dist_train.sh stdunet_mednet_sgd.py $GPU
bash ./tools/dist_train.sh stdunet_MG_sgd.py $GPU
bash ./tools/dist_train.sh stdunet_scratch_sgd.py $GPU
bash ./tools/dist_train.sh stdunet_sgd.py $GPU
