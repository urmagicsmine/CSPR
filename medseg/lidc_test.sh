CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash ./tools/dist_test.sh \
./configs/lidc/res18_3d_BN_fcn.py \
./path_to_pth 4 \
--format-only \

#--eval 'mDice3D'

