CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash ./tools/dist_test.sh \
./configs/LiTS_liver/res3dstdunet_res3d18BN_med3d_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly.py \
./work_dirs/LiTS_liver/res3dstdunet_res3d18BN_med3d_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly/iter_3280.pth 8 \
--eval 'mDice3D' \
--post-process
