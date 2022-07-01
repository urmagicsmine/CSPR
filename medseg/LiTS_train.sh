# Run with seed 5 10 15 20 25
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29511 bash ./tools/dist_train.sh ./configs/LiTS_liver/res3dstdunet_res3d18BN_scratch_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly.py 8 \
--seed 25 \
--deterministic \

fuser -k /dev/nvidia*

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29512 bash ./tools/dist_train.sh ./configs/LiTS_liver/res3dstdunet_res3d18BN_acs_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly.py 8 \
--seed 25 \
--deterministic \

fuser -k /dev/nvidia*

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29513 bash ./tools/dist_train.sh ./configs/LiTS_liver/res3dstdunet_res3d18BN_i3d_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly.py 8 \
--seed 25 \
--deterministic \

fuser -k /dev/nvidia*

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29514 bash ./tools/dist_train.sh ./configs/LiTS_liver/res3dstdunet_res3d18BN_imagenet_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly.py 8 \
--seed 25 \
--deterministic \

fuser -k /dev/nvidia*

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29553 bash ./tools/dist_train.sh ./configs/LiTS_liver/res3dstdunet_res3d18BN_mdkinetics_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly.py 8 \
--seed 25 \
--deterministic \

fuser -k /dev/nvidia*

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29516 bash ./tools/dist_train.sh ./configs/LiTS_liver/res3dstdunet_res3d18BN_med3d_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly.py 8 \
--seed 25 \
--deterministic \

fuser -k /dev/nvidia*

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29515 bash ./tools/dist_train.sh ./configs/LiTS_liver/res3dstdunet_res3d18BN_MG_32x256x256_trainnorm1_padata_500eps_lits_liverrsmix_dcomboloss_G8S2_adam_crop1_fulltxt_decodeonly.py 8 \
--seed 25 \
--deterministic \
