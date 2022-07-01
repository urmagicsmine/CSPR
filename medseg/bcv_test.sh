
CUDA_VISIBLE_DEVICES=1 python ./tools/test.py \
		./configs/bcv/stdunet_sgd.py \
		$path_to_pth \
		--format-only
