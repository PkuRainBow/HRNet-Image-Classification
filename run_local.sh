PYTHON="/opt/conda/bin/python"
# PYTHON="/data/anaconda/envs/torch0.4/bin/python"

$PYTHON -m pip install termcolor
$PYTHON -m pip install yacs

# $PYTHON -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     tools/train.py \
#     --cfg experiments/test_cls_hrnet_w18_sgd_lr1e-1_wd1e-4_bs64_x100_dist.yaml

NUM_GPUS=$1
CONFIG="$2.yaml"

# test_cls_se_hrnet_w18_sgd_lr1e-1_wd1e-4_bs64_x100_dist
# $PYTHON -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --use_env tools/train_orig.py \
#     --cfg experiments/$CONFIG

# cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100

export distribute_training=0
export auto_mix_precision=0

$PYTHON tools/train_orig.py \
    --cfg experiments/$CONFIG