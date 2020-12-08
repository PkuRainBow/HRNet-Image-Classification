
df
cp -r ../../../../dataset/imagenet_zhirong /dev/shm/

NUM_GPUS=$0
CONFIG="$1.yaml" #cls_hrnet_w18_sgd_lr1e-1_wd1e-4_bs64_x100_test_dist.yaml

$PYTHON -m torch.distributed.launch \
    --nproc_per_node=$0 \
    tools/train.py \
    --cfg experiments/$1
