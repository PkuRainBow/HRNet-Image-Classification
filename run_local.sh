PYTHON="/opt/conda/bin/python"

$PYTHON -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py \
    --cfg experiments/test_cls_hrnet_w18_sgd_lr1e-1_wd1e-4_bs64_x100_dist.yaml
