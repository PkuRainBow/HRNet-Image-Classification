
df
cp -r ../../../../dataset/imagenet_zhirong /dev/shm/

PYTHON="/opt/conda/bin/python"

NUM_GPUS=$1
CONFIG="$2.yaml"

$PYTHON -m pip install termcolor
$PYTHON -m pip install yacs

# if [ "$3"x == "1"x ]; then
    # $PYTHON -m torch.distributed.launch \
    #     --nproc_per_node=$NUM_GPUS \
    #     tools/train.py \
    #     --cfg experiments/$CONFIG

# elif [ "$3"x == "2"x ]; then
#     export auto_mix_precision=0

#     $PYTHON -m torch.distributed.launch \
#         --nproc_per_node=$NUM_GPUS \
#         tools/train.py \
#         --cfg experiments/$CONFIG

# elif [ "$3"x == "3"x ]; then
#     export distribute_training=0
#     export auto_mix_precision=0

#     $PYTHON tools/train_orig.py \
#         --cfg experiments/$CONFIG

# else
#   echo "$3"x" is unrecognized settings!"
# fi

$PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    tools/train.py \
    --cfg experiments/$CONFIG
