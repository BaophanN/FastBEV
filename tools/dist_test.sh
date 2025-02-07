#!/usr/bin/env bash

# CONFIG=configs/fastbev/paper/fastbev-r50-cbgs.py 
# CHECKPOINT=work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth
PLUGIN=bevdet
# PLUGIN=fastbev/paper
CONFIG=configs/${PLUGIN}/${PLUGIN}-r50-cbgs.py 
CHECKPOINT=work_dirs/${PLUGIN}-r50-cbgs/epoch_20.pth

GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --format-only --eval-options \
    jsonfile_prefix=outputs/${PLUGIN} | tee work_dirs/${PLUGIN}-r50-cbgs/epoch_20.pth.log