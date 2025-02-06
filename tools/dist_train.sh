#!/usr/bin/env bash

# # with acceleration
# python tools/analysis_tools/benchmark.py configs/bevdet/bevdet-sttiny-accelerated.py $checkpoint
# # without acceleration
# python tools/analysis_tools/benchmark.py configs/bevdet/bevdet-sttiny.py $checkpoint
# ./tools/dist_test.sh configs/fastbev/paper/fastbev-r50-cbgs.py work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth 1 --eval mAP 2>&1 | tee work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth.log
# python tools/convert_fastbev_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16 --int8

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
