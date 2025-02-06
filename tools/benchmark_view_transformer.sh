#!/usr/bin/env bash


# config=configs/fastbev/paper/fastbev-r50-cbgs.py
# checkpoint=work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth

config=configs/bevdet/bevdet-r50-cbgs.py
checkpoint=work_dirs/bevdet-r50-cbgs/epoch_20.pth


# python tools/analysis_tools/benchmark_view_transformer_fastray.py $config $checkpoint
python tools/analysis_tools/benchmark_view_transformer.py $config $checkpoint
