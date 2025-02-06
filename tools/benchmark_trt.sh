#!/usr/bin/env bash
plugin=bevdet

# config=configs/fastbev/paper/fastbev-r50-cbgs.py
# engine=work_dirs/fastbev-r50-cbgsfastbev_int8_fuse.engine

config=configs/bevdet/bevdet-r50-cbgs.py
engine=work_dirs/bevdet_trtbevdet_int8_fuse.engine

python tools/analysis_tools/benchmark_trt.py $config $engine