#!/usr/bin/env bash
plugin=fastbev

# config=configs/bevdet/bevdet-r50-cbgs.py
# checkpoint=work_dirs/bevdet-r50-cbgs/epoch_20_ema.pth
# work_dir=work_dirs/bevdet-r50-cbgs 

config=configs/fastbev/paper/fastbev-r50-cbgs.py
checkpoint=work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth
work_dir=work_dirs/


python tools/convert_${plugin}_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16 --int8
