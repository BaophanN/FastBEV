#!/usr/bin/env bash

PLUGIN=bevdet

VIS_THRESH=0.5
VERSION=val
FORMAT=video
FPS=4
SAVE_PATH=./vis/${PLUGIN}


python ./tools/analysis_tools/vis.py \
       outputs/${PLUGIN}/pts_bbox/results_nusc.json \
       --vis-thred ${VIS_THRESH} \
       --version ${VERSION} \
       --format ${FORMAT} \
       --fps ${FPS} \
       --draw-gt \
       --save_path ${SAVE_PATH} \
       --video-prefix ${PLUGIN}_vis_thresh_0.5
