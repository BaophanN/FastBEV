#!/usr/bin/env bash
#find / -name trtexec 2>/dev/null
# /opt/tensorrt/bin/trtexec
# /workspace/tensorrt/bin/trtexec

    # --int8 \
/workspace/tensorrt/bin/trtexec \
    --onnx=/workspace/source/advanced-fastbev/work_dirs/fastbev_fp16_fuse.onnx \
    --fp16 \
    --workspace=2048 \
    --iterations=20 \
    --verbose \
    --dumpProfile \
    --dumpLayerInfo \
    --saveEngine=/workspace/source/advanced-fastbev/work_dirs/fastbev_fp16_fuse.trt | tee work_dirs/fastbev-r50-cbgs/fastbev_fp16_fuse.log