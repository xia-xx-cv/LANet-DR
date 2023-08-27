#!/bin/bash



python -u seg_test.py \
    --useGPU 1 \
    --seed 2021 \
    --dataset DDR_seg \
    --preprocess 7 \
    --imagesize 512 \
    --net LANet \
    --model [Absolute path to the *.pth.tar]


