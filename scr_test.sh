#!/bin/bash



python -u scr_test.py \
    --useGPU 0 \
    --seed 2021 \
    --dataset DDR \
    --preprocess 7 \
    --imagesize 512 \
    --net LASNet \
    --model [Absolute path to the *.pth.tar]

