#!/bin/bash


python -u scr_train.py \
    --useGPU 1 \
    --seed 2021 \
    --dataset DDR \
    --preprocess 7 \
    --imagesize 512 \
    --withMasks False \
    --keep True \
    --epochs 150 \
    --balanceSample False \
    --net LASNet \
    --scrlr 0.0003 \
    --scrLossSmooth 0.2 \
    --numworkers 8

