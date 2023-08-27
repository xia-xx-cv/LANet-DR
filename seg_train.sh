#!/bin/bash



python -u train_seg.py  \
    --useGPU 1 \
    --seed 2021 \
    --dataset DDR_seg \
    --preprocess 7  \
    --balanceSample False \
    --net LANet \
    --posw 10 \
    --lr 0.001 

