#!/usr/bin/env bash

cd ..

GPU_ID=4
GPOUP_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --group=${GPOUP_ID} \
    --num_folds=4 \
    --arch=FewSg_v0_0 \
    --lr=3.5e-4 \
    --dataset='voc'
