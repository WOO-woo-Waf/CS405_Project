#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0 python train.py --model fcn16s \
    --backbone vgg16 --dataset my_city \
    --lr 0.0001 --epochs 80 --skip-val