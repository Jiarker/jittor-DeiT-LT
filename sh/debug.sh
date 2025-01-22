#! /bin/bash

# CUDA_VISIBLE_DEVICES=0 python train.py \
CUDA_VISIBLE_DEVICES=0  python -m debugpy --listen localhost:5040 --wait-for-client train.py \
    --lr 1e-4 \
    --gamma 0.9 \
    --epochs 40 \
    --weight_decay 1e-4 \
    --momentum 0.9 \
    --accum_iter 1 \
    --log_info Log \
    --save_path ./checkpoints/model.pkl \
    --val_time 5 \
    --model p2t_tiny \
    --dataset CIFAR100 \
    --data_root data/CIFAR100 \
    --batch_size 64 \
    --shuffle False \
    --num_workers 32
