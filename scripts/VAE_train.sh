#!/bin/bash

# set this to your python binary
PYTHONBIN='/home/luyuchen.paul/.conda/envs/pytorch/bin/python'
export PYTHONUNBUFFERED=1

echo "################# TRAIN #####################"
$PYTHONBIN train.py --data /home/luyuchen.paul/ISIC2018_outlier/AKIEC_outlier --cuda \
    --epochs 1 --lr 1e-4 --batch_size 32 --out_dir AKIEC_result
