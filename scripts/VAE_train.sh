#!/bin/bash
# please run this scripts in the train/ directory

# set this to your python binary
# PYTHONBIN='/home/luyuchen.paul/.conda/envs/pytorch/bin/python'
PYTHONBIN='/home/peng/anaconda3/bin/python'

# stdout log
LOGNAME="exp_$(date +"%Y%m%d%H").log"
echo "result is in $LOGNAME"
export PYTHONUNBUFFERED=1

$PYTHONBIN train.py --data /home/peng/ISIC2018_outlier/AKIEC_outlier/vae_train \
    --epochs 45 --lr 1e-4 --batch_size 32 --out_dir AKIEC_result > $LOGNAME &