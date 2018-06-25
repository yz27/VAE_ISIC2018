#!/usr/bin/env bash

# set this to your python binary
export PYTHONBIN='/home/luyuchen.paul/.conda/envs/pytorch/bin/python'
export PYTHONUNBUFFERED=1

# set exp root dir and dataset root dir
export DATA_DIR='/home/luyuchen.paul/NV_outlier/'
export EXP_DIR='/home/luyuchen.paul/exp_results/'

declare -a outliers=("NV")

# loop through outlier classes
for outlier in "${outliers[@]}"
do
    echo "#################TEST ${outlier} ##################"
    $PYTHONBIN outlier_detection.py --cuda \
        --data ${DATA_DIR} \
        --model_path ${EXP_DIR}/${outlier}/best_model.pth.tar \
        --image_size 128
done
