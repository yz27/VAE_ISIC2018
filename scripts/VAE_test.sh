#!/usr/bin/env bash
# set this to your python binary
PYTHONBIN='/home/luyuchen.paul/.conda/envs/pytorch/bin/python'
export PYTHONUNBUFFERED=1

echo "################# TEST #######################"
$PYTHONBIN test.py --data /home/luyuchen.paul/ISIC2018_outlier/AKIEC_outlier --cuda \
    --model_path ./AKIEC_result/best_model.pth.tar
