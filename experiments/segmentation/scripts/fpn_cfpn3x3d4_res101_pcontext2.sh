# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model fpn_cfpn3x3d4 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname fpn_cfpn3x3d4_res101_pcontext2

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fpn_cfpn3x3d4 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/fpn_cfpn3x3d4/fpn_cfpn3x3d4_res101_pcontext2/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fpn_cfpn3x3d4 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/fpn_cfpn3x3d4/fpn_cfpn3x3d4_res101_pcontext2/model_best.pth.tar --split val --mode testval --ms