# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model fpn3x3 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname fpn3x3_res101_pcontext2

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fpn3x3 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/fpn3x3/fpn3x3_res101_pcontext2/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fpn3x3 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/fpn3x3/fpn3x3_res101_pcontext2/model_best.pth.tar --split val --mode testval --ms