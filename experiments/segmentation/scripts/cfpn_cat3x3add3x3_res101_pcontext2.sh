# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model cfpn_cat3x3add3x3 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname cfpn_cat3x3add3x3_res101_pcontext2

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn_cat3x3add3x3 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/cfpn_cat3x3add3x3/cfpn_cat3x3add3x3_res101_pcontext2/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn_cat3x3add3x3 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/cfpn_cat3x3add3x3/cfpn_cat3x3add3x3_res101_pcontext2/model_best.pth.tar --split val --mode testval --ms