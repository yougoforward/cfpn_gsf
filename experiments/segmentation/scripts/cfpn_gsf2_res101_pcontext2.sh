# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model cfpn_gsf2 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname cfpn_gsf2_res101_pcontext2

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn_gsf2 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/cfpn_gsf2/cfpn_gsf2_res101_pcontext2/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn_gsf2 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/cfpn_gsf2/cfpn_gsf2_res101_pcontext2/model_best.pth.tar --split val --mode testval --ms