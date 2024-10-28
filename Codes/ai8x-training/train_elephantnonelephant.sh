#!/bin/sh
python train.py --epochs 200 \
                --optimizer Adam \
                --lr 0.001 \
                --wd 0 \
                --deterministic \
                --compress policies/schedule-elephantnonelephant.yaml \
                --qat-policy policies/qat_policy_en.yaml \
                --model elephant_cnn \
                --dataset elephant_vs_nonelephant \
                --confusion \
                --param-hist \
                --embedding \
                --device MAX78000 "$@"
