#!/bin/sh
python train.py --model elephant_cnn \
                --dataset elephant_vs_nonelephant \
                --confusion \
                --evaluate \
                --exp-load-weights-from ../ai8x-synthesis/trained/elephantnonelephant-qat8-q.pth.tar \
                -8 \
                --device MAX78000 "$@"
