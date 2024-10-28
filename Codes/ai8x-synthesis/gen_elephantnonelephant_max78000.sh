#!/bin/sh

DEVICE="MAX78000"
TARGET="/Ubuntu/home/ab/outputs" 
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET \
                  --prefix elephant-nonelephant \
                  --checkpoint-file trained/elephantnonelephant-qat8-q.pth.tar \
                  --config-file networks/elephant-nonelephant-hwc.yaml \
                  --fifo \
                  --softmax \
                  $COMMON_ARGS "$@"
