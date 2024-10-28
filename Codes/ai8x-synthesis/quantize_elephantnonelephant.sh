#!/bin/sh
python quantize.py trained/elephantnonelephant-qat8-q.pth.tar r trained/elephantnonelephant-qat8-q.pth.tar --device MAX78000 -v "$@"

