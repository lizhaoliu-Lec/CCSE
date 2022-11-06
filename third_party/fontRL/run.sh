#!/bin/bash

echo train MPNet 
CUDA_VISIBLE_DEVICES=9 python train.py --font=2 \
    --output=model/2 \
    --max_step=1 \
    --batch_size=24 \
    --warmup=0 \
    --noise_factor=0.5 ;
echo train BBoxNet
CUDA_VISIBLE_DEVICES=9 python mover_train.py --font=2 ;
echo test
CUDA_VISIBLE_DEVICES=9 python test.py --font=2 ;
echo concatnate
python concat.py --font=2 ;
