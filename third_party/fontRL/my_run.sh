#!/bin/bash

echo train MPNet 
CUDA_VISIBLE_DEVICES=4 python train.py --font=2 \
    --output=model/2 \
    --max_step=1 \
    --batch_size=24 \
    --warmup=0 \
    --noise_factor=0.5 ;
echo train BBoxNet
CUDA_VISIBLE_DEVICES=4 python mover_train.py --font=2 ;
echo test
CUDA_VISIBLE_DEVICES=4 python test.py --font=2 ;
echo concatnate
python concat.py --font=2 ;

# imageNet pretrained resnet 50
# imageNetMover, gpu023
DEAD CUDA_VISIBLE_DEVICES=3 python mover_train.py --font=2 --run-id imageNet --depth 50 --pretrained /home/liulizhao/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
DEAD CUDA_VISIBLE_DEVICES=3 python mover_train.py --font=2 --run-id imageNet2 --depth 50 --pretrained /home/liulizhao/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
# gpu023 card5 imageNetBS96
CUDA_VISIBLE_DEVICES=5 python mover_train.py --font=2 --run-id imageNetBS96 --depth 50 --pretrained /home/liulizhao/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth --bs 96


# hw pretrained resnet 50
# hwMover, gpu023
DEAD 被占卡 CUDA_VISIBLE_DEVICES=5 python mover_train.py --font=2 --run-id hw --depth 50 --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth --d2
DEAD CUDA_VISIBLE_DEVICES=5 python mover_train.py --font=2 --run-id hw2 --depth 50 --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth --d2
# gpu021 card7 hwBS96
DEAD CUDA_VISIBLE_DEVICES=7 python mover_train.py --font=2 --run-id hwBS96 --depth 50 --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth --d2 --bs 96


# hw noPretrained
# noMover, gpu023
DEAD CUDA_VISIBLE_DEVICES=6 python mover_train.py --font=2 --run-id no --depth 50
# gpu018 card0 noBS96
CUDA_VISIBLE_DEVICES=0 python mover_train.py --font=2 --run-id nobs96 --depth 50 --bs 96


# RL gpu22
# warmupMPNET
DEAD CUDA_VISIBLE_DEVICES=7 python train.py --font=2 --output=model --max_step=1 --batch_size=24 --warmup=50 --noise_factor=0.5 --run_id=3
# gpu018 card0 RLbs128
CUDA_VISIBLE_DEVICES=1 python train.py --font=2 --output=model --max_step=1 --batch_size=128 --noise_factor=0.5 --run_id=bs128 --warmup=0

=========================> whole new sep <=========================

# gpu022
# ImageNetBS98Val tmux alias as testVal
CUDA_VISIBLE_DEVICES=1 python mover_train.py --font 2 --run-id ImageNetBS98Val \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth \
                                              --bs 96

# gpu021
# hwBS96Val
CUDA_VISIBLE_DEVICES=7 python mover_train.py --font=2 --run-id hwBS96Val  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2

# hwBS96ValRecover
CUDA_VISIBLE_DEVICES=7 python mover_train.py --font=2 --run-id hwBS96ValRecover  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2

# hwBS96ValStep50Wd1e-4
CUDA_VISIBLE_DEVICES=6 python mover_train.py --font=2 --run-id hwBS96ValStep50Wd1e-4  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --step 50 --wd 1e-4

# hwBS96ValWd1e-2
CUDA_VISIBLE_DEVICES=6 python mover_train.py --font=2 --run-id hwBS96ValWd1e-2  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --wd 1e-2

# gpu022
# hwBS96FreezeBNVal
CUDA_VISIBLE_DEVICES=1 python mover_train.py --font=2 --run-id hwBS96FreezeBNVal  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --f-bn

# gpu021 card7
# hwBS96ValDrop5e-1
CUDA_VISIBLE_DEVICES=7 python mover_train.py --font=2 --run-id hwBS96ValDrop5e-1  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --dropout 0.5

# gpu022
# noNS98Val
CUDA_VISIBLE_DEVICES=5 python mover_train.py --font=2 --run-id noBS96Val  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --bs 96

# gpu022
# KaiBS96Val
CUDA_VISIBLE_DEVICES=6 python mover_train.py --font=2 --run-id KaiBS96Val  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /mnt/cephfs/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_kaiti/20220805.001225/model_0214999.pth \
                                              --bs 96 --d2

# gpu022
# HWKaiHWBS96Val
CUDA_VISIBLE_DEVICES=7 python mover_train.py --font=2 --run-id HWKaiHWBS96Val  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /mnt/cephfs/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_hwk2hw/20220805.175855/model_0239999.pth \
                                              --bs 96 --d2

# gpu021
# hwBS96SGDVal
CUDA_VISIBLE_DEVICES=6 python mover_train.py --font=2 --run-id hwBS96SGDVal  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --optim sgd

# gpu023
# hwBS96SGDLR1e-3Val
CUDA_VISIBLE_DEVICES=3 python mover_train.py --font=2 --run-id hwBS96SGDLR1e-3Val  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --optim sgd --lr 1e-3


# gpu023， card3, 4, 5
# lr=1e-3
# hwBS96ValLR1e-3
CUDA_VISIBLE_DEVICES=3 python mover_train.py --font=2 --run-id hwBS96ValLR1e-3  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --lr 1e-3

# noNS96ValLR1e-3
CUDA_VISIBLE_DEVICES=4 python mover_train.py --font=2 --run-id noNS96ValLR1e-3  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --bs 96 --lr 1e-3

# ImageNetBS98ValLR1e-3
CUDA_VISIBLE_DEVICES=5 python mover_train.py --font 2 --run-id ImageNetBS98ValLR1e-3 \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth \
                                              --bs 96 --lr 1e-3

# gpu021
# ImageNetBS96ValAug2e-1
CUDA_VISIBLE_DEVICES=0 python mover_train.py --font 2 --run-id ImageNetBS96ValAug2e-1 \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth \
                                              --bs 96 --aug-prob 0.2

# gpu021
# hwBS96ValAug2e-1
CUDA_VISIBLE_DEVICES=1 python mover_train.py --font=2 --run-id hwBS96ValAug2e-1  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --aug-prob 0.2

# gpu021
# noNS96ValAug2e-1
CUDA_VISIBLE_DEVICES=6 python mover_train.py --font=2 --run-id noNS96ValAug2e-1  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --bs 96 --aug-prob 0.2

# gpu021
# ImageNetBS96ValAug2e-1Step1e3LR1e-5
CUDA_VISIBLE_DEVICES=4 python mover_train.py --font 2 --run-id ImageNetBS96ValAug2e-1Step1e3LR1e-5 \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth \
                                              --bs 96 --aug-prob 0.2 --step 1000 --lr 1e-5

# gpu021
# noNS96ValAug2e-1Step1e3LR1e-5
CUDA_VISIBLE_DEVICES=7 python mover_train.py --font=2 --run-id noNS96ValAug2e-1Step1e3LR1e-5  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --bs 96 --aug-prob 0.2 --step 1000 --lr 1e-5


# gpu018
# hwBS96ValLR1e-3Step1e3LR1e-5
CUDA_VISIBLE_DEVICES=0 python mover_train.py --font=2 --run-id hwBS96ValLR1e-3Step1e3LR1e-5  \
                                              --depth 50 --num-val 1000 \
                                              --actor-path /home/liulizhao/projects/HWCQA/third_party/fontRL/model/2/Paint-run3/actor-4801.pkl \
                                              --pretrained /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth \
                                              --bs 96 --d2 --lr 1e-3 --step 1000 --lr 1e-5