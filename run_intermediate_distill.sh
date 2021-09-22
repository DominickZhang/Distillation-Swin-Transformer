#!/bin/bash

python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill.yaml --data-path datasets/ --teacher trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_inter_prog_$i --train_intermediate --stage $i --output output/test_inter_prog 

for ((i=1;i<=3;i++)); do
    python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill.yaml --data-path datasets/ --teacher trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_inter_prog_$i --resume output/test_inter_prog/test_inter_prog_$((i-1))/ckpt_epoch_1.pth --train_intermediate --stage $i --output output/test_inter_prog 
done
