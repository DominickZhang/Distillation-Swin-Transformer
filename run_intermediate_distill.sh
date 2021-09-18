#!/bin/bash

for ((i=0;i<=3;i++)); do
    python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  distillation_v2_jinnian.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /sdb/imagenet --teacher ~/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_bash --train_intermediate --stage $i
done
