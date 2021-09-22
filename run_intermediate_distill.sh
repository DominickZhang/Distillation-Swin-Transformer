#!/bin/bash

epoch=0
output_dir="test_inter_prog"


python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /root/FastBaseline/data/imagenet --teacher /mnt/configblob/users/v-jinnian/swin_distill/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_inter_prog_0 --train_intermediate --stage 0 --output output/$output_dir

for ((i=1;i<=3;i++)); do
    python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /root/FastBaseline/data/imagenet --teacher /mnt/configblob/users/v-jinnian/swin_distill/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_inter_prog_$i --resume output/$output_dir/swin_tiny_patch4_window7_224/test_inter_prog_$((i-1))/ckpt_epoch_$epoch.pth --train_intermediate --stage $i --output output/$output_dir 
done
