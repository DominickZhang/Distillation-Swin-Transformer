#!/bin/bash
set -e

epoch_num_interm=100
epoch_num_pred=200
output_dir="output/test_relation_all"
#output_dir="/mnt/configblob/users/v-jinnian/swin_distill/relation_baseline/"
base_lr=5e-5
ar=64
data_path="/root/FastBaseline/data/imagenet"
teacher="/mnt/configblob/users/v-jinnian/swin_distill/trained_models/swin_base_patch4_window7_224_22kto1k.pth"
student_layer_list="[11]"
teacher_layer_list="[23]"

#python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path $data_path --teacher $teacher --batch-size 128 --tag intermediate --train_intermediate --ar $ar --student_layer_list $student_layer_list --teacher_layer_list $teacher_layer_list --total_train_epoch $epoch_num_interm --output $output_dir --base_lr $base_lr

python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill.yaml --data-path $data_path --teacher $teacher --batch-size 128 --tag pred_loss --resume $output_dir/swin_tiny_patch4_window7_224/intermediate/ckpt_epoch_$((epoch_num_interm-1)).pth --resume_weight_only --output $output_dir
--total_train_epoch $epoch_num_pred

