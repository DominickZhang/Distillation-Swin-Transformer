#!/bin/bash
set -e

epoch_num=$1
output_dir=$2
base_lr=$3
ar=$4
data_path=$5
teacher=$6
student_layer_list=$7
teacher_layer_list=$8
accumulation_steps=$9

if [ $accumulation_steps==1 ]
then
batch_size=128
elif [ $accumulation_steps==2 ]
then
batch_size=64
else
echo "accumulation_steps ($accumulation_steps) is not supported!"
exit
fi

echo "python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path $data_path --teacher $teacher --batch-size $((128*accumulation_steps)) --tag joint --joint_distill --ar $ar --student_layer_list $student_layer_list --teacher_layer_list $teacher_layer_list --total_train_epoch $epoch_num --output $output_dir --base_lr $base_lr --load_tar --accumulation-steps $accumulation_steps"


