#### GPU07
## run distillation with pred loss only
 # python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill.yaml --data-path /sdb/imagenet --teacher ~/trained_models/swin_base_patch4_window7_224_22kto1k.pth --batch-size 128 --tag debug_da_trial_0 

## run distillation with intermediate loss
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /sdb/imagenet --teacher ~/trained_models/swin_base_patch4_window7_224_22kto1k.pth --batch-size 128 --train_intermediate --ar 64 --student_layer_list 11 --teacher_layer_list 23 --total_train_epoch 5 --tag debug_da_trial_0 

## run joint distillation
python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_joint.yaml --data-path /sdb/imagenet --teacher ~/trained_models/swin_base_patch4_window7_224_22kto1k.pth --batch-size 128 --joint_distill --ar 64 --student_layer_list 11 --teacher_layer_list 23 --total_train_epoch 5 --tag debug_joint_distill


#### PAI
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /root/FastBaseline/data/imagenet --teacher /mnt/configblob/users/v-jinnian/swin_distill/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_inter_all --train_intermediate --ar 48 --student_layer_list 11 --teacher_layer_list 23

