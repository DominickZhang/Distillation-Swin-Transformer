# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

from main import parse_option, validate, throughput
#from models.swin_transformer import SwinTransformer
from models.swin_transformer_distill_jinnian import SwinTransformerDistill

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            #print('teacher:', torch.max(targets_prob), torch.argmax(targets_prob))
            #print('student:', torch.max(student_likelihood), torch.argmax(targets_prob))
            loss_batch = torch.sum(- targets_prob * student_likelihood, dim=1)
            #print(loss, targets_prob.shape, loss.mean())
            #input()
            return loss_batch.mean()


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Loading teacher model:{config.MODEL.TYPE}/{config.DISTILL.TEACHER}")
    model_teacher = load_teacher_model()
    model_teacher.cuda()

    '''
    optimizer = build_optimizer(config, model_teacher)
    if config.AMP_OPT_LEVEL != "O0":
        model_teacher, optimizer = amp.initialize(model_teacher, optimizer, opt_level=config.AMP_OPT_LEVEL)ll   
    '''
    model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

    if config.DISTILL.DO_DISTILL:
        checkpoint = torch.load(config.DISTILL.TEACHER, map_location='cpu')
        msg = model_teacher.module.load_state_dict(checkpoint['model'], strict=False)
        logger.info(msg)
        del checkpoint
        torch.cuda.empty_cache()

    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.DISTILL.TRAIN_INTERMEDIATE:
        lr_scheduler = None
    else:
        if len(config.DISTILL.INTERMEDIATE_CHECKPOINT):
            checkpoint = torch.load(config.DISTILL.INTERMEDIATE_CHECKPOINT,  map_location='cpu')
            msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            logger.info(msg)
            torch.cuda.empty_cache()
            if config.EVAL_MODE:
                #validate(config, data_loader_val, model, logger, is_intermediate=True, model_teacher=model_teacher)
                validate(config, data_loader_train, model, logger, is_intermediate=True, model_teacher=model_teacher)
                return
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    '''
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    '''
    criterion = soft_cross_entropy
    
    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        #max_accuracy = load_checkpoint(config, model_without_ddp, None, None, logger)
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        if config.DISTILL.TRAIN_INTERMEDIATE:
            train_one_epoch_intermediate_distill(config, model, model_teacher, criterion, data_loader_train, optimizer, epoch, mixup_fn)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        else:
            train_one_epoch_distill(config, model, model_teacher, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

            acc1, acc5, loss = validate(config, data_loader_val, model, logger)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def load_teacher_model():
    embed_dim = 192
    depths = [ 2, 2, 18, 2 ]
    num_heads = [ 6, 12, 24, 48 ]
    window_size = 7
    model = SwinTransformerDistill(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1000,
                                embed_dim=embed_dim,
                                depths=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=4.0,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                # distillation
                                is_student=False,
                                return_midlayer_num = [2, 2, 6, 2],
                                is_return_all_layers=True
                                )
    return model

def train_one_epoch_intermediate_distill(config, model, model_teacher, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler=None):
    #total_epoch = config.TRAIN.EPOCHS
    #layer_stage = epoch // 25 ## 25 epochs for each stage
    layer_stage = 1
    if epoch%25 == 0:
        logger.info("Training stage: %d..."%layer_stage)
    lr_stage_decay_weight = 1e-2
    hidden_loss_weight = [10, 10, 10, 1.0]
    #pred_loss_weight = 1e-10 # different from DataParallel, DistributedDataParallel will throw an error of unused parameters or unused outputs of model in the loss. Therefore, here we still calculate the prediction with a small weight to temporarily avoid this error

    model.train()
    optimizer.zero_grad()

    model_teacher.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    attn_loss_meter = AverageMeter()
    hidden_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    def cal_intermediate_loss(student_attn_list, student_hidden_list,
                            teacher_attn_list, teacher_hidden_list, is_debug=False):
        N = len(student_attn_list)
        ## Attention loss
        attn_loss = 0.
        for student_att, teacher_att in zip(student_attn_list, teacher_attn_list):
            if is_debug:
                print(student_att[0,0,0,:], teacher_att[0,0,0,:])
                input('paused!')
            #student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(student_att.device), student_att)
            #teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(teacher_att.device),teacher_att)
            #tmp_loss = torch.nn.MSELoss()(student_att, teacher_att)
            tmp_loss = torch.nn.L1Loss()(student_att, teacher_att)
            attn_loss += tmp_loss
        ## Hidden loss
        hidden_loss = 0.
        for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
            #tmp_loss = torch.nn.MSELoss()(student_hidden, teacher_hidden)
            if is_debug:
                print(student_hidden[0,0,:], teacher_hidden[0,0,:])
                #print(student_hidden.shape) #128*3136*192
                input('paused!')
            tmp_loss = torch.nn.L1Loss()(student_hidden, teacher_hidden)
            hidden_loss += tmp_loss
        return attn_loss/N, hidden_loss/N

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        attn_outputs, hidden_outputs = model(samples, layer_stage)

        with torch.no_grad():
            attn_outputs_teacher, hidden_outputs_teacher = model_teacher(samples, layer_stage)

        attn_loss, hidden_loss = cal_intermediate_loss(attn_outputs, hidden_outputs, attn_outputs_teacher, hidden_outputs_teacher)
        loss = attn_loss + hidden_loss_weight[layer_stage]*hidden_loss

        optimizer.zero_grad()
        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        else:
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())

        if dist.get_rank() == 0:
            #print(len(optimizer.param_groups[0]['params'])+len(optimizer.param_groups[1]['params'])) #189
            print(optimizer)
            #print(len(list(model.named_parameters()))) #189
            print(list(model.named_parameters())[-8:-4])
            #print(model.fit_dense_C)
            #print(model.module.features[0].grad)
        
        optimizer.step()

        if dist.get_rank() == 0:
            #print(len(optimizer.param_groups[0]['params'])+len(optimizer.param_groups[1]['params']))
            #print(optimizer, optimizer.param_groups[0])
            #print(model.module)
            #print(model.module.features[0].weight)
            print(list(model.named_parameters())[-8:-4])
            #print(model.fit_dense_C)
            print(model.module.features[0].grad)
        input('paused!')

        if lr_scheduler is not None:
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        attn_loss_meter.update(attn_loss.item(), targets.size(0))
        hidden_loss_meter.update(hidden_loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'attn_loss {attn_loss_meter.val:.4f} ({attn_loss_meter.avg:.4f})\t'
                f'hidden_loss {hidden_loss_meter.val:.4f} ({hidden_loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_one_epoch_distill(config, model, model_teacher, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    model_teacher.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    pred_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        with torch.no_grad():
            outputs_teacher = model_teacher(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            #loss = criterion(outputs, targets)
            loss_pred = criterion(outputs/config.DISTILL.TEMPERATURE,
                            outputs_teacher/config.DISTILL.TEMPERATURE)

            loss = loss_pred / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            #loss = criterion(outputs, targets)
            loss_pred = criterion(outputs/config.DISTILL.TEMPERATURE,
                            outputs_teacher/config.DISTILL.TEMPERATURE)
            
            loss = loss_pred

            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        pred_loss_meter.update(loss_pred.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'pred_loss {pred_loss_meter.val:.4f} ({pred_loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")





if __name__ == '__main__':
    ## eval:
    # python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  distillation_v2_jinnian.py --do_distill --eval --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /sdb/imagenet --teacher ~/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 192 --tag dist_intermediate_eval --intermediate_checkpoint ~/trained_models/swin_tiny_intermediate.pth

    ## train:
    # python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  distillation_v2_jinnian.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_v2.yaml --data-path datasets/ --teacher trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag dist_v2
    # python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  distillation_v2_jinnian.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path datasets/ --teacher trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag dist_v2 --train_intermediate
    # CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  distillation_v2_jinnian.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /sdb/imagenet  --teacher ~/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test --train_intermediate
    # python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 1234  distillation_v2_jinnian.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_v2.yaml --data-path datasets/ --teacher trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag dist_v2 --intermediate_checkpoint trained_models/swin_tiny_intermediate.pth

    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
