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

#from models.swin_transformer import SwinTransformer
#from models.swin_transformer_distill_jinnian import SwinTransformerDistill
from models.swin_transformer_distill_relation import SwinTransformerRelation

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # Jinnian: add distillation parameters
    parser.add_argument('--do_distill', action='store_true', help='start distillation')
    parser.add_argument('--teacher', default='', type=str, metavar='PATH', help='the path for teacher model')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='the temperature for distillation loss')
    parser.add_argument('--train_intermediate', action='store_true', help='whether to train with intermediate loss')
    parser.add_argument('--progressive', action='store_true', help='whether to use progressive distillation')
    parser.add_argument('--stage', default=0, type=int, help='the index of stage in Swin Transformer to be trained')
    parser.add_argument('--alpha', default=0.0, type=float, help='the weight to balance the soft label loss and ground-truth label loss')
    parser.add_argument('--load_tar', action='store_true', help='whether to load data from tar files')
    parser.add_argument('--ar', default=1, type=int, help='The number of relative heads')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            #print('teacher:', torch.max(targets_prob), torch.argmax(targets_prob))
            #print('student:', torch.max(student_likelihood), torch.argmax(targets_prob))
            #loss_batch = torch.sum(- targets_prob * student_likelihood, dim=-1)
            #print(loss, targets_prob.shape, loss.mean())
            #input()
            return (- targets_prob * student_likelihood).mean()

def cal_relation_loss(student_attn_list, teacher_attn_list, Ar):
    N = len(student_attn_list)
    relation_loss = 0.
    for student_att, teacher_att in zip(student_attn_list, teacher_attn_list):
        B, N, Cs = student_att.shape
        _, _, Ct = teacher_att.shape
        for i in range(3):
            for j in range(3):
                As_ij = (student_att[i].reshape(B, N, Ar, Cs//Ar).transpose(1, 2))@(student_att[j].reshape(B, N, Ar, Cs//Ar).permute(0, 2, 3, 1)) / (Cs/Ar)**0.5
                At_ij = (teacher_att[i].reshape(B, N, Ar, Ct//Ar).transpose(1, 2))@(teacher_att[j].reshape(B, N, Ar, Ct//Ar).permute(0, 2, 3, 1)) / (Ct/Ar)**0.5
                relation_loss += soft_cross_entropy(As_ij, At_ij)
    return relation_loss/(9. * N)


def cal_intermediate_loss(student_attn_list, student_hidden_list,
                            teacher_attn_list, teacher_hidden_list, is_debug=False):
        N = len(student_attn_list)
        ## Attention loss
        attn_loss = 0.
        for student_att, teacher_att in zip(student_attn_list, teacher_attn_list):
            if is_debug:
                print(student_att[0,0,0,:], teacher_att[0,0,0,:])
                input('paused!')
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(student_att.device), student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(teacher_att.device), teacher_att)
            tmp_loss = torch.nn.MSELoss()(student_att, teacher_att)
            #tmp_loss = torch.nn.L1Loss()(student_att, teacher_att)
            attn_loss += tmp_loss
        ## Hidden loss
        hidden_loss = 0.
        for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
            #tmp_loss = torch.nn.MSELoss()(student_hidden, teacher_hidden)
            if is_debug:
                print(student_hidden[0,0,:], teacher_hidden[0,0,:])
                #print(student_hidden.shape) #128*3136*192
                input('paused!')
            tmp_loss = torch.nn.MSELoss()(student_hidden, teacher_hidden)
            #tmp_loss = torch.nn.L1Loss()(student_hidden, teacher_hidden)
            hidden_loss += tmp_loss
        return attn_loss/N, hidden_loss/N

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    if config.DISTILL.DO_DISTILL:
        logger.info(f"Loading teacher model:{config.MODEL.TYPE}/{config.DISTILL.TEACHER}")
        model_teacher = load_teacher_model()
        model_teacher.cuda()
        model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        checkpoint = torch.load(config.DISTILL.TEACHER, map_location='cpu')
        msg = model_teacher.module.load_state_dict(checkpoint['model'], strict=False)
        logger.info(msg)
        del checkpoint
        torch.cuda.empty_cache()

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.DISTILL.TRAIN_INTERMEDIATE:
        lr_scheduler = None
        criterion_soft = cal_relation_loss
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
        criterion_soft = soft_cross_entropy

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion_truth = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion_truth = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion_truth = torch.nn.CrossEntropyLoss()
    
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
        if config.DISTILL.TRAIN_INTERMEDIATE and config.DISTILL.PROGRESSIVE:
            max_accuracy = load_checkpoint(config, model_without_ddp, None, None, logger)
            validate(config, data_loader_val, model, logger, is_intermediate=True, model_teacher=model_teacher)
        elif config.DISTILL.TRAIN_INTERMEDIATE:
            max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
            validate(config, data_loader_val, model, logger, is_intermediate=True, model_teacher=model_teacher)
        else:
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
        if config.DISTILL.DO_DISTILL and config.DISTILL.TRAIN_INTERMEDIATE:
            train_one_epoch_intermediate(config, model, model_teacher, criterion_soft, data_loader_train, optimizer, epoch, mixup_fn)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        elif config.DISTILL.DO_DISTILL:
            train_one_epoch_distill(config, model, model_teacher, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, criterion_soft=criterion_soft, criterion_truth=criterion_truth)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

            acc1, acc5, loss = validate(config, data_loader_val, model, logger)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        else:
            train_one_epoch(config, model, criterion_truth, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
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
    model = SwinTransformerRelation(img_size=224,
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

def train_one_epoch_intermediate(config, model, model_teacher, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler=None):
    layer_id_s_list = [11]
    layer_id_t_list = [23]

    model.train()
    optimizer.zero_grad()

    model_teacher.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    #attn_loss_meter = AverageMeter()
    #hidden_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        qkv_s = model(samples, layer_id_s_list)

        with torch.no_grad():
            qkv_t = model_teacher(samples, layer_id_t_list)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(qkv_s, qkv_t)

            loss = loss / config.TRAIN.ACCUMULATION_STEPS
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
                if lr_scheduler is not None:
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(qkv_s, qkv_t)
            
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
            if lr_scheduler is not None:
                lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
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
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")



def train_one_epoch_intermediate_progressive(config, model, model_teacher, criterion, data_loader, optimizer, epoch, mixup_fn, layer_stage, lr_scheduler=None):
    #total_epoch = config.TRAIN.EPOCHS
    #layer_stage = epoch // 25 ## 25 epochs for each stage
    #layer_stage = epoch // 5
    if epoch == 0:
        logger.info("Training stage: %d..."%layer_stage)
    #hidden_loss_weight = [10, 10, 10, 1.0]
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

    for idx, (samples, targets) in enumerate(data_loader):
        '''
        if idx > 10:
            break
        '''
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        attn_outputs, hidden_outputs = model(samples, layer_stage)

        with torch.no_grad():
            attn_outputs_teacher, hidden_outputs_teacher = model_teacher(samples, layer_stage)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            attn_loss, hidden_loss = cal_intermediate_loss(attn_outputs, hidden_outputs, attn_outputs_teacher, hidden_outputs_teacher)
            loss = attn_loss + hidden_loss

            loss = loss / config.TRAIN.ACCUMULATION_STEPS
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
                if lr_scheduler is not None:
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            attn_loss, hidden_loss = cal_intermediate_loss(attn_outputs, hidden_outputs, attn_outputs_teacher, hidden_outputs_teacher)
            loss = attn_loss + hidden_loss
            
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


def train_one_epoch_distill(config, model, model_teacher, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, criterion_soft=None, criterion_truth=None):
    model.train()
    optimizer.zero_grad()

    model_teacher.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_soft_meter = AverageMeter()
    loss_truth_meter = AverageMeter()

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
            loss_truth = config.DISTILL.ALPHA*criterion_truth(outputs, targets)
            loss_soft = (1.0 - config.DISTILL.ALPHA)*criterion_soft(outputs/config.DISTILL.TEMPERATURE,
                            outputs_teacher/config.DISTILL.TEMPERATURE)
            loss = loss_truth + loss_soft

            loss = loss / config.TRAIN.ACCUMULATION_STEPS
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
            loss_truth = config.DISTILL.ALPHA*criterion_truth(outputs, targets)
            loss_soft = (1.0 - config.DISTILL.ALPHA)*criterion_soft(outputs/config.DISTILL.TEMPERATURE,
                            outputs_teacher/config.DISTILL.TEMPERATURE)
            loss = loss_truth + loss_soft

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
        loss_soft_meter.update(loss_soft.item(), targets.size(0))
        loss_truth_meter.update(loss_truth.item(), targets.size(0))
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
                f'loss_soft {loss_soft_meter.val:.4f} ({loss_soft_meter.avg:.4f})\t'
                f'loss_truth {loss_truth_meter.val:.4f} ({loss_truth_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
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
            loss = criterion(outputs, targets)
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
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")



@torch.no_grad()
def validate(config, data_loader, model, logger, is_intermediate=False, model_teacher=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    loss_attn_list = [AverageMeter() for _ in range(4)]
    loss_hidden_list = [AverageMeter() for _ in range(4)]
    loss_attn_meter = AverageMeter()
    loss_hidden_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        if is_intermediate:
            model_teacher.eval()
            loss_attn = 0.0
            loss_hidden = 0.0
            layer_num = [2, 2, 6, 2]
            for layer_stage in range(4):
                student_attn_list, student_hidden_list = model(images, layer_stage)
                teacher_attn_list, teacher_hidden_list = model_teacher(images, layer_stage)
                attn_loss, hidden_loss = cal_intermediate_loss(student_attn_list, student_hidden_list, teacher_attn_list, teacher_hidden_list)
                loss_attn_list[layer_stage].update(attn_loss.item(), target.size(0))
                loss_hidden_list[layer_stage].update(hidden_loss.item(), target.size(0))
                loss_attn += attn_loss*layer_num[layer_stage]
                loss_hidden += hidden_loss*layer_num[layer_stage]
            loss_attn /= sum(layer_num)
            loss_hidden /= sum(layer_num)
            loss = loss_attn + loss_hidden
            loss_attn_meter.update(loss_attn.item(), target.size(0))
            loss_hidden_meter.update(loss_hidden.item(), target.size(0))
            loss_meter.update(loss.item(), target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Attention Loss {loss_attn_meter.val:.4f} ({loss_attn_meter.avg:.4f})\t'
                    f'Hidden Loss {loss_hidden_meter.val:.4f} ({loss_hidden_meter.avg:.4f})\t'
                    f'Attention_Loss_0 {loss_attn_list[0].val:.4f} ({loss_attn_list[0].avg:.4f})\t'
                    f'Attention_Loss_1 {loss_attn_list[1].val:.4f} ({loss_attn_list[1].avg:.4f})\t'
                    f'Attention_Loss_2 {loss_attn_list[2].val:.4f} ({loss_attn_list[2].avg:.4f})\t'
                    f'Attention_Loss_3 {loss_attn_list[3].val:.4f} ({loss_attn_list[3].avg:.4f})\t'
                    f'Hidden_Loss_0 {loss_hidden_list[0].val:.4f} ({loss_hidden_list[0].avg:.4f})\t'
                    f'Hidden_Loss_1 {loss_hidden_list[1].val:.4f} ({loss_hidden_list[1].avg:.4f})\t'
                    f'Hidden_Loss_2 {loss_hidden_list[2].val:.4f} ({loss_hidden_list[2].avg:.4f})\t'
                    f'Hidden_Loss_3 {loss_hidden_list[3].val:.4f} ({loss_hidden_list[3].avg:.4f})\t'
                    f'Mem {memory_used:.0f}MB')
        else:
            output = model(images)

            # measure accuracy and record loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
    if is_intermediate:
        logger.info(f' * Loss {loss_meter.avg:.3f}  Attention Loss {loss_attn_meter.avg:.3f}  Hidden Loss {loss_hidden_meter.avg:.3f}  Attention_Loss_0 {loss_attn_list[0].avg:.3f} Attention_Loss_1 {loss_attn_list[1].avg:.3f} Attention_Loss_2 {loss_attn_list[2].avg:.3f} Attention_Loss_3 {loss_attn_list[3].avg:.3f} Hidden_Loss_0 {loss_hidden_list[0].avg:.3f} Hidden_Loss_1 {loss_hidden_list[1].avg:.3f} Hidden_Loss_2 {loss_hidden_list[2].avg:.3f} Hidden_Loss_3 {loss_hidden_list[3].avg:.3f}')
        return
    else:
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    ## eval:
    # python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --eval --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /sdb/imagenet --teacher ~/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 192 --tag dist_intermediate_eval --intermediate_checkpoint ~/trained_models/swin_tiny_intermediate.pth
 

    ## train:
    #### original training
    #python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --cfg configs/swin_tiny_patch4_window7_224_distill.yaml --data-path /root/FastBaseline/data/imagenet --batch-size 128 --tag baseline --output /mnt/configblob/users/v-jinnian/swin_distill

    #### distillation with pred loss
    # python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill.yaml --data-path /root/FastBaseline/data/imagenet --teacher /mnt/configblob/users/v-jinnian/swin_distill/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag dist_org --output /mnt/configblob/users/v-jinnian/swin_distill
    # python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill.yaml --data-path /sdb/imagenet --teacher ~/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag dist_alpha --output /mnt/configblob/users/v-jinnian/swin_distill

    #### distillation with intermediate loss
    # CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /root/FastBaseline/data/imagenet --teacher /mnt/configblob/users/v-jinnian/swin_distill/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_inter_all --train_intermediate --stage -1
    # python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /root/FastBaseline/data/imagenet --teacher /mnt/configblob/users/v-jinnian/swin_distill/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_inter_prog_$i --resume output/test_inter_prog/test_inter_prog_$i/ckpt_epoch_1.pth --train_intermediate --stage $i --output output/test_inter_prog
    #### distillation with relation loss
    #python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234  main.py --do_distill --cfg configs/swin_tiny_patch4_window7_224_distill_intermediate.yaml --data-path /root/FastBaseline/data/imagenet --teacher /mnt/configblob/users/v-jinnian/swin_distill/trained_models/swin_large_patch4_window7_224_22kto1k.pth --batch-size 128 --tag test_inter_all --train_intermediate --ar 48
    

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
