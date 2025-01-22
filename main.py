"""
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/Jittor-Image-Models/Jittor-Image-Models
"""

import argparse
import numpy as np
from tqdm import tqdm
import jittor as jt
from jittor import nn
from jittor import transform

# from jvcore.utils.model_ema import ModelEma, _load_checkpoint_for_ema, get_state_dict
import logging
import time
import wandb
import json
import datetime

from pathlib import Path
import os

from jvcore.models import create_model, load_checkpoint
from jvcore.data import create_dataset, build_dataset
from jvcore.scheduler import create_cos_scheduler

from utils.arguments import get_args_parser
from utils.samplers import RASampler
from utils.mix import Mixup, Mixup_transmix
# from utils.native import NativeScalerWithGradNormCount, NativeScaler
from utils.losses import SoftTargetCrossEntropy, DistillationLossMultiCrop, DistillationLoss
from utils.engine import evaluate, train_one_epoch

import moco.loader
import moco.builder
from moco import resnet_cifar_paco

jt.flags.use_cuda = 1
    
def main(args):
    if not args.eval:
        args.no_distillation = False
    if "distilled" not in args.model:
        args.no_distillation = True
        print("\nNO DISTILLATION\n")
        time.sleep(2)    

    name = args.name_exp
    if args.log_results:
        wandb.init(project=args.project_name, name=name)
        wandb.run.log_code(".")
        wandb.config.update(args)

    # TODO 设置数据集
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # TODO 数据集采样形成长尾分布
    cls_num_list = dataset_train.get_cls_num_list()
    args.cls_num_list = cls_num_list

    beta = args.beta
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = jt.float32(per_cls_weights)

    args.categories = []
    if args.data_set == "CIFAR10LT":
        args.categories = [3, 7]
    if args.data_set == "CIFAR100LT":
        args.categories = [36, 71]

    sampler_train = RASampler(dataset_train, shuffle=True)
    # sampler_train = jt.dataset.RandomSampler(dataset_train, num_samples=len(dataset_train)*3)
    sampler_val = jt.dataset.SequentialSampler(dataset_val)

    # TODO 设置DataLoader
    data_loader_train = jt.dataset.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    data_loader_val = jt.dataset.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )    

    # TODO 混合
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None

    if mixup_active:
        if not args.transmix:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nb_classes,
            )
        else:
            print("USING TRANSMIX\n")
            mixup_fn = Mixup_transmix(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nb_classes,
            )

    # TODO 加载模型
    print("[INFORMATION] THe model being used is ", args.model)
    print("[INFORMATION] Model loaded from custom file")
    model = create_model(
        args.model, 
        pretrained=False, 
        # checkpoint_path = 'pretrained/student_pretrained/deit_best_checkpoint.pth',
        checkpoint_path=args.student_path,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path)
    
    # print("Model", model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", n_parameters)

    # model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device="cpu" if args.model_ema_force_cpu else "",
    #         resume="",
    #     )

    # TODO 设置teacher_model
    teacher_model = moco.builder.MoCo(
    getattr(resnet_cifar_paco, args.teacher_model),
    args.moco_dim,
    args.moco_k,
    args.moco_m,
    args.moco_t,
    args.mlp,
    args.feat_dim,
    args.normalize,
    num_classes=args.nb_classes,
    )

    # teacher_checkpoint_path = 'pretrained/teacher_pretrained/cf100_100_teacher.pth'
    load_checkpoint(teacher_model, args.teacher_path)
    teacher_model.eval()

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size / 512.0
        args.lr = linear_scaled_lr

    # TODO 设置优化器
    optimizer = nn.AdamW(model.parameters(), args.lr)
    # if args.accum_iter > 1:
    #     loss_scaler = NativeScalerWithGradNormCount()
    # else:
    #     loss_scaler = NativeScaler()

    # TODO 设置lr_scheduler
    # lr_scheduler = myCosineAnnealingLR(
    #     optimizer, 
    #     warmup_t=args.warmup_epochs,
    #     warmup_lr_init=args.warmup_lr,
    #     T_max=args.epochs,
    #     eta_min=args.min_lr)
    # lr_scheduler.step(epoch=0, is_init=True)
    lr_scheduler, _ = create_cos_scheduler(args, optimizer)
    print("WARMUP EPOCHS = ", args.warmup_epochs)

    # TODO 设置criterion
    if mixup_active:
        print("Critera: SoftTargetCrossEntropy")
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = jt.nn.cross_entropy_loss()    
    if args.bce_loss:
        print("Criteria: BCE Loss")
        criterion = jt.nn.BCEWithLogitsLoss()

    # TODO 设置损失函数
    if not args.no_distillation:
        print("Criteria: Distillation Loss")
        if args.drw == None:
            weighted_distillation = (args.weighted_distillation,)
        else:
            weighted_distillation = False

        # Modified distillation loss for multi-crop
        if args.multi_crop:
            print("Multi-crop Distillation")
            distillation_loss = DistillationLossMultiCrop
        else:
            print("Normal Distillation")
            distillation_loss = DistillationLoss

        criterion = distillation_loss(
            criterion,
            teacher_model,
            args.distillation_type,
            args.distillation_alpha,
            args.distillation_tau,
            args.input_size,
            args.teacher_size,
            weighted_distillation,
            per_cls_weights,
            args,
        )            

    output_dir = Path(args.output_dir)
    # TODO 加载以前的参数
    if args.resume:
        print("RESUMING FROM CHECKPOINT")
        checkpoint = jt.load(args.resume)
        print("CHECKPOINT LOADED")

        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])    # TODO lr_scheduler加载以前的参数
            args.start_epoch = checkpoint["epoch"] + 1
            # if args.model_ema:
            #     _load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
        lr_scheduler.step(args.start_epoch)

        if args.early_stopping:
            print("Early Stopping Stage")
            model.module.head_dist.weight.requires_grad = False
            model.module.head_dist.bias.requires_grad = False
            args.distillation_alpha = 0

    # TODO 模型测试
    if args.eval:
        print("EVALUATION OF MODEL")
        test_stats = evaluate(data_loader_val, model, args)
        return  
    
    # TODO 模型训练
    start_time = time.time()
    max_accuracy_avg = 0.0
    max_head_avg = 0.0
    max_med_avg = 0.0
    max_tail_avg = 0.0
    max_accuracy_cls = 0.0
    max_head_cls = 0.0
    max_med_cls = 0.0
    max_tail_cls = 0.0
    max_accuracy_dist = 0.0
    max_head_dist = 0.0
    max_med_dist = 0.0
    max_tail_dist = 0.0
    start_epoch = args.start_epoch
    epochs = args.epochs
    print(f"Start training for {epochs} epochs")

    for epoch in range(start_epoch, epochs):

        # Code modified for DeiT-LT
        data_loader_train.sampler.set_epoch(epoch)

        if (
            args.drw is not None and epoch >= args.drw
        ):  # Do reweighting after specified number of epochs
            if not args.no_distillation:
                if args.weighted_baseloss:
                    print("USING Reweighted CE Class Loss in DRW")

                    if args.bce_loss:
                        base_criterion = nn.BCEWithLogitsLoss(
                            weight=per_cls_weights
                        )
                    else:
                        if args.no_mixup_drw:
                            base_criterion = nn.CrossEntropyLoss(
                                weight=per_cls_weights
                            )
                        else:
                            base_criterion = SoftTargetCrossEntropy()

                    criterion = DistillationLoss(
                        base_criterion,
                        teacher_model,
                        args.distillation_type,
                        args.distillation_alpha,
                        args.distillation_tau,
                        args.input_size,
                        args.teacher_size,
                        args.weighted_distillation,
                        per_cls_weights,
                        args,
                    )
                else:
                    print("USING CE Class Loss in DRW")

                    if args.bce_loss:
                        base_criterion = nn.BCEWithLogitsLoss()
                    else:
                        if args.no_mixup_drw:
                            base_criterion = nn.CrossEntropyLoss()
                        else:
                            base_criterion = SoftTargetCrossEntropy()

                    criterion = DistillationLoss(
                        base_criterion,
                        teacher_model,
                        args.distillation_type,
                        args.distillation_alpha,
                        args.distillation_tau,
                        args.input_size,
                        args.teacher_size,
                        args.weighted_distillation,
                        per_cls_weights,
                        args,
                    )
            else:
                print("Using CE Loss in DRW")
                criterion = nn.CrossEntropyLoss(weight=per_cls_weights)

        print("The distillation type is ", str(args.distillation_type))

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            teacher_model=teacher_model,
            optimizer=optimizer,
            epoch=epoch,
            max_norm=args.clip_grad,
            # model_ema=model_ema,
            mixup_fn=mixup_fn,
            data_loader=data_loader_train,
            set_training_mode=args.finetune
            == "",  # keep in eval mode during finetuning
            lr_scheduler=lr_scheduler,
            args=args,
        )

        if args.log_results:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "cls_loss": train_stats["cls_loss"],
                    "dst_loss": train_stats["dst_loss"],
                    "lr": optimizer.lr,
                    # "head_acc_avg": test_stats["head_acc_avg"],
                    # "med_acc_avg": test_stats["med_acc_avg"],
                    # "tail_acc_avg": test_stats["tail_acc_avg"],
                    # "head_acc_cls": test_stats["head_acc_cls"],
                    # "med_acc_cls": test_stats["med_acc_cls"],
                    # "tail_acc_cls": test_stats["tail_acc_cls"],
                    # "head_acc_dist": test_stats["head_acc_dist"],
                    # "med_acc_dist": test_stats["med_acc_dist"],
                    # "tail_acc_dist": test_stats["tail_acc_dist"],
                    # "best_acc_avg": max_accuracy_avg,
                    # "best_head_avg": max_head_avg,
                    # "best_med_avg": max_med_avg,
                    # "best_tail_avg": max_tail_avg,
                    "sim_12: ": train_stats["sim_12"],
                }
            )

    #     print("Loss        = ", train_stats["loss"])
    #     print("Cls Loss    = ", train_stats["cls_loss"])
    #     print("Dst Loss    = ", train_stats["dst_loss"])

        if args.accum_iter == 1:
            lr_scheduler.step(epoch)

        if epoch % args.eval_freq != 0 and epoch + 1 != epochs:
            continue

        test_stats = evaluate(data_loader_val, model, args)
        checkpoint_paths = []
        if args.output_dir:
            if args.drw != None and epoch == args.drw - 1:
                checkpoint_paths.append(
                    output_dir / (name + f"_epoch_{str(epoch)}_DRW_checkpoint.pth")
                )

            if (epoch + 1) % args.save_freq == 0:
                checkpoint_paths.append(
                    output_dir / (name + f"_epoch_{str(epoch)}_checkpoint.pth")
                )

            checkpoint_paths.append(output_dir / (name + "_checkpoint.pth"))
            print(checkpoint_paths)
            for checkpoint_path in checkpoint_paths:
                checkpoint_path = str(checkpoint_path)
                print("Saving at ", checkpoint_path)
                jt.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        # "model_ema": get_state_dict(model_ema),
                        "args": args,
                        "head_acc_avg": test_stats["head_acc_avg"],
                        "med_acc_avg": test_stats["med_acc_avg"],
                        "tail_acc_avg": test_stats["tail_acc_avg"],
                        "head_acc_cls": test_stats["head_acc_cls"],
                        "med_acc_cls": test_stats["med_acc_cls"],
                        "tail_acc_cls": test_stats["tail_acc_cls"],
                        "head_acc_dist": test_stats["head_acc_dist"],
                        "med_acc_dist": test_stats["med_acc_dist"],
                        "tail_acc_dist": test_stats["tail_acc_dist"],
                    },
                    checkpoint_path,
                )

        if max_accuracy_avg < test_stats["acc1_avg"]:
            max_accuracy_avg = test_stats["acc1_avg"]
            max_head_avg = test_stats["head_acc_avg"]
            max_med_avg = test_stats["med_acc_avg"]
            max_tail_avg = test_stats["tail_acc_avg"]

            max_accuracy_cls = test_stats["acc1_cls"]
            max_head_cls = test_stats["head_acc_cls"]
            max_med_cls = test_stats["med_acc_cls"]
            max_tail_cls = test_stats["tail_acc_cls"]

            max_accuracy_dist = test_stats["acc1_dist"]
            max_head_dist = test_stats["head_acc_dist"]
            max_med_dist = test_stats["med_acc_dist"]
            max_tail_dist = test_stats["tail_acc_dist"]

            if args.output_dir:
                checkpoint_paths = [output_dir / (name + "_best_checkpoint.pth")]
                for checkpoint_path in checkpoint_paths:
                    checkpoint_path = str(checkpoint_path)
                    jt.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            # "model_ema": get_state_dict(model_ema),
                            "best_acc_avg": max_accuracy_avg,
                            "head_acc_avg": max_head_avg,
                            "med_acc_avg": max_med_avg,
                            "tail_acc_avg": max_tail_avg,
                            "best_acc_cls": max_accuracy_cls,
                            "head_acc_cls": max_head_cls,
                            "med_acc_cls": max_med_cls,
                            "tail_acc_cls": max_tail_cls,
                            "best_acc_dist": max_accuracy_dist,
                            "head_acc_dist": max_head_dist,
                            "med_acc_dist": max_med_dist,
                            "tail_acc_dist": max_tail_dist,
                        },
                        checkpoint_path,
                    )

        print("\nBEST NUMBERS ----->")
        print("Overall / Head / Med / Tail")
        print(
            "AVERAGE: ",
            round(max_accuracy_avg, 3),
            " / ",
            round(max_head_avg, 3),
            " / ",
            round(max_med_avg, 3),
            " / ",
            round(max_tail_avg, 3),
        )
        print(
            "CLS    : ",
            round(max_accuracy_cls, 3),
            " / ",
            round(max_head_cls, 3),
            " / ",
            round(max_med_cls, 3),
            " / ",
            round(max_tail_cls, 3),
        )
        print(
            "DIST   : ",
            round(max_accuracy_dist, 3),
            " / ",
            round(max_head_dist, 3),
            " / ",
            round(max_med_dist, 3),
            " / ",
            round(max_tail_dist, 3),
        )
        print("\n\n")

        if args.log_results:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            wandb.log(
                {
                    # "epoch": epoch,
                    # "train_loss": train_stats["loss"],
                    # "cls_loss": train_stats["cls_loss"],
                    # "dst_loss": train_stats["dst_loss"],
                    # "lr": optimizer.lr,
                    "head_acc_avg": test_stats["head_acc_avg"],
                    "med_acc_avg": test_stats["med_acc_avg"],
                    "tail_acc_avg": test_stats["tail_acc_avg"],
                    "head_acc_cls": test_stats["head_acc_cls"],
                    "med_acc_cls": test_stats["med_acc_cls"],
                    "tail_acc_cls": test_stats["tail_acc_cls"],
                    "head_acc_dist": test_stats["head_acc_dist"],
                    "med_acc_dist": test_stats["med_acc_dist"],
                    "tail_acc_dist": test_stats["tail_acc_dist"],
                    "best_acc_avg": max_accuracy_avg,
                    "best_head_avg": max_head_avg,
                    "best_med_avg": max_med_avg,
                    "best_tail_avg": max_tail_avg,
                    # "sim_12: ": train_stats["sim_12"],
                }
            )

            if args.output_dir:
                with (output_dir / (name + "_log.txt")).open("a") as f:
                    # f.write(json.dumps(log_stats) + "\n")
                    for key, value in log_stats.items():
                        f.write(f"{key}: {value}\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))





if __name__ == '__main__':
    # TODO 参数加载
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    args.name_exp = (
        args.model
        + "_"
        + args.teacher_model
        + "_"
        + str(args.epochs)
        + "_"
        + args.data_set
        + "_"
        + "imb"
        + str(int(1 / args.imb_factor))
        + "_"
        + str(args.batch_size)
        + "_"
        + args.experiment
    )
    if args.output_dir:
        Path(os.path.join(Path(args.output_dir), str(args.name_exp))).mkdir(
            parents=True, exist_ok=True
        )
    args.output_dir = Path(os.path.join(Path(args.output_dir), str(args.name_exp)))    

    main(args)
