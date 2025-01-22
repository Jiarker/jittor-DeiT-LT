# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import datetime
import math

import jittor as jt

# import torch
# import torch.distributed as dist
# from torch import inf
from collections import defaultdict, deque


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window or the global series average."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = jt.array(list(self.deque))
        return jt.median(d).item()

    @property
    def avg(self):
        d = jt.array(list(self.deque), dtype=jt.float32)
        return jt.mean(d).item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, jt.Var):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            # meter.synchronize_between_processes()  # Jittor does not support distributed training
            pass

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        len_iter = math.ceil(len(iterable) / iterable.batch_size)
        space_fmt = ":" + str(len(str(len_iter))) + "d"
        log_msg = [
            header,
            "[{i}/{len_iter}]",
            # "[{i}/{len_iter}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        # if jt.has_cuda:
        #     log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iter - 1:
            # if True:
                eta_seconds = iter_time.global_avg * (len_iter - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # if jt.has_cuda:
                #     print(
                #         log_msg.format(
                #             i,
                #             len(iterable),
                #             eta=eta_string,
                #             meters=str(self),
                #             time=str(iter_time),
                #             data=str(data_time),
                #             # memory=jt.cuda.max_memory_allocated() / MB,
                #         )
                #     )
                # else:
                print(
                    log_msg.format(
                        i=i,
                        len_iter=len_iter,
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                    )
                )

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len_iter
            )
        )

# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate with half-cycle cosine after warmup"""
#     if epoch < args.warmup_epochs:
#         lr = args.lr * epoch / args.warmup_epochs
#     else:
#         lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
#             1.0
#             + math.cos(
#                 math.pi
#                 * (epoch - args.warmup_epochs)
#                 / (args.epochs - args.warmup_epochs)
#             )
#         )
#     for param_group in optimizer.param_groups:
#         if "lr_scale" in param_group:
#             param_group["lr"] = lr * param_group["lr_scale"]
#         else:
#             param_group["lr"] = lr
#     return lr