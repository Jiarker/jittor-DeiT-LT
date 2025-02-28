import jittor as jt
# from jittor.lr_scheduler import CosineAnnealingLR
from jittor.optim import Optimizer

from typing import List, Optional, Union, Any, Dict
import logging
import math
import abc
from abc import ABC

class myCosineAnnealingLR(object):
    def __init__(self, optimizer, warmup_t, warmup_lr_init, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.lr
        self.base_lr_pg = [pg.get("lr") for pg in optimizer.param_groups]
        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t
        #TODO set last_epoch is not ready

    def get_lr(self, base_lr, now_lr, epoch):
        if self.last_epoch == 0:
            return base_lr
        # TODO warm_up策略
        if epoch < self.warmup_t:
            warmup_steps = (base_lr - self.warmup_lr_init) / self.warmup_t
            return (self.warmup_lr_init + epoch * warmup_steps)
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return (now_lr + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2)
        return  ((1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (now_lr - self.eta_min) + self.eta_min)

    def step(self, epoch, is_init=False):
        if not is_init:
            self.last_epoch += 1
        self.update_lr(epoch)
            
    def update_lr(self, epoch):
        self.optimizer.lr = self.get_lr(self.base_lr, self.optimizer.lr, epoch)
        for i, param_group in enumerate(self.optimizer.param_groups):
            if param_group.get("lr") != None:
                param_group["lr"] = self.get_lr(self.base_lr_pg[i], param_group["lr"], epoch)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

# _logger = logging.getLogger(__name__)

# def scheduler_kwargs(cfg, decreasing_metric: Optional[bool] = None):
#     """ cfg/argparse to kwargs helper
#     Convert scheduler args in argparse args or cfg (.dot) like object to keyword args.
#     """
#     eval_metric = getattr(cfg, 'eval_metric', 'top1')
#     if decreasing_metric is not None:
#         plateau_mode = 'min' if decreasing_metric else 'max'
#     else:
#         plateau_mode = 'min' if 'loss' in eval_metric else 'max'
#     kwargs = dict(
#         sched=cfg.sched,
#         num_epochs=getattr(cfg, 'epochs', 100),
#         decay_epochs=getattr(cfg, 'decay_epochs', 30),
#         decay_milestones=getattr(cfg, 'decay_milestones', [30, 60]),
#         warmup_epochs=getattr(cfg, 'warmup_epochs', 5),
#         cooldown_epochs=getattr(cfg, 'cooldown_epochs', 0),
#         patience_epochs=getattr(cfg, 'patience_epochs', 10),
#         decay_rate=getattr(cfg, 'decay_rate', 0.1),
#         min_lr=getattr(cfg, 'min_lr', 0.),
#         warmup_lr=getattr(cfg, 'warmup_lr', 1e-5),
#         warmup_prefix=getattr(cfg, 'warmup_prefix', False),
#         noise=getattr(cfg, 'lr_noise', None),
#         noise_pct=getattr(cfg, 'lr_noise_pct', 0.67),
#         noise_std=getattr(cfg, 'lr_noise_std', 1.),
#         noise_seed=getattr(cfg, 'seed', 42),
#         cycle_mul=getattr(cfg, 'lr_cycle_mul', 1.),
#         cycle_decay=getattr(cfg, 'lr_cycle_decay', 0.1),
#         cycle_limit=getattr(cfg, 'lr_cycle_limit', 1),
#         k_decay=getattr(cfg, 'lr_k_decay', 1.0),
#         plateau_mode=plateau_mode,
#         step_on_epochs=not getattr(cfg, 'sched_on_updates', False),
#     )
#     return kwargs

# def create_cos_scheduler(
#         args,
#         optimizer: Optimizer,
#         updates_per_epoch: int = 0,
# ):
#     return create_cos_scheduler_v2(
#         optimizer=optimizer,
#         **scheduler_kwargs(args),
#         updates_per_epoch=updates_per_epoch,
#     )

# def create_cos_scheduler_v2(
#         optimizer: Optimizer,
#         sched: str = 'cosine',
#         num_epochs: int = 300,
#         decay_epochs: int = 90,
#         decay_milestones: List[int] = (90, 180, 270),
#         cooldown_epochs: int = 0,
#         patience_epochs: int = 10,
#         decay_rate: float = 0.1,
#         min_lr: float = 0,
#         warmup_lr: float = 1e-5,
#         warmup_epochs: int = 0,
#         warmup_prefix: bool = False,
#         noise: Union[float, List[float]] = None,
#         noise_pct: float = 0.67,
#         noise_std: float = 1.,
#         noise_seed: int = 42,
#         cycle_mul: float = 1.,
#         cycle_decay: float = 0.1,
#         cycle_limit: int = 1,
#         k_decay: float = 1.0,
#         plateau_mode: str = 'max',
#         step_on_epochs: bool = True,
#         updates_per_epoch: int = 0,
# ):
#     t_initial = num_epochs
#     warmup_t = warmup_epochs
#     decay_t = decay_epochs
#     cooldown_t = cooldown_epochs

#     if not step_on_epochs:
#         assert updates_per_epoch > 0, 'updates_per_epoch must be set to number of dataloader batches'
#         t_initial = t_initial * updates_per_epoch
#         warmup_t = warmup_t * updates_per_epoch
#         decay_t = decay_t * updates_per_epoch
#         decay_milestones = [d * updates_per_epoch for d in decay_milestones]
#         cooldown_t = cooldown_t * updates_per_epoch

#     # warmup args
#     warmup_args = dict(
#         warmup_lr_init=warmup_lr,
#         warmup_t=warmup_t,
#         warmup_prefix=warmup_prefix,
#     )

#     # setup noise args for supporting schedulers
#     if noise is not None:
#         if isinstance(noise, (list, tuple)):
#             noise_range = [n * t_initial for n in noise]
#             if len(noise_range) == 1:
#                 noise_range = noise_range[0]
#         else:
#             noise_range = noise * t_initial
#     else:
#         noise_range = None
#     noise_args = dict(
#         noise_range_t=noise_range,
#         noise_pct=noise_pct,
#         noise_std=noise_std,
#         noise_seed=noise_seed,
#     )

#     # setup cycle args for supporting schedulers
#     cycle_args = dict(
#         cycle_mul=cycle_mul,
#         cycle_decay=cycle_decay,
#         cycle_limit=cycle_limit,
#     )

#     lr_scheduler = None
#     # if sched == 'cosine':
#     lr_scheduler = CosineLRScheduler(
#         optimizer,
#         t_initial=t_initial,
#         lr_min=min_lr,
#         t_in_epochs=step_on_epochs,
#         **cycle_args,
#         **warmup_args,
#         **noise_args,
#         k_decay=k_decay,
#         initialize=True,
#     )

#     if hasattr(lr_scheduler, 'get_cycle_length'):
#         # For cycle based schedulers (cosine, tanh, poly) recalculate total epochs w/ cycles & cooldown
#         # NOTE: Warmup prefix added in get_cycle_lengths() if enabled
#         t_with_cycles_and_cooldown = lr_scheduler.get_cycle_length() + cooldown_t
#         if step_on_epochs:
#             num_epochs = t_with_cycles_and_cooldown
#         else:
#             num_epochs = t_with_cycles_and_cooldown // updates_per_epoch
#     else:
#         if warmup_prefix:
#             num_epochs += warmup_epochs

#     return lr_scheduler, num_epochs

# class Scheduler(ABC):
#     """ Parameter Scheduler Base Class
#     A scheduler base class that can be used to schedule any optimizer parameter groups.

#     Unlike the builtin PyTorch schedulers, this is intended to be consistently called
#     * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
#     * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

#     The schedulers built on this should try to remain as stateless as possible (for simplicity).

#     This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
#     and -1 values for special behaviour. All epoch and update counts must be tracked in the training
#     code and explicitly passed in to the schedulers on the corresponding step or step_update call.

#     Based on ideas from:
#      * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
#      * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
#     """

#     def __init__(
#             self,
#             optimizer: Optimizer,
#             param_group_field: str,
#             t_in_epochs: bool = True,
#             noise_range_t=None,
#             noise_type='normal',
#             noise_pct=0.67,
#             noise_std=1.0,
#             noise_seed=None,
#             initialize: bool = True,
#     ) -> None:
#         self.optimizer = optimizer
#         self.param_group_field = param_group_field
#         self._initial_param_group_field = f"initial_{param_group_field}"
#         # if initialize:
#         #     for i, group in enumerate(self.optimizer.param_groups):
#         #         if param_group_field not in group:
#         #             raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
#         #         group.setdefault(self._initial_param_group_field, group[param_group_field])
#         # else:
#         #     for i, group in enumerate(self.optimizer.param_groups):
#         #         if self._initial_param_group_field not in group:
#         #             raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")

#         for i, group in enumerate(self.optimizer.param_groups):
#             # if param_group_field not in group:
#             #     raise KeyError(f"{param_group_field} missing from optimizer")
#             group.setdefault(self._initial_param_group_field, group[param_group_field])

#         self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
#         self.metric = None  # any point to having this for all?
#         self.t_in_epochs = t_in_epochs
#         self.noise_range_t = noise_range_t
#         self.noise_pct = noise_pct
#         self.noise_type = noise_type
#         self.noise_std = noise_std
#         self.noise_seed = noise_seed if noise_seed is not None else 42
#         self.update_groups(self.base_values)

#     def state_dict(self) -> Dict[str, Any]:
#         return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

#     def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#         self.__dict__.update(state_dict)

#     @abc.abstractmethod
#     def _get_lr(self, t: int) -> List[float]:
#         pass

#     def _get_values(self, t: int, on_epoch: bool = True) -> Optional[List[float]]:
#         proceed = (on_epoch and self.t_in_epochs) or (not on_epoch and not self.t_in_epochs)
#         if not proceed:
#             return None
#         return self._get_lr(t)

#     def step(self, epoch: int, metric: float = None) -> None:
#         self.metric = metric
#         values = self._get_values(epoch, on_epoch=True)
#         if values is not None:
#             values = self._add_noise(values, epoch)
#             self.update_groups(values)

#     def step_update(self, num_updates: int, metric: float = None):
#         self.metric = metric
#         values = self._get_values(num_updates, on_epoch=False)
#         if values is not None:
#             values = self._add_noise(values, num_updates)
#             self.update_groups(values)

#     def update_groups(self, values):
#         if not isinstance(values, (list, tuple)):
#             values = [values] * len(self.optimizer.param_groups)
#         for param_group, value in zip(self.optimizer.param_groups, values):
#             if 'lr_scale' in param_group:
#                 param_group[self.param_group_field] = value * param_group['lr_scale']
#             else:
#                 param_group[self.param_group_field] = value

#     def _add_noise(self, lrs, t):
#         if self._is_apply_noise(t):
#             noise = self._calculate_noise(t)
#             lrs = [v + v * noise for v in lrs]
#         return lrs

#     def _is_apply_noise(self, t) -> bool:
#         """Return True if scheduler in noise range."""
#         apply_noise = False
#         if self.noise_range_t is not None:
#             if isinstance(self.noise_range_t, (list, tuple)):
#                 apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
#             else:
#                 apply_noise = t >= self.noise_range_t
#         return apply_noise

#     # def _calculate_noise(self, t) -> float:
#     #     g = torch.Generator()
#     #     g.manual_seed(self.noise_seed + t)
#     #     if self.noise_type == 'normal':
#     #         while True:
#     #             # resample if noise out of percent limit, brute force but shouldn't spin much
#     #             noise = jt.randn(1, generator=g).item()
#     #             if abs(noise) < self.noise_pct:
#     #                 return noise
#     #     else:
#     #         noise = 2 * (jt.rand(1, generator=g).item() - 0.5) * self.noise_pct
#     #     return noise

#     def _calculate_noise(self, t) -> float:
#         if self.noise_type == 'normal':
#             while True:
#                 # resample if noise out of percent limit, brute force but shouldn't spin much
#                 noise = jt.randn(1).item()
#                 if abs(noise) < self.noise_pct:
#                     return noise
#         else:
#             noise = 2 * (jt.rand(1).item() - 0.5) * self.noise_pct
#         return noise

# # class CustomLR(object):
# #     def __init__(self, optimizer, gamma, last_epoch=-1):
# #         self.optimizer = optimizer
# #         self.gamma = gamma
# #         self.last_epoch = last_epoch
# #         self.base_lr = optimizer.lr
# #         self.base_lr_pg = [pg.get("lr") for pg in optimizer.param_groups]

# #     def get_lr(self, base_lr, now_lr):
# #         ## Get the current lr
# #         if self.last_epoch == 0:
# #             return base_lr
# #         return base_lr * self.gamma ** self.last_epoch


# #     def step(self, epoch):
# #         ## Update the lr, External interface function
# #         self.last_epoch = epoch
# #         self.update_lr()
     
# #     def update_lr(self):
# #         # How to update the lr
# #         self.optimizer.lr = self.get_lr(self.base_lr, self.optimizer.lr)
# #         for i, param_group in enumerate(self.optimizer.param_groups):
# #             if param_group.get("lr") != None:
# #                 param_group["lr"] = self.get_lr(self.base_lr_pg[i], param_group["lr"])

# class CosineLRScheduler(Scheduler):
#     """
#     Cosine decay with restarts.
#     This is described in the paper https://arxiv.org/abs/1608.03983.

#     Inspiration from
#     https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

#     k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
#     """

#     def __init__(
#             self,
#             optimizer: Optimizer,
#             t_initial: int,
#             lr_min: float = 0.,
#             cycle_mul: float = 1.,
#             cycle_decay: float = 1.,
#             cycle_limit: int = 1,
#             warmup_t=0,
#             warmup_lr_init=0,
#             warmup_prefix=False,
#             t_in_epochs=True,
#             noise_range_t=None,
#             noise_pct=0.67,
#             noise_std=1.0,
#             noise_seed=42,
#             k_decay=1.0,
#             initialize=True,
#     ):
#         super().__init__(
#             optimizer,
#             param_group_field="lr",
#             t_in_epochs=t_in_epochs,
#             noise_range_t=noise_range_t,
#             noise_pct=noise_pct,
#             noise_std=noise_std,
#             noise_seed=noise_seed,
#             initialize=initialize,
#         )

#         assert t_initial > 0
#         assert lr_min >= 0
#         if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
#             _logger.warning(
#                 "Cosine annealing scheduler will have no effect on the learning "
#                 "rate since t_initial = t_mul = eta_mul = 1.")
#         self.t_initial = t_initial
#         self.lr_min = lr_min
#         self.cycle_mul = cycle_mul
#         self.cycle_decay = cycle_decay
#         self.cycle_limit = cycle_limit
#         self.warmup_t = warmup_t
#         self.warmup_lr_init = warmup_lr_init
#         self.warmup_prefix = warmup_prefix
#         self.k_decay = k_decay
#         if self.warmup_t:
#             self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
#             super().update_groups(self.warmup_lr_init)
#         else:
#             self.warmup_steps = [1 for _ in self.base_values]

#     def _get_lr(self, t: int) -> List[float]:
#         if t < self.warmup_t:
#             lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
#         else:
#             if self.warmup_prefix:
#                 t = t - self.warmup_t

#             if self.cycle_mul != 1:
#                 i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
#                 t_i = self.cycle_mul ** i * self.t_initial
#                 t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
#             else:
#                 i = t // self.t_initial
#                 t_i = self.t_initial
#                 t_curr = t - (self.t_initial * i)

#             gamma = self.cycle_decay ** i
#             lr_max_values = [v * gamma for v in self.base_values]
#             k = self.k_decay

#             if i < self.cycle_limit:
#                 lrs = [
#                     self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
#                     for lr_max in lr_max_values
#                 ]
#             else:
#                 lrs = [self.lr_min for _ in self.base_values]

#         return lrs

#     def get_cycle_length(self, cycles=0):
#         cycles = max(1, cycles or self.cycle_limit)
#         if self.cycle_mul == 1.0:
#             t = self.t_initial * cycles
#         else:
#             t = int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))
#         return t + self.warmup_t if self.warmup_prefix else t
    
#     # TODO 接口函数
#     def step(self, epoch):
#         self.optimizer.lr = self._get_lr(epoch)
#         for i, param_group in enumerate(self.optimizer.param_groups):
#             if param_group.get("lr") != None:
#                 param_group["lr"] = self._get_lr(epoch)

#     def state_dict(self) -> Dict[str, Any]:
#         return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}