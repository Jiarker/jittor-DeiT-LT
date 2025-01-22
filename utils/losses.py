import math
import jittor as jt
import jittor.nn as nn
import jittor.nn as F
import numpy as np

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def execute(self, x: jt.Var, target: jt.Var) -> jt.Var:
        # 计算 log_softmax
        log_softmax = jt.log(nn.softmax(x, dim=-1))
        # 计算损失
        loss = jt.sum(-target * log_softmax, dim=-1)
        # 返回平均损失
        return jt.mean(loss)
    
class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: nn.Module, teacher_model: nn.Module,
                 distillation_type: str, alpha: float, tau: float, input_size: int, teacher_size: int, weighted_distillation: bool, weight: list, args=None):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.input_size = input_size
        self.teacher_size = teacher_size
        self.weighted_distillation = weighted_distillation
        self.weight = weight
        self.args = args

    def execute(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Var, or a Tuple[Var, Var], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, jt.Var):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs[0], outputs[1]
        try:
            base_loss = self.base_criterion(outputs, labels)
        except:
            labels = F.one_hot(labels.to(jt.int64), num_classes=len(self.weight)).cuda()
            base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == 'none':
            if not self.weighted_distillation:
                token_2_loss = SoftTargetCrossEntropy()(outputs_kd, labels)
            else:
                token_2_loss = nn.CrossEntropyLoss(weight=self.weight)(outputs_kd, labels)
            loss = base_loss * (1 - self.alpha) + token_2_loss * self.alpha
            return loss, base_loss * (1 - self.alpha), token_2_loss * self.alpha

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is expected to return a Tuple[Var, Var] with the output of the class_token and the dist_token")

        # don't backprop through the teacher
        with jt.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            pred_t = F.log_softmax(teacher_outputs / T, dim=1)
            if self.weighted_distillation:
                pred_t = pred_t * self.weight
                pred_t = pred_t / pred_t.sum(1)[:, None]

            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                pred_t,
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            # distillation_targets = teacher_outputs.argmax(dim=1)
            distillation_targets, _ = teacher_outputs.argmax(dim=1)
            if self.args.map_targets:
                distillation_targets = jt.array(np.array(self.args.class_map)[distillation_targets.detach().cpu()]).type(jt.LongTensor).cuda()

            if self.weighted_distillation:
                distillation_loss = nn.cross_entropy_loss(outputs_kd, distillation_targets, weight=self.weight)
            else:
                distillation_loss = nn.cross_entropy_loss(outputs_kd, distillation_targets)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss, base_loss * (1 - self.alpha), distillation_loss * self.alpha
    
class DistillationLossMultiCrop(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self,
                 base_criterion: nn.Module,
                 teacher_model: nn.Module,
                 distillation_type: str,
                 alpha: float,
                 tau: float,
                 input_size: int,
                 teacher_size: int,
                 weighted_distillation: bool,
                 weight: list,
                 args=None):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.input_size = input_size
        self.teacher_size = teacher_size
        self.weighted_distillation = weighted_distillation
        self.weight = weight
        self.args = args

    def execute(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Var, or a Tuple[Var, Var], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        outputs_kd = None
        if not isinstance(outputs, jt.Var):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd, _, _ = outputs
        try:
            base_loss = self.base_criterion(outputs, labels)
        except:
            labels = F.one_hot(labels.to(jt.int64), num_classes=len(self.weight)).cuda()
            base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == "none":
            if not self.weighted_distillation:
                token_2_loss = SoftTargetCrossEntropy()(outputs_kd, labels)
            else:
                token_2_loss = nn.CrossEntropyLoss(weight=self.weight)(outputs_kd, labels)
            loss = base_loss * (1 - self.alpha) + token_2_loss * self.alpha
            return loss, base_loss * (1 - self.alpha), token_2_loss * self.alpha

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is expected to return a Tuple[Var, Var] with the output of the class_token and the dist_token")

        # don't backprop through the teacher

        with jt.no_grad():
            if isinstance(inputs, list):
                teacher_out_global = self.teacher_model(inputs[0])
                teacher_out_local = self.teacher_model(inputs[1])
                teacher_outputs = jt.concat([teacher_out_global, teacher_out_local], dim=0)
            else:
                teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            T = self.tau
            pred_t = F.log_softmax(teacher_outputs / T, dim=1)
            if self.weighted_distillation:
                pred_t = pred_t * self.weight
                pred_t = pred_t / pred_t.sum(1)[:, None]

            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                pred_t,
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == "hard":
            distillation_targets = teacher_outputs.argmax(dim=1).cuda()
            if not self.args.local_global_teacher:
                outputs_kd_final = outputs_kd[: distillation_targets.size(0)]
            else:
                outputs_kd_final = outputs_kd
            if self.args.map_targets:
                distillation_targets = jt.array(np.array(self.args.class_map)[distillation_targets.detach().cpu()]).type(jt.LongTensor).cuda()
            if self.weighted_distillation:
                distillation_loss = F.cross_entropy(outputs_kd_final, distillation_targets, weight=self.weight)
            else:
                distillation_loss = F.cross_entropy(outputs_kd_final, distillation_targets)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss, base_loss * (1 - self.alpha), distillation_loss * self.alpha