# import jittor as jt
# from jittor.optim import amp

# import jittor as jt

# class NativeScaler:
#     def __init__(self):
#         self.scaler = amp.GradScaler()

#     def dispatch_clip_grad(parameters, max_norm, norm_type=2):
#         """
#         Clip gradients of an iterable of parameters at specified value.
#         This is a Jittor implementation of gradient clipping similar to PyTorch's torch.nn.utils.clip_grad_norm_
        
#         Args:
#             parameters (iterable): iterable of parameters to clip gradients of
#             max_norm (float or int): max norm of the gradients
#             norm_type (float or int): type of the used norm. Default: 2
#         """
#         if isinstance(parameters, dict):
#             parameters = parameters.values()
#         parameters = [p for p in parameters if p.grad is not None]
#         max_norm = float(max_norm)
#         norm_type = float(norm_type)
#         if len(parameters) == 0:
#             return
        
#         total_norm = 0.0
#         for p in parameters:
#             param_norm = jt.norm(p.grad, p=norm_type).item()
#             total_norm += param_norm ** norm_type
        
#         total_norm = total_norm ** (1. / norm_type)
#         clip_coef = max_norm / (total_norm + 1e-6)  # 1e-6 是为了防止除以0的情况发生
#         for p in parameters:
#             p.grad = p.grad * clip_coef

#     def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, need_update=True):
#         with amp.scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward(create_graph=create_graph)
#         if need_update:
#             if clip_grad is not None:
#                 assert parameters is not None
#                 # unscale the gradients of optimizer's assigned params in-place
#                 for param in parameters:
#                     param.grad = self.scaler.unscale(param.grad)
#                 # dispatch_clip_grad is a custom function to clip gradients
#                 self.dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
#             self.scaler.step(optimizer)
#             self.scaler.update()

#     def state_dict(self):
#         return self.scaler.state_dict()

#     def load_state_dict(self, state_dict):
#         self.scaler.load_state_dict(state_dict)

# class NativeScalerWithGradNormCount:
#     def __init__(self):
#         self.scaler = jt.optim.amp.GradScaler()

#     def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
#         with jt.optim.amp.autocast():
#             self.scaler.scale(loss).backward(create_graph=create_graph)
#         if update_grad:
#             if clip_grad is not None:
#                 assert parameters is not None
#                 self.scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
#                 norm = self.clip_grad_norm(parameters, clip_grad)
#             else:
#                 self.scaler.unscale_(optimizer)
#                 norm = self.get_grad_norm(parameters)
#             self.scaler.step(optimizer)
#             self.scaler.update()
#         else:
#             norm = None
#         return norm

#     def clip_grad_norm(self, parameters, max_norm):
#         """
#         Clip gradients of an iterable of parameters at specified value.
#         This is a Jittor implementation of gradient clipping similar to PyTorch's torch.nn.utils.clip_grad_norm_
#         """
#         total_norm = 0
#         for p in parameters:
#             if p.grad is not None:
#                 param_norm = jt.norm(p.grad).item()
#                 total_norm += param_norm ** 2
#         total_norm = total_norm ** 0.5
#         clip_coef = max_norm / (total_norm + 1e-6)
#         for p in parameters:
#             if p.grad is not None:
#                 p.grad = p.grad * clip_coef

#     def get_grad_norm(self, parameters):
#         """
#         Calculate the gradient norm of an iterable of parameters.
#         """
#         total_norm = 0
#         for p in parameters:
#             if p.grad is not None:
#                 param_norm = jt.norm(p.grad).item()
#                 total_norm += param_norm ** 2
#         return total_norm ** 0.5

#     def state_dict(self):
#         return self.scaler.state_dict()

#     def load_state_dict(self, state_dict):
#         self.scaler.load_state_dict(state_dict)