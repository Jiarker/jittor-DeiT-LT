# import jittor as jt
# import jittor.nn as nn
# import jittor.nn as F
# import jittor.init as init
# from jittor.nn import Parameter
# from .registry import register_model

# __all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


# def _weights_init(m):
#     classname = m.__class__.__name__
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight)


# class NormedLinear(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(NormedLinear, self).__init__()
#         self.weight = Parameter(jt.randn(in_features, out_features))
#         self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

#     def execute(self, x):
#         out = jt.normalize(x, dim=1) @ jt.normalize(self.weight, dim=0)
#         return out


# class LambdaLayer(nn.Module):
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd

#     def execute(self, x):
#         return self.lambd(x)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, option='A'):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         # self.shortcut = LambdaLayer(lambda x: F.pad(x, (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             # self.shortcut = LambdaLayer(lambda x: F.pad(x, (0, 0, 0, 0, planes//4, planes//4), "constant", 0))            
#             if option == 'A':
#                 """
#                 For CIFAR10 ResNet paper uses option A.
#                 """
#                 # self.shortcut = LambdaLayer(lambda x: F.pad(x, (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
#                 self.shortcut = LambdaLayer(lambda x:
#                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
#             elif option == 'B':
#                 self.shortcut = nn.Sequential(
#                      nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                      nn.BatchNorm2d(self.expansion * planes)
#                 )

#     def execute(self, x):
#         out1 = F.relu(self.bn1(self.conv1(x)))
#         out2 = self.bn2(self.conv2(out1))
#         out2 += self.shortcut(x)
#         out3 = F.relu(out2)
#         return out3


# class ResNet_s(nn.Module):

#     def __init__(self, block, num_blocks, num_classes=10, use_norm=False, return_features=False):
#         super(ResNet_s, self).__init__()
#         factor = 1 
#         self.in_planes = 16 * factor

#         self.conv1 = nn.Conv2d(3, 16 * factor, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16 * factor)
#         self.layer1 = self._make_layer(block, 16 * factor, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32 * factor, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64 * factor, num_blocks[2], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         if use_norm:
#             self.fc = NormedLinear(64 * factor, num_classes)
#         else:
#             self.fc = nn.Linear(64 * factor, num_classes)
#         self.apply(_weights_init)
#         self.return_encoding = return_features
        

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def execute(self, x):   # [128, 3, 32, 32]
#         out1 = F.relu(self.bn1(self.conv1(x))) # [128, 16, 32, 32]
#         out2 = self.layer1(out1) # [128, 16, 32, 32]  
#         out3 = self.layer2(out2)
#         out4 = self.layer3(out3)
#         out5 = self.avgpool(out4)
#         encoding = out5.view(out5.size(0), -1)
#         out6 = self.fc(encoding)
#         if self.return_encoding:
#             return out6, encoding
#         else:
#             return out6

# @register_model
# def resnet20():
#     return ResNet_s(BasicBlock, [3, 3, 3])

# @register_model
# def resnet32(pretrained=False, num_classes=10, use_norm=False, return_features=False):
#     return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm, return_features=return_features)

# @register_model
# def resnet44():
#     return ResNet_s(BasicBlock, [7, 7, 7])

# @register_model
# def resnet56():
#     return ResNet_s(BasicBlock, [9, 9, 9])

# @register_model
# def resnet110():
#     return ResNet_s(BasicBlock, [18, 18, 18])

# @register_model
# def resnet1202():
#     return ResNet_s(BasicBlock, [200, 200, 200])


# # def test(net):
# #     import numpy as np
# #     total_params = 0

# #     for x in filter(lambda p: p.requires_grad, net.parameters()):
# #         total_params += np.prod(x.data.numpy().shape)
# #     print("Total number of params", total_params)
# #     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


# # if __name__ == "__main__":
# #     for net_name in __all__:
# #         if net_name.startswith('resnet'):
# #             print(net_name)
# #             test(globals()[net_name]())
# #             print()