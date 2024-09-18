import torch
import torch.nn as nn
import torchvision
# noinspection PyUnresolvedReferences
# from torchsummary import summary

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)
#可选择的densenet模型
__all__ = ['DenseNet121', 'DenseNet169','DenseNet201','DenseNet264']
 
'''-------------一、构造初始卷积层-----------------------------'''
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
 
'''-------------二、定义Dense Block模块-----------------------------'''
 
'''---（1）构造Dense Block内部结构---'''
#BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv
class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
           # growth_rate：增长率。一层产生多少个特征图
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)
 
    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)
 
'''---（2）构造Dense Block模块---'''
class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size , drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        # 随着layer层数的增加，每增加一层，输入的特征图就增加一倍growth_rate
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.layers(x)
 
 
'''-------------三、构造Transition层-----------------------------'''
#BN+1×1Conv+2×2AveragePooling
class _TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace,out_channels=plance,kernel_size=1,stride=1,padding=0,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )
 
    def forward(self, x):
        return self.transition_layer(x)
 
 
'''-------------四、搭建DenseNet网络-----------------------------'''
class DenseNet(Backbone):
    def __init__(self, init_channels=64, growth_rate=None, blocks = None, num_classes = None,out_features=None,):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.out_features = out_features
        self.num_classes = num_classes
        self._out_feature_channels = {'stem': 64, 'dens2': 256, 'dens3': 512, 'dens4': 1024, 'dens5': 1024}
        self._out_feature_strides = {'stem': 4, 'dens2': 4, 'dens3': 8, 'dens4': 16, 'dens5': 32}
        self.conv1 = Conv1(in_planes=3, places=init_channels)
 
        blocks*4
 
        #第一次执行特征的维度来自于前面的特征提取
        num_features = init_channels
 
        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        #num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2
 
        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2
 
        # 第3个DenseBlock有24个DenseLayer, 执行DenseBlock（24,256,32,4）
        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（1024,512）
        self.transition3 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第3回合之后，第4个DenseBlock的输入的feature应该是：num_features = 512
        num_features = num_features // 2
 
        # 第4个DenseBlock有16个DenseLayer, 执行DenseBlock（16,512,32,4）
        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate
 
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, num_classes)
 
    def forward(self, x):
        assert x.dim() == 4, f"DensNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.conv1(x)# torch.Size([1, 64, 56, 56]) 就是resnet的stem部分
        if "stem" in self.out_features:
            outputs["stem"] = x
        dens2 = self.layer1(x)# torch.Size([1, 256, 56, 56]) 
        
        dens3 = self.transition1(dens2)
        dens3 = self.layer2(dens3)# torch.Size([1, 512, 28, 28])
        
        dens4 = self.transition2(dens3)  
        dens4 = self.layer3(dens4)# torch.Size([1, 1024, 14, 14])
        
        dens5 = self.transition3(dens4)
        dens5 = self.layer4(dens5)# torch.Size([1,1024, 7, 7])
        out = self.out_features
        
        for name, des in zip(out, [dens2, dens3, dens4, dens5]):
                outputs[name] = des
                
        if self.num_classes is  None:   
             dens6 = self.avgpool(dens5)# torch.Size([1, 1024 1, 1])
            #  dens6 = x.view(dens6.size(0), -1)
             dens6 = torch.flatten(dens6, 1)
             dens6= self.fc(dens6) 
             if "linear" in self.out_features:
                 outputs["linear"] = x             
        return outputs   
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.out_features
        }
    
    
def build_densetnet_backbone(cfg, input_shape):
    """
    Create a denseNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT # 2
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES# ['res2', 'res3', 'res4', 'res5']改成densenet
    depth               = cfg.MODEL.RESNETS.DEPTH# 50
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS# 1
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP# 64
    bottleneck_channels = num_groups * width_per_group# 64
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS# 64
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS# 256
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1# False
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION# 1
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE# [False, False, False, False]
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED# False
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS# 1
    densenet_name       = cfg.MODEL.DENSENET.NAME
    num_classes         = cfg.MODEL.DENSENET.NUM_CLASSES
    out_features        = cfg.MODEL.DENSENET.OUT_FEATURES# ('dens2', 'dens3', 'dens4', 'dens5')
    in_channels         = cfg.MODEL.DENSENET.STEM_OUT_CHANNELS# 64
    # fmt: on
    # out = ('dens2', 'dens3', 'dens4', 'dens5')
    if densenet_name== 'densenet121':
        return DenseNet(init_channels=in_channels, growth_rate=32, blocks=[6, 12, 24, 16], num_classes=num_classes, out_features=out_features)
    elif densenet_name == 'densenet169':
        return DenseNet(init_channels=in_channels, growth_rate=32, blocks=[6, 12, 32, 32], num_classes=num_classes, out_features=out_features)
    elif densenet_name == 'densenet201':
        return DenseNet(init_channels=in_channels, growth_rate=32, blocks=[6, 12, 48, 32], num_classes=num_classes, out_features=out_features)
    else:
        return DenseNet(init_channels=in_channels, growth_rate=32, blocks=[6, 12, 64, 48], num_classes=num_classes, out_features=out_features)
