# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from .backbone import Backbone


BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    构建整个模型架构，由 ``cfg.MODEL.BACKBONE.NAME` 定义。
    根据配置函数里面的内容，找到对应的函数，然后调用创建模型。
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)) # ShapeSpec(channels=3, height=None, width=None, stride=None)

    backbone_name = cfg.MODEL.BACKBONE.NAME # build_resnet_fpn_backbone / build_vovnet_fpn_backbone/ build_swin_backbone
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)# 
    assert isinstance(backbone, Backbone)
    return backbone
