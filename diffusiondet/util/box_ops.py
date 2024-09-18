# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1) # unbind将张量按维度分解为元组 最后一维度分解为四个张量 x_c:tensor shapetorch.Size([1, 500])
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter 

    iou = inter / union
    return iou, union # iou, 并集


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1] # 覆盖预测边界框与真实边界框之间的最小框(的面积)

    return iou - (area - union) / area

def distance_box_iou(boxes1, boxes2):
    """Calculating DIoU from https://zhuanlan.zhihu.com/p/270663039
    requires abs xyxy
    Args:
        boxes1,boxes2(tensor:[num_preds,4]):
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    # 计算相交框的中心点，以求解中心点距离的平方 
    center1 = box_xyxy_to_cxcywh(boxes1)[:,:2] # cxcy [nums,2]
    center2 = box_xyxy_to_cxcywh(boxes2)[:,:2]
    p2 = torch.pow(center1[:,None,0] - center2[:,0],2) + torch.pow(center1[:,None,1] - center2[:,1],2)
    
    # 计算最小外接矩形对角线长度的平方 left/right top/bottom in/out
    lto = torch.min(boxes1[:, None, :2], boxes2[:, :2]) # 最小外接矩形左上角点
    rbo = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    c2 = torch.pow(rbo[:,:,0] - lto[:,:,0],2) + torch.pow(rbo[:,:,1] - lto[:,:,1],2)
    
    iou, _ = box_iou(boxes1, boxes2)
    diou = iou - p2 / c2
    
    return diou

def complete_box_iou(boxes1, boxes2):
    """Calculating CIoU 问题是宽和高不能同时增大或者减小
    requires abs xyxy
    Args:
        boxes1,boxes2(tensor:[num_preds,4]):
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, _ = box_iou(boxes1, boxes2)
    # 计算相交框的中心点，以求解中心点距离的平方 
    center1 = box_xyxy_to_cxcywh(boxes1)[:,:2] # cxcy [nums,2]
    center2 = box_xyxy_to_cxcywh(boxes2)[:,:2]
    p2 = torch.pow(center1[:,None,0] - center2[:,0],2) + torch.pow(center1[:,None,1] - center2[:,1],2) # [N, M, 2]
    
    # 计算最小外接矩形对角线长度的平方 left/right top/bottom in/out
    lto = torch.min(boxes1[:, None, :2], boxes2[:, :2]) # 最小外接矩形左上角点
    rbo = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    c2 = torch.pow(rbo[:,:,0] - lto[:,:,0],2) + torch.pow(rbo[:,:,1] - lto[:,:,1],2)
    
    wh1 = box_xyxy_to_cxcywh(boxes1)[:,2:]
    wh2 = box_xyxy_to_cxcywh(boxes2)[:,2:]
    prewh = (wh1[:,0] / wh1[:,1]).clamp(min=0) # 容易为nan of inf
    arctan = torch.atan(prewh)[:, None] - torch.atan(wh2[:,0] / wh2[:,1])
    v = (4.0 / torch.pi ** 2) * (arctan ** 2)
    iou, _ = box_iou(boxes1, boxes2)
    alpha = v / (1 - iou + v)
    
    ciou = iou - p2 / c2 - alpha * v
    
    return ciou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
