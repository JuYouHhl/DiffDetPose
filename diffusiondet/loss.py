# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .util import box_ops
from .util.misc import get_world_size, is_dist_avail_and_initialized
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, complete_box_iou, distance_box_iou, generalized_box_iou
import math

def angle2matrix(bz_r_pose):
    # Convert angles to radians
    bz_r_pose_rad = bz_r_pose * (math.pi / 180.0)

    # 计算欧拉角的余弦和正弦值
    cos_x = torch.cos(bz_r_pose_rad[:, 0])
    sin_x = torch.sin(bz_r_pose_rad[:, 0])
    cos_y = torch.cos(bz_r_pose_rad[:, 1])
    sin_y = torch.sin(bz_r_pose_rad[:, 1])
    cos_z = torch.cos(bz_r_pose_rad[:, 2])
    sin_z = torch.sin(bz_r_pose_rad[:, 2])

    # 构建旋转矩阵
    rotation_matrixes = torch.zeros(bz_r_pose.size(0), 3, 3, dtype=torch.float32, device=bz_r_pose.device)
    rotation_matrixes[:, 0, 0] = cos_y * cos_z
    rotation_matrixes[:, 0, 1] = - cos_x * sin_z + sin_x * sin_y * cos_z
    rotation_matrixes[:, 0, 2] = sin_x * sin_z + cos_x * sin_y * cos_z
    rotation_matrixes[:, 1, 0] = cos_y * sin_z
    rotation_matrixes[:, 1, 1] = cos_x * cos_z + sin_x * sin_y * sin_z
    rotation_matrixes[:, 1, 2] = - sin_x * cos_z + cos_x * sin_y * sin_z
    rotation_matrixes[:, 2, 0] = - sin_y
    rotation_matrixes[:, 2, 1] = sin_x * cos_y
    rotation_matrixes[:, 2, 2] = cos_x * cos_y

    return rotation_matrixes

def matrix2angle(bz_rotation_matrix):
    # bz_rotation_matrix:[numgts,3,3]
    
    factor = (180.0 / math.pi)
    sy = torch.sqrt(bz_rotation_matrix[:, 0, 0] * bz_rotation_matrix[:, 0, 0] +  bz_rotation_matrix[:, 1, 0] * bz_rotation_matrix[:, 1, 0])
    singular = sy < 1e-6
    
    rx = torch.atan2(bz_rotation_matrix[:, 2, 1] , bz_rotation_matrix[:, 2, 2])  # 
    ry = torch.atan2(-bz_rotation_matrix[:, 2, 0], sy)
    rz = torch.atan2(bz_rotation_matrix[:, 1, 0], bz_rotation_matrix[:, 0, 0])

    rx[singular] = torch.atan2(-bz_rotation_matrix[singular, 1, 2], bz_rotation_matrix[singular, 1, 1])
    ry[singular] = torch.atan2(-bz_rotation_matrix[singular, 2, 0], sy[singular])
    rz[singular] = 0

    # Convert radians to Euler angles
    angles = torch.stack((rx, ry, rz), dim=1) * factor # [numgts,3]
    # angle_list = []
    # for R in bz_rotation_matrix:
    #     sy = torch.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    #     singular = sy < 1e-6

    #     if not singular:
    #         rx = torch.atan2(R[2, 1] , R[2, 2]) # 
    #         ry = torch.atan2(-R[2, 0], sy)
    #         rz = torch.atan2(R[1, 0], R[0, 0])
    #     else:
    #         rx = torch.atan2(-R[1, 2], R[1, 1])
    #         ry = torch.atan2(-R[2, 0], sy)
    #         rz = 0
            
    #     # Convert radians to Euler angles
    #     angle = torch.tensor((rx*factor,ry*factor,rz*factor), dtype=torch.float32, device=R.device)
    #     angle_list.append(angle)
        
    return angles # torch.stack(angle_list,dim=0)

# def carla_box3d(bz_pose, bz_prelwhs):
#     src_box_center = bz_pose[:, :3] # [num_dts,3] 平移
#     src_rotation_matrix = angle2matrix(bz_pose[:, 3:]) # 旋转矩阵 [num_dts,3，3]
#     l, w, h = bz_prelwhs[:,0],bz_prelwhs[:,1],bz_prelwhs[:,2] # [num_dts]
    
#     return pre_3dboxes

def box3d(src_pose6dof, target_3dboxes_lwh):
    """ 
    Args:
        src_pose6dof (tensor[num_dts,6]):预测的6dof
        target_3dboxes_lwh (tensor[num_dts,3]): 3dbox的lwh真值
    Returns:
        pre_3dboxes (): 预测出的3dbox的三维位置
    """
    # pre_3dboxes_list = []
    src_box_center = src_pose6dof[:, :3] # [num_dts,3] 平移
    src_rotation_matrix = angle2matrix(src_pose6dof[:, 3:]) # 旋转矩阵 [num_dts,3，3]
    
    l, w, h = target_3dboxes_lwh[:,0],target_3dboxes_lwh[:,1],target_3dboxes_lwh[:,2] # [num_dts]
    x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1).to(src_rotation_matrix.device) # [num_dts,8]
    zeros=torch.zeros(l.shape,device=src_rotation_matrix.device) # [num_dts]
    y_corners = torch.stack([zeros,zeros,zeros,zeros,-h,-h,-h,-h], dim=1).to(src_rotation_matrix.device) # [num_dts,8]
    z_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1).to(src_rotation_matrix.device)
    corners = torch.stack([x_corners, y_corners, z_corners], dim=2) # [num_dts,8,3]
    pre_3dboxes = torch.matmul(corners, src_rotation_matrix.transpose(1, 2)) + src_box_center.unsqueeze(1).expand(-1, 8, -1) # [num_dts,8,3]
    
    # for box_center, rotation_matrix, box_lwh in zip(src_box_center, src_rotation_matrix, target_3dboxes_lwh):
    #     l, w, h = box_lwh[0], box_lwh[1], box_lwh[2]
    #     # 3d框在目标物体坐标系下的中心坐标为[0,-h/2,0]，3d框的8个顶点在x、y、z轴上的坐标
    #     x_corners = torch.tensor([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], device=rotation_matrix.device)
    #     y_corners = torch.tensor([0,0,0,0,-h,-h,-h,-h], device=rotation_matrix.device)
    #     z_corners = torch.tensor([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], device=rotation_matrix.device)
    #     corners = torch.stack([x_corners, y_corners, z_corners],dim=1) # [8,3]
    #     corners_3d = torch.matmul(corners, rotation_matrix.t()) + box_center # [8,3]
    #     pre_3dboxes_list.append(corners_3d)
        
    return pre_3dboxes # torch.stack(pre_3dboxes_list,dim=0) # list 13 [8,3]

def xyzs_to_xys(bz_tgt_P2, bz_src_3dboxes):
    """_summary_

    Args:
        bz_tgt_P2 (tensor[nums,3,4]): _description_
        bz_src_3dboxes (tensor[nums,8,3]): 3dboxes的三维坐标
    Return
        3dboxes_xy(tensor[nums,8,2]): 投影后的3dboxes的二维坐标
    """
    ones_column = torch.ones((bz_src_3dboxes.shape[0],bz_src_3dboxes.shape[1],1), device=bz_src_3dboxes.device, dtype=torch.float32)
    boxes_xyz_homo = torch.cat((bz_src_3dboxes, ones_column), dim=2)  # [13, 8, 4]
    boxes_xy = torch.matmul(boxes_xyz_homo, bz_tgt_P2.permute(0, 2, 1))  # [13, 8, 3]
    boxes_xy = boxes_xy[:, :, :2] / boxes_xy[:, :, 2:]
    
    # boxes_xy_list = []
    # for P2,boxes_xyz in zip(bz_tgt_P2, bz_src_3dboxes):
    #     ones_column = torch.ones(boxes_xyz.shape[0], 1, device=boxes_xyz.device, dtype=torch.float32)
    #     boxes_xyz_homo = torch.cat((boxes_xyz, ones_column), dim=1) # [8,4]
    #     boxes_xy = torch.matmul(boxes_xyz_homo, P2.t()) # [8,3]
    #     boxes_xy = boxes_xy[:, :2]/boxes_xy[:, 2:]
    #     boxes_xy_list.append(boxes_xy)
    # boxes_xy = torch.stack(boxes_xy_list, dim=0) # [13,8,2]
    return boxes_xy[:, :, :2]

def projection(bz_tgt_P2, bz_src_3dboxes):
    """将3dbox投影到图像上，并获取这八个点的最小外接矩形

    Args:
        bz_tgt_P2 (tensor[13,3,4]): 相机投影矩阵
        bz_src_3dboxes (tensor[13,8,3]): 需要投影的3dbox的八个点的三维坐标
    return:
        min_max_coords (tensor[13,4,2]):八个点的最小外接矩形的四个点的二维坐标
    """
    ones_column = torch.ones((bz_src_3dboxes.shape[0],bz_src_3dboxes.shape[1],1), device=bz_src_3dboxes.device, dtype=torch.float32) # tensor[13,8,1]
    boxes_xyz_homo = torch.cat((bz_src_3dboxes, ones_column), dim=2)  # [13, 8, 4]
    boxes_xy = torch.matmul(boxes_xyz_homo, bz_tgt_P2.permute(0, 2, 1))  # [13, 8, 3]
    boxes_xy = boxes_xy[:, :, :2] / boxes_xy[:, :, 2:]
    # 获取最小外接矩形
    x = boxes_xy[:, :, 0]
    y = boxes_xy[:, :, 1]
    min_x, _ = torch.min(x, dim=1, keepdim=True)
    max_x, _ = torch.max(x, dim=1, keepdim=True)
    min_y, _ = torch.min(y, dim=1, keepdim=True)
    max_y, _ = torch.max(y, dim=1, keepdim=True)
    min_max_coords = torch.cat((min_x, min_y, max_x, max_y), dim=1)  # [13, 1, 4]
    # boxes_xy_list = []
    # for P2,boxes_xyz in zip(bz_tgt_P2, bz_src_3dboxes):
    #     ones_column = torch.ones(boxes_xyz.shape[0], 1, device=boxes_xyz.device, dtype=torch.float32)
    #     boxes_xyz_homo = torch.cat((boxes_xyz, ones_column), dim=1) # [8,4]
    #     boxes_xy = torch.matmul(boxes_xyz_homo, P2.t()) # [8,3]
    #     boxes_xy = boxes_xy[:, :2]/boxes_xy[:, 2:]
    #     boxes_xy_list.append(boxes_xy)
    # boxes_xy = torch.stack(boxes_xy_list, dim=0) # [13,8,2]
    # # 获取最小外接矩形
    # x = boxes_xy[:,:,0]
    # y = boxes_xy[:,:,1]
    # min_x, _ = torch.min(x, dim=1, keepdim=True)
    # max_x, _ = torch.max(x, dim=1, keepdim=True)
    # min_y, _ = torch.min(y, dim=1, keepdim=True)
    # max_y, _ = torch.max(y, dim=1, keepdim=True)
    # min_max_coords = torch.cat((min_x, min_y, max_x, max_y), dim=1) # [13,4]
    return min_max_coords

class SetCriterionDynamicK(nn.Module):
    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_fed_loss:
            self.fed_loss_num_classes = 50
            from detectron2.data.detection_utils import get_fed_loss_cls_weights
            cls_weight_fun = lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER)  # noqa
            fed_loss_cls_weights = cls_weight_fun()
            assert (
                    len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        batch_size = len(targets)

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)# Size([2, 500])
        src_logits_list = []
        target_classes_o_list = []
        # target_classes[idx] = target_classes_o
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0] # 表示哪些预测边界框被选中 ture or flase
            gt_multi_idx = indices[batch_idx][1]# 表示每个选中的预测边界框所匹配的真实边界框的索引。
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]# 第几张图预测的类别
            target_classes_o = targets[batch_idx]["labels"] # 第几张图真值目标类别
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx] # 将预测到真实边界框的索引对应的类别写到 Size([2, 500])

            src_logits_list.append(bz_src_logits[valid_query])# 每个批次中被选中的预测类别 Size([28, 10])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])# 每个批次中预测到真实边界框的索引对应的类别

        if self.use_focal or self.use_fed_loss:
            num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1 # 所有批次中匹配的真实边界框的总数61

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device) # 全零张量torch.Size([2, 500, 11])
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) #将真实类别信息填充到相应的位置

            gt_classes = torch.argmax(target_classes_onehot, dim=-1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]# 移除 one-hot 编码的最后一列（填充列）

            src_logits = src_logits.flatten(0, 1)# Size([1000, 10]) [批次数 * 预测框数量, 类别数]
            target_classes_onehot = target_classes_onehot.flatten(0, 1)# Size([1000, 10])
            if self.use_focal:# Size([1000, 10]) Focal Loss
                cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
            else:
                cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
            if self.use_fed_loss: # False
                K = self.num_classes # 类别数 
                N = src_logits.shape[0]# 批次数
                fed_loss_classes = self.get_fed_loss_classes(
                    gt_classes,
                    num_fed_loss_classes=self.fed_loss_num_classes,
                    num_classes=K,
                    weight=self.fed_loss_cls_weights,
                )
                fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
                fed_loss_classes_mask[fed_loss_classes] = 1
                fed_loss_classes_mask = fed_loss_classes_mask[:K]
                weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()

                loss_ce = torch.sum(cls_loss * weight) / num_boxes
            else:
                loss_ce = torch.sum(cls_loss) / num_boxes

            losses = {'loss_ce': loss_ce}
        else:
            raise NotImplementedError

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'] # 预测的边界框 Size([2, 500, 4])

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]# 表示哪些预测边界框被选中 ture or flase
            gt_multi_idx = indices[batch_idx][1]# 表示每个选中的预测边界框所匹配的真实边界框的索引。
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy'] # 本批次的图片大小真值
            bz_src_boxes = src_boxes[batch_idx] 
            bz_target_boxes = targets[batch_idx]["boxes"]  # normalized (cx, cy, w, h) 归一化的框真值
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  # absolute (x1, y1, x2, y2)框的真值
            pred_box_list.append(bz_src_boxes[valid_query])# Size([28, 4])被选中的边界框的预测值
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  # normalize (x1, y1, x2, y2) 归一化被选中的边界框的预测值
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])# 被选中的归一化的框真值
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])# 被选中的框真值

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_norm = torch.cat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0] # 61

            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses
    
    def loss_pose6dof(self, outputs, targets, indices, num_boxes):
        """6dofloss, consisting of l1 loss of rotation and translation.
        targets dicts must contain the key "translation_matrix" containing a tensor of dim [nb_target_boxes, 3]
        targets dicts must contain the key "rotation_matrix" containing a tensor of dim [nb_target_boxes, 3, 3]
        """
        assert 'pred_poses' in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_poses = outputs['pred_poses']  # [2,500,6]
        batch_size = len(targets)
        
        pred_translation_list = []
        pred_rotation_list = []
        tgt_translation_list = []
        tgt_rotation_list = []
        tgt_area_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]  # [500] 预测框是否被选中 true or false
            gt_multi_idx = indices[batch_idx][1] # [30] 被选中的预测框的所匹配的真值框的索引
            if len(gt_multi_idx) == 0:
                continue
            bz_target_translation_vector = targets[batch_idx]['translation_matrix'][gt_multi_idx] # 平移 [78,3]
            bz_target_rotation_matrix = targets[batch_idx]['rotation_matrix'][gt_multi_idx] # 旋转[78,3,3]
            # bz_target_rotation_angle = matrix2angle(bz_target_rotation_matrix)
            bz_src_translation_vector = src_poses[batch_idx][:, :3][valid_query]  # 预测平移 [500,3]
            bz_src_rotation_angle = src_poses[batch_idx][:, 3:][valid_query]  # 预测旋转 [500,3]
            bz_target_area = targets[batch_idx]["area"][gt_multi_idx] 
            
            pred_translation_list.append(bz_src_translation_vector) # [30,3]
            pred_rotation_list.append(bz_src_rotation_angle)  # [30,3]
            tgt_translation_list.append(bz_target_translation_vector) # [30,3]
            tgt_rotation_list.append(bz_target_rotation_matrix) # [30,3]
            tgt_area_list.append(bz_target_area)

        if len(pred_translation_list) != 0 or len(pred_rotation_list) != 0:
            src_translation = torch.cat(pred_translation_list) # batch个预测正样本被合并 [61,3]
            src_rotation = torch.cat(pred_rotation_list)  # [61,9]
            target_translation = torch.cat(tgt_translation_list) # [61,3]
            target_rotation = torch.cat(tgt_rotation_list) # [61,9]
            num_boxes = src_translation.shape[0] # (src_translation.shape[0]+ src_rotation.shape[0])/2 
            
            # # 1、使用真值距离作为权重
            # # 3dbox底面中心点到相机的距离，
            # trans2camera = torch.norm(target_translation,dim=1).unsqueeze(1) # [500,3]->[500,1]
            # # 最大值设置为200
            # trans2camera_max = 200 
            # translation_weight = 4*torch.sin(trans2camera * math.pi / trans2camera_max / 2) + 1 # 范围为(1,5)
            
            # # 2、xyz三个维度给与不同的权重
            # x_weight = 4*torch.sin(target_translation[:,0].abs() * math.pi / 62 / 2) + 1 # 范围为(1,5)
            # y_weight = 4*torch.sin(target_translation[:,1].abs() * math.pi / 42 / 2) + 1 # 范围为(1,5)
            # z_weight = 4*torch.sin(target_translation[:,2].abs() * math.pi / 198 / 2) + 1 # 范围为(1,5)
            # xyz_weight = torch.cat((x_weight.unsqueeze(1), y_weight.unsqueeze(1), z_weight.unsqueeze(1)),dim=1)
            
            # # 3、使用面积作为权重，映射函数为exp(1 - x/20000)，面积越小权重越大
            # tgt_area = torch.cat(tgt_area_list) # 
            # area_weight = torch.exp(1 - tgt_area/20000).unsqueeze(1)

            losses = {}
            loss_translation = F.l1_loss(src_translation, target_translation, reduction='none') # [61,3]
            losses['loss_translation'] = loss_translation.sum() / num_boxes / 3 # 100 3某个轴上的平移误差，单位为m
            loss_rotation = F.l1_loss(src_rotation, target_rotation, reduction='none') # [61,3]
            losses['loss_rotation'] = loss_rotation.sum() / num_boxes / 3  # 某个轴上的旋转欧拉角误差，单位为°
        else:
            losses = {'loss_translation': outputs['pred_boxes'].sum() * 0,
                        'loss_rotation': outputs['pred_boxes'].sum() * 0}

        return losses
    
    def loss_union3d2d(self, outputs, targets, indices, num_boxes):
        """ 
        Project 3dbox into 2d space and then obtain its maximum circumscribed rectangle, and design an l1loss with the result using gt2dbox
        """
        assert 'pred_poses' in outputs
        src_poses = outputs['pred_poses']  # [2,500,6] 
        src_boxes = outputs['pred_boxes'] # 预测的边界框 Size([2, 500, 4]) abs xyxy
        batch_size = len(targets)
        
        src_3dboxes_list = []
        tgt_3dboxes_list = []
        src_boxes_list = []
        target_boxes_list = []
        src_2dboxes_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]# 表示哪些预测边界框被选中 ture or flase
            gt_multi_idx = indices[batch_idx][1]# 表示每个选中的预测边界框所匹配的真实边界框的索引。
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy'] # 本批次的图片大小真值
            # 第一部分 pre3dbox->gt3dbox: L1(pre3dbox, gt3dbox)
            bz_target_3dboxes = targets[batch_idx]['3dboxes'][gt_multi_idx] # tensor
            tgt_3dboxes_list.append(bz_target_3dboxes)
            bz_target_3dboxes_lwh = targets[batch_idx]['lwhs'][gt_multi_idx] # [13,3]
            bz_src_pose6dof = src_poses[batch_idx][valid_query] # [13,6]
            bz_src_3dboxes = box3d(bz_src_pose6dof, bz_target_3dboxes_lwh) # [13,8,3]
            src_3dboxes_list.append(bz_src_3dboxes)
            
            # 第二部分pre3dbox->pre2dbox：L1(2dbox, pre2dbox) & L1(2dbox, gt2dbox)
            bz_tgt_P2 = targets[batch_idx]['P2s'][gt_multi_idx] # [30,3,4]
            bz_src_2dboxes = projection(bz_tgt_P2, bz_src_3dboxes) # 通过P2投影矩阵将pre3d投影到图像上得到3dbox的xy坐标 [13,4]  abs xyxy
            src_2dboxes_list.append(bz_src_2dboxes/bz_image_whwh) # norm(x,y,x,y)
            bz_src_boxes = src_boxes[batch_idx][valid_query] # [30,4]预测的2dbox
            src_boxes_list.append(bz_src_boxes/bz_image_whwh) # abs xyxy 
            bz_target_boxes = targets[batch_idx]["boxes"][gt_multi_idx] # [30,4]真值2dbox norm(cx,cy,w,h)
            target_boxes_list.append(box_cxcywh_to_xyxy(bz_target_boxes))
        
        if len(target_boxes_list) != 0:
            src_3dboxes = torch.cat(src_3dboxes_list) # prepose+gtlwh->pre3d [30,8,3]
            tgt_3dboxes = torch.cat(tgt_3dboxes_list) # gt3d [30,8,3]
            src_2dboxes = torch.cat(src_2dboxes_list) # pre3d+P2->2dbox[30,4]
            src_o_boxes = torch.cat(src_boxes_list) # pre2dbox[30,4]
            tgt_boxes = torch.cat(target_boxes_list) # gt2dbox[30,4]
            
            num_boxes = tgt_boxes.shape[0]
            
            losses = {}
            loss_3dbox = F.l1_loss(src_3dboxes, tgt_3dboxes, reduction='none')
            losses['loss_3dbox'] = loss_3dbox.sum() / num_boxes / 24 # 每个方向上的误差
            # if losses['loss_3dbox']>4:
            #     losses['loss_3dbox'] = a * losses['loss_3dbox'] + b # 线性映射 将4-550的数值映射到4-5之间
            loss_2dbox2tgtbox = F.l1_loss(src_2dboxes, tgt_boxes, reduction='none')
            losses['loss_2dbox2tgtbox'] = loss_2dbox2tgtbox.sum() / num_boxes
            loss_2dbox2srcbox = F.l1_loss(src_2dboxes, src_o_boxes, reduction='none')
            losses['loss_2dbox2srcbox'] = loss_2dbox2srcbox.sum() / num_boxes # inf
        else:
            losses = {  'loss_3dbox': outputs['pred_poses'].sum() * 0,
                        'loss_2dbox2tgtbox': outputs['pred_poses'].sum() * 0,
                        'loss_2dbox2srcbox': outputs['pred_poses'].sum() * 0,
            } 
            
        return losses
    
    def loss_3dgiou(self, outputs, targets, indices, num_boxes):
        """ 
        Project 3dbox into 2d space and then obtain its maximum circumscribed rectangle, and design an l1loss with the result using gt2dbox
        """
        assert 'pred_poses' in outputs
        src_poses = outputs['pred_poses']  # [2,500,6] 
        src_boxes = outputs['pred_boxes'] # 预测的边界框 Size([2, 500, 4]) abs xyxy
        batch_size = len(targets)
        
        src_3dboxes_list = []
        tgt_3dboxes_list = []
        src_boxes_list = []
        target_boxes_list = []
        src_2dboxes_abs_xyxy_list = []
        src_2dboxes_list = []
        tgt_2dboxes_abs_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]# 表示哪些预测边界框被选中 ture or flase
            gt_multi_idx = indices[batch_idx][1]# 表示每个选中的预测边界框所匹配的真实边界框的索引。
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy'] # 本批次的图片大小真值
            # 第一部分 pre3dbox->gt3dbox: L1(pre3dbox, gt3dbox)
            bz_target_3dboxes = targets[batch_idx]['3dboxes'][gt_multi_idx] # tensor
            tgt_3dboxes_list.append(bz_target_3dboxes)
            bz_target_3dboxes_lwh = targets[batch_idx]['lwhs'][gt_multi_idx] # [13,3]
            trans_factor = torch.tensor([[31,21,100]],device=bz_image_whwh.device)
            rotate_factor = torch.tensor([[90,40,90]],device=bz_image_whwh.device)  
            bz_src_pose6dof = src_poses[batch_idx][valid_query]*torch.cat((trans_factor,rotate_factor),dim=-1) # [13,6]
            bz_src_3dboxes = box3d(bz_src_pose6dof, bz_target_3dboxes_lwh) # [13,8,3]
            src_3dboxes_list.append(bz_src_3dboxes)
            
            # 第二部分pre3dbox->pre2dbox：L1(2dbox, pre2dbox) & L1(2dbox, gt2dbox)
            bz_tgt_P2 = targets[batch_idx]['P2s'][gt_multi_idx] # [30,3,4]
            bz_src_2dboxes = projection(bz_tgt_P2, bz_src_3dboxes) # 通过P2投影矩阵将pre3d投影到图像上得到3dbox的xy坐标 [13,4]  abs xyxy
            src_2dboxes_abs_xyxy_list.append(bz_src_2dboxes)
            src_2dboxes_list.append(bz_src_2dboxes/bz_image_whwh) # norm(x,y,x,y)
            bz_src_boxes = src_boxes[batch_idx][valid_query] # [30,4]预测的2dbox
            src_boxes_list.append(bz_src_boxes/bz_image_whwh) # abs xyxy 
            bz_target_boxes = targets[batch_idx]["boxes"][gt_multi_idx] # [30,4]真值2dbox norm(cx,cy,w,h)
            target_boxes_list.append(box_cxcywh_to_xyxy(bz_target_boxes))
            tgt_2dboxes_abs_xyxy_list.append(targets[batch_idx]["boxes_xyxy"][gt_multi_idx]) # 真值2dbox abs xyxy
        
        if len(tgt_3dboxes_list) != 0:
            src_3dboxes = torch.cat(src_3dboxes_list) # prepose+gtlwh->pre3d [30,8,3]
            tgt_3dboxes = torch.cat(tgt_3dboxes_list) # gt3d [30,8,3]
            src_2dboxes_abs_xyxy = torch.cat(src_2dboxes_abs_xyxy_list)  
            src_2dboxes = torch.cat(src_2dboxes_list) # pre3d+P2->2dbox[30,4] 
            src_o_boxes = torch.cat(src_boxes_list) # pre2dbox[30,4]
            tgt_boxes = torch.cat(target_boxes_list) # gt2dbox[30,4]
            tgt_2dboxes_abs_xyxy = torch.cat(tgt_2dboxes_abs_xyxy_list)
            
            num_boxes = tgt_3dboxes.shape[0]
            
            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_3dbox = F.l1_loss(src_2dboxes, tgt_boxes, reduction='none') # norm xyxy
            losses['loss_3dbbox'] = loss_3dbox.sum() / num_boxes 
            # Calculating 3d giou requires abs xyxy
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_2dboxes_abs_xyxy, tgt_2dboxes_abs_xyxy)) 
            losses['loss_3dgiou'] = loss_giou.sum() / num_boxes
        else:
            losses = {  'loss_3dbbox': outputs['pred_poses'].sum() * 0,
                        'loss_3dgiou': outputs['pred_poses'].sum() * 0,
            } 
            
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'pose6dof': self.loss_pose6dof,
            'union3d2d': self.loss_union3d2d,
            '3dgiou': self.loss_3dgiou,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, _ = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cfg, cost_class: float = 1, 
                 cost_bbox: float = 1, 
                 cost_giou: float = 1,
                 cost_t: float = 1, 
                 cost_r: float = 1, 
                 cost_3dgiou: float = 1,
                 cost_3dbbox: float = 1,
                 cost_mask: float = 1, 
                 use_focal: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_t = cost_t
        self.cost_r = cost_r
        self.cost_3dgiou = cost_3dgiou
        self.cost_3dbbox = cost_3dbbox
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.ota_k = cfg.MODEL.DiffusionDet.OTA_K
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"
    
    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.use_focal or self.use_fed_loss:
                out_prob = outputs["pred_logits"].sigmoid()  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"] # [batch_size,  num_queries, 4]
                out_pose = outputs["pred_poses"] # [batch_size,  num_queries, 6]  [2,500,6]
            else:
                out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]
                out_pose = outputs["pred_poses"]

            indices = []
            matched_ids = []
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]  # [num_proposals, 4]
                bz_out_prob = out_prob[batch_idx] # [num_proposals, 10]
                bz_pose = out_pose[batch_idx] # [num_proposals, 6]
                bz_tgt_ids = targets[batch_idx]["labels"] # [num_proposals]=78
                num_insts = len(bz_tgt_ids)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    continue
                
                bz_gtboxs = targets[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
                bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']  # [num_gt, 4]
                
                pre_trans = bz_pose[:, :3] # [500, 3]
                ones_column = torch.ones((pre_trans.shape[0],1), device=pre_trans.device, dtype=torch.float32)
                pre_trans_column = torch.cat((pre_trans, ones_column), dim=1) # [500, 4]
                pre_trans_xy = torch.matmul(pre_trans_column, targets[batch_idx]["P2s"][0].permute(1,0))  # [13, 8, 3]
                pre_trans_center = pre_trans_xy[:, :2] / pre_trans_xy[:, 2:] 
           
                pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy)# 边界框iou损失(交并比) Size([500, 78]) 为什么使用iou而不是giou
                
                
                # Compute the classification cost. 分类损失
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log()) # Size([500, 10])
                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log()) # Size([500, 10])
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids] # 正样本损失减去负样本损失
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                else:
                    cost_class = -bz_out_prob[:, bz_tgt_ids]

                # Compute the L1 cost between boxes
                # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
                # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
                # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

                bz_image_size_out = targets[batch_idx]['image_size_xyxy'] # [1920., 1088., 1920., 1088.]
                bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt'] # [1920., 1088., 1920., 1088.]

                bz_out_bbox_ = bz_boxes / bz_image_size_out  # 预测的边界框normalize (x1, y1, x2, y2)
                bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  # 真值的边界框normalize (x1, y1, x2, y2)
                cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1) # 1范数（绝对值之和）Size([500, 78])normalize (x1, y1, x2, y2)

                # 取负使其一起参与到成本最小化的优化过程中GIoU = IoU - (C - U) / C 
                cost_giou = -generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy) # 计算广义交并比损失即GIoU Size([500, 78])

                # # 计算6dof loss，平移和旋转分开
                bz_pose_translation_vector = bz_pose[:, :3] # 获取预测值的平移分量 Size([500, 3])
                bz_pose_rotation_angle = bz_pose[:, 3:] # 获取预测值的旋转分量 Size([500, 3])
                # # 获取真值的旋转分量；与其形容为三个旋转角？弧度角？
                bz_gtpose_translation_vector = targets[batch_idx]["translation_matrix"] # size([78,3])
                bz_gtpose_rotation_angle = targets[batch_idx]["rotation_matrix"]
                # # 计算平移损失，平移可以直接用预测值与真值进行l1计算，因为真值已经是物体在相机坐标系下的三维坐标（rope3d到kitti格式转换时也没有对真值做变换）
                cost_t_pose = torch.cdist(bz_pose_translation_vector, bz_gtpose_translation_vector, p=1)# Size([500, 78])
                cost_r_pose = torch.cdist(bz_pose_rotation_angle, bz_gtpose_rotation_angle, p=1)# Size([500, 78])
                
                gious_pose = generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy) # [-1,1] iou, union = box_iou(boxes1, boxes2)
                _,index = torch.max(gious_pose,dim=-1)
                bz_prelwhs = targets[batch_idx]['lwhs'][index] 
                # trans_factor = torch.tensor([[31,21,100]],device=bz_pose.device)
                # rotate_factor = torch.tensor([[90,40,90]],device=bz_pose.device) 
                # bz_3dboxes = box3d(bz_pose*torch.cat((trans_factor,rotate_factor),dim=-1),bz_prelwhs)
                # bz_3dboxes = carla_box3d(bz_pose, bz_prelwhs)
                # bz_gtP2s = targets[batch_idx]['P2s'][index]
                # bz_proj2dboxes = projection(bz_gtP2s, bz_3dboxes) # [num_dts,4]
                # cost_3dgiou = -generalized_box_iou(bz_proj2dboxes, bz_gtboxs_abs_xyxy) # abs xyxy
                # cost_3dbbox = torch.cdist(bz_proj2dboxes/bz_image_size_out, bz_tgt_bbox_, p=1)# [500, 78] normalize (x1, y1, x2, y2)          
                
                pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy)
                # pair_wise_pose_ious = ops.box_iou(bz_proj2dboxes, bz_gtboxs_abs_xyxy)
                # detect_pose_ious = (pair_wise_ious + pair_wise_pose_ious)/2
                
                cost_pose = self.cost_t * cost_t_pose + self.cost_r * cost_r_pose # + self.cost_3dgiou * cost_3dgiou + self.cost_3dbbox * cost_3dbbox

                # 仅拿预测与真值做是否在目标区域内的true与false的判断，返回的也是该位置的true与false值
                fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                    box_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h) 预测
                    box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),  # absolute (cx, cy, w, h) 真值
                    # box_xyxy_to_cxcywh(bz_proj2dboxes),
                    expanded_strides=32
                )# 锚点是否在目标边界框内或中心区域内,锚点是否在目标边界框内和中心区域内  # pose可还原出2d投影，能算做一种限制 def ...
                
                # cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + cost_pose + 100.0 * (~is_in_boxes_and_center)
                # 100.0 * (~is_in_boxes_and_center)：一个惩罚项(~按位取反)，用于处理那些既不在目标边界框内，也不在目标边界框中心附近的预测边界框。这个惩罚项的权重是100.0。
                # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt] biou...
                cost_detect = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)
                cost = cost_detect + cost_pose # detect + pose -> total cost
                cost[~fg_mask] = cost[~fg_mask] + 10000.0 # 未成功匹配的预测边界框的损失值增加10000.0

                # if bz_gtboxs.shape[0]>0: 动态K匹配算法,预测边界框分配真实边界框 indices_batchi([500], [30])
                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0]) # loss,边界框损失，78
                # indices表示预测边界框与真实边界框的匹配关系，matched_query_id表示与真实边界框匹配的预测边界框的索引
                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)
                
        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides): # 预测，真值
        "增加了3dbox的2d投影中心是否在gtbox内部的判断"
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  # 从(cx, cy, w, h)格式转换为(x1, y1, x2, y2)格式

        anchor_center_x = boxes[:, 0].unsqueeze(1) # 是 预测boxs 中心点的xy坐标 [500, 1]
        anchor_center_y = boxes[:, 1].unsqueeze(1)
        # proj_center_x = proj_boxes[:, 0].unsqueeze(1) # 是 3dbox的2d投影 中心点的xy坐标 [500, 1]
        # proj_center_y = proj_boxes[:, 1].unsqueeze(1)
        

        # whether the center of each anchor is inside a gt box判断预测boxs的中心点是否在真值box四个点范围内
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0) # [500, 1]+[1, 78]->[500,78]中为true、false
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long() 
                        # + c_l.long() + c_r.long() + c_t.long() + c_b.long() 
                        ) == 4)  # [500,78]锚点完全位于真值框的四个顶点范围内
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query] 500  计算完全位于真值框的四个顶点范围内锚点的个数
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()
                        # + c_l.long() + c_r.long() + c_t.long() + c_b.long()
                          ) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all # [500] 计算锚点是否在目标边界框内或中心区域内，并将结果存储在is_in_boxes_anchor变量中。
        is_in_boxes_and_center = (is_in_boxes & is_in_centers) # [500,78] 计算锚点是否同时在目标边界框内和中心区域内，并将结果存储在is_in_boxes_and_center变量中。

        return is_in_boxes_anchor, is_in_boxes_and_center # *10000  *100

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt): # loss,边界框损失，真值框的个数
        matching_matrix = torch.zeros_like(cost)  # [500,num_gt]78
        ious_in_boxes_matrix = pair_wise_ious  # [500,num_gt]
        n_candidate_k = self.ota_k 
        # n_candidate_k = 50

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        # 从ious_in_boxes_matrix中选择最大的n_candidate_k个iou值，并将它们的和作为dynamic_k。相似度最大，最接近真值
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)  # topk_ious：[n_candidate_k, 78]
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)  # [78]第0维度求和，并限制元素不小于1，保证一个gt至少有一个正样本

        for gt_idx in range(num_gt):
            # 找到cost中与其匹配的dynamic_k个最小值的索引，并将matching_matrix中对应位置的值设为1
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)  # pos_idx[1]
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1) # 维度1上求和[500,78]->[500]

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]
        # 返回匹配结果和最小cost对应的查询索引;selected_query[500], gt_indices[30], matched_query_id[78]
        return (selected_query, gt_indices), matched_query_id
