# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): #time:tensor([999], device='cuda:0')
        device = time.device
        half_dim = self.dim // 2 #128
        embeddings = math.log(10000) / (half_dim - 1) #math.log(10000) / 127 = 0.07252236513367073
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)# 返回返回大小为end−start/step 的一维张量 以 step为步长等间隔取值 start默认为0，step步长，默认值：1 torch.Size([128])
        embeddings = time[:, None] * embeddings[None, :]# Size([1, 1]) Size([1, 128]) embeddings:Size([1, 128])
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # 拼接，最后一维 Size([1, 256])
        return embeddings



class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        # ROIPooler(
        #   (level_poolers): ModuleList(
        #     (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, aligned=True)
        #     (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2, aligned=True)
        #     (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2, aligned=True)
        #     (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2, aligned=True)
        #   )
        # Build heads.
        num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES # 10
        d_model = cfg.MODEL.DiffusionDet.HIDDEN_DIM # 256
        dim_feedforward = cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD #2048
        nhead = cfg.MODEL.DiffusionDet.NHEADS # 8
        dropout = cfg.MODEL.DiffusionDet.DROPOUT #0.0
        activation = cfg.MODEL.DiffusionDet.ACTIVATION #activation：‘relu’
        num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS # 6
        num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        rcnn_head = RCNNHead(cfg, d_model, num_proposals, num_classes, dim_feedforward, nhead, dropout, activation)# (self_attn): MultiheadAttention((out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True))(inst_interact): DynamicConv(
        # rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)# (self_attn): MultiheadAttention((out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True))(inst_interact): DynamicConv(
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads # 6
        self.return_intermediate = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION # ture
        
        # Gaussian random feature embedding layer for time
        self.d_model = d_model # 256
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential( # Sequential是一个有序的容器，网络层将按照在传入构造函数的顺序依次被添加到计算图中执行，同时以 Module 的形式进行封装。
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL # ture
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS #false
        self.num_classes = num_classes# 80
        if self.use_focal or self.use_fed_loss:
            prior_prob = cfg.MODEL.DiffusionDet.PRIOR_PROB # 0.01
            self.bias_value = -math.log((1 - prior_prob) / prior_prob) # bias.value -4.59511985013459
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():# parameters()：返回一个迭代器，包含模型的所有参数。torch.Size([768, 256])
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)# xaiver_uniform_()：使用均匀分布初始化权重，均匀分布的范围是[-a, a]，其中a = sqrt(6 / (fan_in + fan_out))，其中fan_in是输入神经元的数量，fan_out是输出神经元的数量。

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES # infeatures = ['p2', 'p3', 'p4', 'p5']
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION # 7
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features) # (0.25, 0.125, 0.0625, 0.03125)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO # 2
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE # 'ROIAlignV2'

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features] # [256, 256, 256, 256]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, t, init_features):# 特征图，预测框，时间tensor([999], device='cuda:0')，None
        # assert t shape (batch_size)
        time = self.time_mlp(t) # 位置编码，升维self.time_mlp is a nn.Sequential time.shape Size([1, 1024])

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_pred_pose = []

        bs = len(features[0]) # features[0].shape torch.Size([1, 256, 200, 272]) len= 1
        bboxes = init_bboxes # [2, 500, 4]
        num_boxes = bboxes.shape[1]  # 500

        if init_features is not None:# None
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None
        
        for head_idx, rcnn_head in enumerate(self.head_series): # head_idx:0, 1, 2, 3, 4, 5 RCNN_head(())
            # class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler, time)
            class_logits, pred_bboxes, proposal_features, pred_pose = rcnn_head(features, bboxes, proposal_features, self.box_pooler, time) # (batchsize,num_anchors,...)
            # class_logits = class_logits.detach()
            # pred_bboxes = pred_bboxes.detach()
            # pred_pose = pred_pose.detach()
            if self.return_intermediate:# True
                inter_class_logits.append(class_logits)# list inter_class_logits[0].shape Size([1, 500, 80]) 最后 list 6
                inter_pred_bboxes.append(pred_bboxes)# list inter_pred_bboxes[0].shape Size([1, 500, 4])
                inter_pred_pose.append(pred_pose)# torch.Size([2, 500, 6])
            bboxes = pred_bboxes.detach()#Size([1, 500, 4])
            # detach是将tensor从计算图中分离出来，不再参与梯度计算，但是仍然可以使用，这样可以节省显存
        
        if self.return_intermediate: # 是否返回中间结果
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), torch.stack(inter_pred_pose) # 拼接成Size([6, 1, 500, 80])torch.stack是将多个tensor按照指定维度进行拼接

        return class_logits[None], pred_bboxes[None], pred_pose[None]
    
class RCNNHead(nn.Module): # 多头注意力机制的网络

    def __init__(self, cfg, d_model,num_proposals, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 增加kan
        from .kan import KAN
        # self.kan1 = KAN([d_model, num_proposals, d_model*8]) # 替换了self.linear1、2
        # self.kan2 = KAN([d_model*8, num_proposals, d_model])
        # self.kan_block_time_mlp = KAN([d_model*4, 1, d_model*2])
        # self.kan_block_time_mlp_pose = KAN([d_model*4, 1, d_model*2])

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)        

        # dynamic.
        self.self_pose_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_pose_interact = DynamicConv(cfg)

        self.linear_pose1 = nn.Linear(d_model, dim_feedforward) # dim_feedforward -> 10086
        self.linear_pose2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm_pose1 = nn.LayerNorm(d_model)
        self.norm_pose2 = nn.LayerNorm(d_model)
        self.norm_pose3 = nn.LayerNorm(d_model)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))
        self.block_time_mlp_pose = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = cfg.MODEL.DiffusionDet.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            # cls_module.append(KAN([d_model, num_proposals, d_model]))
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)
        
        # reg.
        num_reg = cfg.MODEL.DiffusionDet.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            # reg_module.append(KAN([d_model, num_proposals, d_model]))
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pose_R
        num_pose_R = cfg.MODEL.DiffusionDet.POSE_R
        pose_module_R = list()
        for _ in range(num_pose_R):
            # pose_module_R.append(KAN([d_model, num_proposals, d_model]))
            pose_module_R.append(nn.Linear(d_model, d_model, False))
            pose_module_R.append(nn.LayerNorm(d_model))
            pose_module_R.append(nn.ReLU(inplace=True))
        self.pose_module_R = nn.ModuleList(pose_module_R)
        
        #pose_T
        num_pose_T = cfg.MODEL.DiffusionDet.POSE_T # 3
        pose_module_T = list()
        for _ in range(num_pose_T):
            # pose_module_T.append(KAN([d_model, num_proposals, d_model]))
            pose_module_T.append(nn.Linear(d_model, d_model, False))
            pose_module_T.append(nn.LayerNorm(d_model))
            pose_module_T.append(nn.ReLU(inplace=True))
        self.pose_module_T = nn.ModuleList(pose_module_T)
        
        # pred.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(d_model, num_classes) # KAN([d_model, num_proposals, num_classes]) 
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1) # KAN([d_model, num_proposals, num_classes + 1]) 
        self.bboxes_delta = nn.Linear(d_model,4) # KAN([d_model, num_proposals, 4])
        self.pose_delta_R = nn.Linear(d_model,3) # KAN([d_model, num_proposals, 3])
        self.pose_delta_T = nn.Linear(d_model,3) # KAN([d_model, num_proposals, 3]) # 
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, time_emb):# 特征图，随机的预测框，None, ROIpooler，时间time.shape Size([1, 1024])
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)gcvb
        """

        N, nr_boxes = bboxes.shape[:2] # [2,500,4] torch.sum((bboxes[:,:, 0] >= bboxes[:,:, 2]) | (bboxes[:,:, 1] >= bboxes[:,:, 3])).item() = 63 
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))# ([500, 4]) Boxes:此结构将框列表存储为 Nx4 torch.Tensor。支持一些关于盒子的常用方法
        roi_features = pooler(features, proposal_boxes) # ROIAlign roi_features shape torch.Size([500, 256, 7, 7])

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)# view .Size([1, 500, 256, 49]) mean是对最后一维度求均值，得到的是Size([1, 500, 256])

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)# view Size([500, 256, 49]) permute Size([49, 500, 256])

        # pose_feature = self.pose(N, nr_boxes, pro_features,roi_features, time_emb) # no grad 
        
        # self_att.Self-Attention
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)#view Size([1, 500, 256]) permute Size([500, 1, 256])
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0] #self_attn是多头注意力机制 pro_features2.Size([500, 1, 256])
        pro_features = pro_features + self.dropout1(pro_features2)# dropout1是随机失活，防止过拟合
        pro_features = self.norm1(pro_features)# norm1是归一化

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model) # reshape Size([1, 500, 256])
        pro_features2 = self.inst_interact(pro_features, roi_features)# inst_interact是动态卷积，pro_features2.Size([500, 256])
        pro_features = pro_features + self.dropout2(pro_features2)# Size([1, 500, 256])
        obj_features = self.norm2(pro_features)# Size([1, 500, 256])

        # obj_feature.
        # obj_features2 = self.kan2(self.dropout(self.kan1(obj_features.squeeze(dim=0)))).unsqueeze(0)
        
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)# Size([1, 500, 256])
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)# transpose是转置，reshape Size([500, 256])

        # scale_shift = self.kan_block_time_mlp(time_emb)
        scale_shift = self.block_time_mlp(time_emb)# time_emb.Size([1, 1024]) scale_shift.Size([1, 512])
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)# repeat_interleave是沿着某个维度重复，dim=0是沿着第0维度重复，nr_boxes=500，所以scale_shift.Size([500, 512])
        scale, shift = scale_shift.chunk(2, dim=1)# chunk拆分scale and shift都是Size([500, 256])
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        pose_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)# liner+layernorm+relu
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)# liner+layernorm+relu
        # for pose_layer in self.pose_module:
        #     pose_feature = pose_layer(pose_feature)    
        
        # pose_feature = self.pose(N, nr_boxes, pro_features,roi_features, time_emb)
        pose_feature_R = pose_feature.clone() # self.pose_R(N, nr_boxes, pro_features,roi_features, time_emb)
        pose_feature_T = pose_feature.clone() # self.pose_T(N, nr_boxes, pro_features,roi_features, time_emb)
        for pose_layer_R in self.pose_module_R:
            pose_feature_R = pose_layer_R(pose_feature_R) 
            
        for pose_layer_T in self.pose_module_T:
            pose_feature_T = pose_layer_T(pose_feature_T)
            
        class_logits = self.class_logits(cls_feature) 
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pose_deltas_R = self.pose_delta_R(pose_feature_R)
        pose_deltas_T = self.pose_delta_T(pose_feature_T)
        pose_deltas_TR = torch.cat((pose_deltas_T, pose_deltas_R), dim=1)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))# torch.Size([500, 4]) 生成的box是xyxy abs形式
        # pred_pose = self.apply_pose(pose_deltas, bboxes.view(-1, 4))# torch.Size([500, 4])
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features, pose_deltas_TR.view(N, nr_boxes, -1)
    
    def pose(self,N, nr_boxes, pro_features, roi_features, time_emb):
        
        # self_att.Self-Attention
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)#view Size([1, 500, 256]) permute Size([500, 1, 256])
        pro_features2 = self.self_pose_attn(pro_features, pro_features, value=pro_features)[0] #self_attn是多头注意力机制 pro_features2.Size([500, 1, 256])
        pro_features = pro_features + self.dropout1(pro_features2)# dropout1是随机失活，防止过拟合
        pro_features = self.norm_pose1(pro_features)# norm1是归一化

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model) # reshape Size([1, 500, 256])
        pro_features2 = self.inst_pose_interact(pro_features, roi_features)# inst_interact是动态卷积，pro_features2.Size([500, 256])
        pro_features = pro_features + self.dropout2(pro_features2)# Size([1, 500, 256])
        obj_features = self.norm_pose2(pro_features)# Size([1, 500, 256])

        # obj_feature.
        # obj_features2 = self.kan2(self.dropout(self.kan1(obj_features.squeeze(dim=0)))).unsqueeze(0)
        
        obj_features2 = self.linear_pose2(self.dropout(self.activation(self.linear_pose1(obj_features)))) # _pose
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm_pose3(obj_features)# Size([1, 500, 256])
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)# transpose是转置，reshape Size([500, 256])

        scale_shift = self.block_time_mlp_pose(time_emb)# time_emb.Size([1, 1024]) scale_shift.Size([1, 512])
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)# repeat_interleave是沿着某个维度重复，dim=0是沿着第0维度重复，nr_boxes=500，所以scale_shift.Size([500, 512])
        scale, shift = scale_shift.chunk(2, dim=1)# chunk拆分scale and shift都是Size([500, 256])
        fc_feature = fc_feature * (scale + 1) + shift
        
        pose_feature = fc_feature.clone()
        
        return pose_feature

    def apply_deltas(self, deltas, boxes):# deltas是通过rcnnhead，boxes是一开始随机生成的500个框
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)# boxes是（x1,y1,x2, y2）[1000,4]

        #  torch.sum((boxes[:, 0] >= boxes[:, 2]) | (boxes[:, 1] >= boxes[:, 3])).item() = 63 
        widths = boxes[:, 2] - boxes[:, 0] 
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights #(2.0, 2.0, 1.0, 1.0)
        dx = deltas[:, 0::4] / wx # 取出deltas的x坐标，然后除以wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)# 限制dw的最大值
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None] # 在gtbox的中心点处将gtwh放缩dxdy倍，得到预测后的中心点
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes
    
    def apply_pose(self, deltas, boxes):
        pass
    
class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DiffusionDet.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.DiffusionDet.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
