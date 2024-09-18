# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Instances, ROIMasks

from detectron2.structures import Boxes, ImageList, Instances#, Pose

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK, box3d, projection
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from .util.misc import nested_tensor_from_tensor_list

__all__ = ["DiffusionDet"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0] # 1
    out = a.gather(-1, t) # gather是按照t的索引取a的值，a的shape是[1,500,4]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))) #torch.Size([1, 1, 1])


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1 # 1001
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64) # linsapace是等差数列，从0到1000，1001个数 x:tensor([0., 1., 2., ..., 999., 1000.], dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2 #torch.size([1001])tensor([9.9984e-01, 9.9980e-01, 9.9976e-01,  ..., 9.7135e-06, 2.4284e-06,3.7494e-33], dtype=torch.float64)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] #tensor([1.0000e+00, 9.9996e-01, 9.9991e-01,  ..., 9.7150e-06, 2.4288e-06,3.7500e-33], dtype=torch.float64)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) #torch.size([1000]) tensor([4.1284e-05, 4.6142e-05, 5.0999e-05, 5.5857e-05, 6.0716e-05, 6.5574e-05,
    return torch.clip(betas, 0, 0.999) # clip:将tensor中的元素限制在一个范围内，小于min的元素变为min，大于max的元素变为max


@META_ARCH_REGISTRY.register()
class DiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self,  cfg): #Self: DiffusionDet
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE) # device = 'cuda'

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES # in_features = ['p2', 'p3', 'p4', 'p5', 'p6']
        self.num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES # num_classes = 80
        self.num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS# num_proposals = 500
        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM # hidden_dim = 256
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS # num_heads = 6

        # Build Backbone.
        self.backbone = build_backbone(cfg) # backbone:FPN (fpn_lateral2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))(fpn_output2): (fpn_lateral3):(fpn_output3): 
        self.size_divisibility = self.backbone.size_divisibility # size_divisibility = 32

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP # 1
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps) #torch.size([1000]) betas:tensor([4.1284e-05, 4.6142e-05, 5.0999e-05, 5.5857e-05, 6.0716e-05, 6.5574e-05,)
        alphas = 1. - betas # tensor([1.0000, 1.0000, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999
        alphas_cumprod = torch.cumprod(alphas, dim=0)# cumprod:计算累积乘积，dim=0表示按行计算，dim=1表示按列计算
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.) # F.pad:对输入的tensor在指定的维度上进行填充，value=1.表示填充的值为1 （1,0)表示在第0维度上填充1个1，第1维度上填充0个1
        timesteps, = betas.shape # timesteps = 1000
        self.num_timesteps = int(timesteps) 

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default(1, 1000) = 1 
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps # True
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE #scale = 2.0
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas) #register_buffer:将一个tensor注册为buffer，buffer不是参数，不会被优化器优化，但是会被保存到模型中
        self.register_buffer('alphas_cumprod', alphas_cumprod) 
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) #torch.size([1000])

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20))) # clamp:将tensor中的元素限制在一个范围内，min=1e-20表示小于1e-20的元素都被替换为1e-20
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head. 
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape()) # backbone.output_shape():{'p2': (256, 64),channels=256, height=None, width=None, stride=4), 'p3': (256, 32), 'p4': (256, 16), 'p5': (256, 8), 'p6': (256, 4)}
        # Loss parameters:
        class_weight = cfg.MODEL.DiffusionDet.CLASS_WEIGHT# class_weight = 2.0
        giou_weight = cfg.MODEL.DiffusionDet.GIOU_WEIGHT# giou_weight = 2.0
        l1_weight = cfg.MODEL.DiffusionDet.L1_WEIGHT# l1_weight = 5.0
        tarnslation_weight = cfg.MODEL.DiffusionDet.T_WEIGHT
        rotation_weight = cfg.MODEL.DiffusionDet.R_WEIGHT
        THdbox_weight = cfg.MODEL.DiffusionDet.THD_weight
        TWOdbox2tgtbox_weight = cfg.MODEL.DiffusionDet.TWODTGT_weight
        TWOdbox2srcbox_weight = cfg.MODEL.DiffusionDet.TWOSRC_WEIGHT
        
        giou_3d_weight = cfg.MODEL.DiffusionDet.GIOU_3d_WEIGHT
        bbox_3d_weight = cfg.MODEL.DiffusionDet.BBox_3d_WEIGHT
        
        no_object_weight = cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT# no_object_weight = 0.1
        self.deep_supervision = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION# true
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL# ture
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS #false
        self.use_nms = cfg.MODEL.DiffusionDet.USE_NMS# true

        # Build Criterion.
        matcher = HungarianMatcherDynamicK( # HungarianMatcherDynamicK:匈牙利算法，用于匹配预测框和真实框
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, 
            cost_t=tarnslation_weight,cost_r=rotation_weight,
            cost_3dgiou=giou_3d_weight,cost_3dbbox=bbox_3d_weight,
            use_focal=self.use_focal
        )
        # loss_bbox:5.0 loss_ce:2.0 loss_giou:2.0 0.05 1 0.005 0.125 0.125
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight,
                       "loss_translation":tarnslation_weight,"loss_rotation":rotation_weight,
                       "loss_3dbox":THdbox_weight,"loss_2dbox2tgtbox":TWOdbox2tgtbox_weight,"loss_2dbox2srcbox":TWOdbox2srcbox_weight,
                       "loss_3dgiou":giou_3d_weight,"loss_3dbbox":bbox_3d_weight} 
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()}) # .items()以列表返回可遍历的(键, 值) 元组数组
            weight_dict.update(aux_weight_dict)

        losses = ["labels","boxes","pose6dof"] # ["labels","boxes","pose6dof","union3d2d","3dgiou"]

        self.criterion = SetCriterionDynamicK( # 计算DiffusionDet的loss 
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1) # view(3, 1, 1)将tensor转换为3*1*1的tensor tensor([[[123.6750]],[[116.2800]],[[103.5300]]], device='cuda:0')
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def predict_noise_from_start(self, x_t, t, x0):
        return ( # torch.Size([1000]),时间tensor([999],正态分布的随机数torch.Size([1, 500, 4])
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / #extract取999对应的值tensor([[[20291.1696]]], device='cuda:0', dtype=torch.float64)
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):# 特征图，放缩后图像的宽高，正态分布的随机数，时间tensor([999]，None，True
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale) # 将随机数x限制在-2和2之间 Shape不变 torch.Size([1, 500, 4])
        x_boxes = ((x_boxes / self.scale) + 1) / 2 # 将x(张量)限制在0和1之间 Shape不变 torch.Size([1, 500, 4])
        x_boxes = box_cxcywh_to_xyxy(x_boxes) # 将x_boxes转换为(x1, y1, x2, y2)是左上和右下的坐标 Shape不变 torch.Size([1, 500, 4])
        x_boxes = x_boxes * images_whwh[:, :, :] # None表示在第二个维度上扩展，images_whwh从torch.Size([1, 4])到torch.Size([1, 1, 4]) x_boxes：torch.Size([1, 500, 4])
        # outputs_coord：xyxy abs
        outputs_class, outputs_coord, outputs_pose = self.head(backbone_feats, x_boxes, t, None)# 特征图，预测框，时间tensor，None
        # outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)# 特征图，预测框，时间tensor，None
        # outputs_class: (batch, num_proposals, num_classes) torch.Size([6, 1, 500, 80]),outputs_coord: (batch, num_proposals, 4) torch.Size([6, 1, 500, 4])
        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, :, :]
        x_start = box_xyxy_to_cxcywh(x_start)# 左上右下坐标转为中心点坐标和宽高
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)# torch.Size([1, 500, 4]) 根据公式得到了扩散模型的Zt

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord, outputs_pose

    @torch.no_grad()# torch.no_grad()上下文管理器，用于禁止梯度计算
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        """_summary_

        Args:
            batched_inputs (list:1): 一个batch的验证集的数据(真值)
            backbone_feats (list:4): backone提取的特征图
            images_whwh (_type_): _description_
            images (_type_): _description_
            clip_denoised (bool, optional): _description_. Defaults to True.
            do_postprocess (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        batch = images_whwh.shape[0] # batch = 1
        shape = (batch, self.num_proposals, 4) #(1,500,4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        #total_timesteps = 1000 sampling_timesteps = 1 eta = 1.0 objective = 'pred_x0'
        
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1) #times:tensor([-1., 999.])
        times = list(reversed(times.int().tolist())) # times:[999, -1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(999, -1)]

        img = torch.randn(shape, device=self.device)# 生成正态分布的随机数radnn class：tensor shape:torch.Size([1, 500, 4])  

        ensemble_score, ensemble_label, ensemble_coord, ensemble_pose= [], [], [], []
        x_start = None
        for time, time_next in time_pairs: # time = 999 time_next = -1
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long) # tensor([999], device='cuda:0') full:返回一个tensor，用fill_value填充
            self_cond = x_start if self.self_condition else None # None

            preds, outputs_class, outputs_coord, outputs_pose = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                        self_cond, clip_x_start=clip_denoised) # 特征图，放缩后图像的宽高，正态分布的随机数，时间tensor([999]，None，True
            # preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
            #                                                             self_cond, clip_x_start=clip_denoised) # 特征图，放缩后图像的宽高
            
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start #pred两个pred_noise, pred_x_start都是torch.Size([1, 500, 4])

            if self.box_renewal:  # filter
                # score_per_image, box_per_image, pose_per_image = outputs_class[-1][0], outputs_coord[-1][0], outputs_pose[-1][0] # outputs_class[-1].shape Size([1, 500, 80]) outputs_class[-1][0].shape Size([500, 80])
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5 # 0.5
                score_per_image = torch.sigmoid(score_per_image)# torch.Size([500, 10])经过sigmoid函数
                value, _ = torch.max(score_per_image, -1, keepdim=False) # max返回最大值和最大值的索引 在最后一个维度上 value Size([500])
                keep_idx = value > threshold # keep_idx Size([500]) keep_idx是一个bool类型的tensor
                num_remain = torch.sum(keep_idx)# sum返回所有元素的和 num_remain:tensor(25, device='cuda:0')

                # pred_noise = pred_noise[:, keep_idx, :]# torch.Size([1, 25, 4])
                x_start = x_start[:, keep_idx, :]# torch.Size([1, 25, 4])
                img = img[:, keep_idx, :]# torch.Size([1, 25, 4])
            if time_next < 0: # time_next = -1
                img = x_start
                continue # 回到for

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1: # self.sampling_timesteps = 1
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            pose_per_image = torch.cat(ensemble_pose, dim =0)
            if self.use_nms: # true
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], "pred_poses": outputs_pose[-1]}
            # output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]# torch.Size([1, 500, 80]) 
            box_pred = output["pred_boxes"]# torch.Size([1, 500, 4])
            box_pose = output["pred_poses"]# torch.Size([1, 500, 6])
            image_sizes = [torch.tensor([images.shape[-2], images.shape[-1]]).cuda()]
            
            # 获取3dbox的投影box；box_pose：[2, 500, 6]
            # gious_pose = generalized_box_iou(box_pred[0], batched_inputs[4][0].to(device=box_pose.device))
            # _,index = torch.max(gious_pose,dim=-1)
            # bz_prelwhs = batched_inputs[-4][0].to(device=box_pose.device)[index] 
            # trans_factor = torch.tensor([[31,21,100]],device=box_pose.device)
            # rotate_factor = torch.tensor([[90,40,90]],device=box_pose.device) 
            # bz_3dboxes = box3d(box_pose[0]*torch.cat((trans_factor,rotate_factor),dim=-1),bz_prelwhs)
            # bz_gtP2s = batched_inputs[-3][0].to(device=box_pose.device)[index]
            # bz_proj2dboxes = projection(bz_gtP2s, bz_3dboxes)[None,:,:]
            
            results = self.inference(box_cls, box_pred, image_sizes, box_pose)# images.image_sizes:[(800, 1067)]
            # results = self.inference1(box_cls, box_pred, image_sizes)
        if do_postprocess: # ture
            processed_results = []
            for results_per_image, image_size in zip(results, image_sizes):
                height = image_size[0] # 480
                width = image_size[1] # 640
                # Resize the output instances.  Returns:Instances: the resized output from the model, based on the output resolution
                r = detector_postprocess(results_per_image, height, width) #instances(num_instances=88, image_height=480, image_width=640, fields=[pred_boxes: Boxes(tensor scores: pred_classes: 
                processed_results.append({"instances": r})
            return processed_results
    
    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    
    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        batched_inputs = Data_handing(batched_inputs)
        (zoom_imgs_shape,imgidx,sweep_imgs,gt_labels,gt_2dboxes,gt_3dboxes,gt_lwhs,gt_P2s,gt_translation_matrixs,gt_rotation_matrixs) = batched_inputs
        
        if torch.cuda.is_available():
            # 原本变量为list[tensor[device:cpu]]，此处将device改到gpu上
            zoom_imgs_shape = zoom_imgs_shape.cuda()
            sweep_imgs = sweep_imgs.cuda() 
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
            gt_2dboxes = [gt_2dbox.cuda() for gt_2dbox in gt_2dboxes]
            gt_3dboxes = [gt_3dbox.cuda() for gt_3dbox in gt_3dboxes]
            gt_lwhs = [gt_lwh.cuda() for gt_lwh in gt_lwhs]
            gt_P2s = [gt_P2.cuda() for gt_P2 in gt_P2s]
            gt_translation_matrixs = [gt_translation_matrix.cuda() for gt_translation_matrix in gt_translation_matrixs]
            gt_rotation_matrixs = [gt_rotation_matrix.cuda() for gt_rotation_matrix in gt_rotation_matrixs]
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = sweep_imgs.shape 

        images = sweep_imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)# torch.Size([2, 3, 864, 1536])

        
        # Feature Extraction.
        src = self.backbone(images)
        
        features = list()
        for f in self.in_features:
            feature = src[f]#.detach()
            features.append(feature)
        
        # Prepare Proposals. zoom_imgs_shape.shape:[1,1,4]
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, zoom_imgs_shape, images) # ddim_sample results:instances类型
            return results


        if self.training:
            # 获取图像的instances0: Instances(num_instances=3, image_height=768, image_width=939, fields=[gt_boxes: Boxes(tensor([[499.4634,  65.2913, 560.6661, 105.4887], gt_classes: tensor([29,  0,  0], device='cuda:0')])
            gt_instances = []
            for i in range(len(gt_3dboxes)):
                instances = Instances((zoom_imgs_shape[i, 0, 1], zoom_imgs_shape[i, 0, 0])) # 
                instances.gt_boxes = Boxes(gt_2dboxes[i])
                instances.gt_classes = gt_labels[i]
                instances.gt_3dboxes = gt_3dboxes[i]
                instances.gt_lwhs = gt_lwhs[i]
                instances.gt_P2s = gt_P2s[i]
                instances.gt_translation_matrixs = gt_translation_matrixs[i]
                instances.gt_rotation_matrixs = gt_rotation_matrixs[i]
                gt_instances.append(instances)
                
            # x_boxes:
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            zoom_imgs_shape = zoom_imgs_shape.squeeze(1).to(self.device)
            x_boxes = x_boxes * zoom_imgs_shape[:, None, :] # normal xyxy -> xyxy
            # Draw the diffusion frame on the image
            # visualize_diffusion(x_boxes, images)
            # outputs_class, outputs_coord = self.head(features, x_boxes, t, None) # 63
            # output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1] }

            outputs_class, outputs_coord, outputs_pose = self.head(features, x_boxes, t, None) # 63
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], "pred_poses": outputs_pose[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, "pred_poses": c}
                                         for a, b ,c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_pose[:-1])]
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def prepare_diffusion_trans_concat(self, gt_trans):
        """
        :param gt_trans: (x, y, z), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 3, device=self.device)

        num_gt = gt_trans.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_trans = torch.as_tensor([[0.5, 0.5, 0.5]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_trans = (gt_trans * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_trans, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = x #box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long() # 1000
        noise = torch.randn(self.num_proposals, 4, device=self.device) # 500

        num_gt = gt_boxes.shape[0] # 30
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6;均值为0.5，方差为1
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4) # 
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0) # normal cxcywh 
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale # [-2,2]

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale) # (-2,2)
        x = ((x / self.scale) + 1) / 2.  # (0,1)

        diff_boxes = box_cxcywh_to_xyxy(x) # normal cxcywh -> normal xyxy 

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_3dboxes = targets_per_image.gt_3dboxes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy # 归一化
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes) # normal xyxy -> normal cxcywh [0,1]
            
            gt_trans = targets_per_image.gt_translation_matrixs
            # gt_rotate = targets_per_image.gt_rotation_matrixs
            # d_boxes, d_noise, d_t = self.prepare_diffusion_trans_concat(gt_trans)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            gt_lwhs = targets_per_image.gt_lwhs
            gt_P2s = targets_per_image.gt_P2s
            target["image_size_xyxy"] = image_size_xyxy.to(self.device) # 图像宽高 [4]
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device) # norm(cx,cy,w,h)
            target["3dboxes"] = gt_3dboxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device) # (xyxy)没有归一化的2dbox
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device) # [78,4] 78个2dbox所在图像的宽高宽高
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            target["lwhs"] = gt_lwhs.to(self.device)
            target["P2s"] = gt_P2s.to(self.device) 
            target["translation_matrix"] = targets_per_image.gt_translation_matrixs.to(self.device)
            target["rotation_matrix"] = targets_per_image.gt_rotation_matrixs.to(self.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)
    
    def inference1(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            pose_pred (Tensor): tensors of shape (batch_size, num_proposals, 6).
                The tensor predicts 6dof of the box (x,y,z,rx,ry,rz)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)# torch.Size([1, 500, 80])
            # self.num_classes = 80 arange(80) -> [0, 1, 2, ..., 79]
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
            # unsqueeze在第0维增加维度 torch.Size([1, 80])repeat对应第0维复制500次，第一维复制1次torch.Size([500, 80])flatten将第0维和第1维合并 torch.Size([40000])
            # for i, (scores_per_image, box_pred_per_image, image_size, pose_pred_per_image) in enumerate(zip( 
            #         scores, box_pred, image_sizes, pose_pred
            # )):
            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip( 
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)# result.image_size:(800, 1067)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False) # 返回前500个最大值和对应的索引 
                labels_per_image = labels[topk_indices]# torch.Size([500])按索引值取类别值
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4) # torch.Size([40000, 4]) view是改变tensor的形状，tensor的元素的总数是不变。-1表示自适应
                box_pred_per_image = box_pred_per_image[topk_indices] # torch.Size([500, 4])按索引值取坐标值
                # pose_pred_per_image = pose_pred_per_image.view(-1, 1, 6).repeat(1, self.num_classes, 1).view(-1, 6)
                # pose_pred_per_image = pose_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1: # flase
                    return box_pred_per_image, scores_per_image, labels_per_image, pose_pred_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)# 坐标值，类别分数值，类别值 batched_nms  非极大值抑制 返回torch.Size([106])
                    box_pred_per_image = box_pred_per_image[keep] # torch.Size([106, 4])
                    scores_per_image = scores_per_image[keep]# torch.Size([106])
                    labels_per_image = labels_per_image[keep]# torch.Size([106])
                    # pose_per_image = pose_pred_per_image[keep]# torch.Size([106, 6])

                result.pred_boxes = Boxes(box_pred_per_image)# Boxes class的形式也是(x1, y1, x2, y2)
                # result.pred_pose = Pose(pose_pred_perimage)# how to define Pose
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                # result.pred_poses = pose_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                # result.pred_pose = pose_pred_perimage
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def inference(self, box_cls, box_pred, image_sizes, pose_pred):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            pose_pred (Tensor): tensors of shape (batch_size, num_proposals, 6).
                The tensor predicts 6dof of the box (x,y,z,rx,ry,rz)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)# torch.Size([1, 500, 80])
            # self.num_classes = 80 arange(80) -> [0, 1, 2, ..., 79]
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
            # unsqueeze在第0维增加维度 torch.Size([1, 80])repeat对应第0维复制500次，第一维复制1次torch.Size([500, 80])flatten将第0维和第1维合并 torch.Size([40000])
            for i, (scores_per_image, box_pred_per_image, image_size, pose_pred_per_image) in enumerate(zip( 
                    scores, box_pred, image_sizes, pose_pred
            )):
                result = Instances(image_size)# result.image_size:(800, 1067)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False) # 返回前500个最大值和对应的索引 
                labels_per_image = labels[topk_indices]# torch.Size([500])按索引值取类别值
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4) # torch.Size([40000, 4]) view是改变tensor的形状，tensor的元素的总数是不变。-1表示自适应
                box_pred_per_image = box_pred_per_image[topk_indices] # torch.Size([500, 4])按索引值取坐标值
                pose_pred_per_image = pose_pred_per_image.view(-1, 1, 6).repeat(1, self.num_classes, 1).view(-1, 6)
                pose_pred_per_image = pose_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1: # flase
                    return box_pred_per_image, scores_per_image, labels_per_image, pose_pred_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)# 坐标值，类别分数值，类别值 batched_nms  非极大值抑制 返回torch.Size([106])
                    box_pred_per_image = box_pred_per_image[keep] # torch.Size([106, 4])
                    scores_per_image = scores_per_image[keep]# torch.Size([106])
                    labels_per_image = labels_per_image[keep]# torch.Size([106])
                    pose_per_image = pose_pred_per_image[keep]# torch.Size([106, 6])

                result.pred_boxes = Boxes(box_pred_per_image)# Boxes is tensor <class 'detectron2.structures.boxes.Boxes'>
                # result.pred_pose = Pose(pose_pred_perimage)# how to define Pose
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                result.pred_poses = pose_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                # result.pred_pose = pose_pred_perimage
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
    
    
def Data_handing(data):
    zoom_images_whwh = list()
    imgidx_batch = list()
    imgs_batch = list()
    gt_labels_batch = list()
    gt_2dboxes_batch = list()
    gt_3dboxes_batch = list()
    gt_lwhs_batch = list()
    gt_P2s_batch = list()
    gt_translation_matrixs_batch = list()
    gt_rotation_matrixs_batch = list()
    for iter_data in data:
        (
            imgId, #图片名称
            imgidx,
            images_whwh,# 缩放后的长宽
            # images_path, # 图像路径
            sweep_imgs,# 几何变换后的图像
            gt_labels,
            gt_2dboxes,
            gt_3dboxes,
            gt_lwhs, 
            gt_P2s,
            gt_translation_matrixs,
            gt_rotation_matrixs
        ) = iter_data[:11]
        zoom_images_whwh.append(images_whwh)
        imgidx_batch.append(imgidx)
        imgs_batch.append(sweep_imgs)
        gt_labels_batch.append(gt_labels)
        gt_2dboxes_batch.append(gt_2dboxes)
        gt_3dboxes_batch.append(gt_3dboxes)
        gt_lwhs_batch.append(gt_lwhs)
        gt_P2s_batch.append(gt_P2s)
        gt_translation_matrixs_batch.append(gt_translation_matrixs) # (numgts,3)
        gt_rotation_matrixs_batch.append(gt_rotation_matrixs) # (numgts,3,3)
    ret_list = [
        torch.stack(zoom_images_whwh),
        imgidx_batch,
        torch.stack(imgs_batch),
        gt_labels_batch,
        gt_2dboxes_batch,
        gt_3dboxes_batch,
        gt_lwhs_batch,
        gt_P2s_batch,
        gt_translation_matrixs_batch,
        gt_rotation_matrixs_batch
    ]
    return ret_list

def visualize_diffusion(images, x_boxes):
    x_boxes0 = x_boxes[0]  # 选择第一个索引，如果需要
    x_boxes_np = x_boxes0.cpu().detach().numpy() 
    image = images[0]  # 选择第一个索引，如果需要
    image_np = image.permute(1, 2, 0).cpu().detach().numpy()
    import matplotlib.pyplot as plt
    import numpy as np
    # Assuming image_np is a NumPy array with shape (1088, 1920, 3) and might have values outside the [0, 1] range
    # # Normalize the image data to [0, 1]
    image_np = np.clip(image_np, 0, 1)

    # Display the image
    plt.imshow(image_np, vmin=0, vmax=1)  # Ensure the image is within the valid range
    # x_boxes_np = x_boxes_np[:50]# 只显示前50个
    # Display boxes on the image
    for box in x_boxes_np:
        x_min, y_min, x_max, y_max = box
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none'))  # Changing edgecolor to 'r' for red rectangle
        plt.text(x_min, y_min, '矩形', color='red', fontsize=12, fontproperties='SimHei')  # Changing text color to red

    # Display the image with red boxes
    plt.show()
