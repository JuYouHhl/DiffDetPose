# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
import os
import cv2
import random
import mmcv
import torchvision
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
# from nuscenes.utils.data_classes import Box

from PIL import Image,ImageEnhance
from torch.utils.data import Dataset
from read_dataset.visual_utils import test_draw_3d_box_on_image

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]
carlaCLASS = [
    'car',
    'pedestrian',
]

H = 1080
W = 1920
final_dim = (864, 1536)

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone', # traffic_cone
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

backbone_conf = {
    'x_bound': [0, 140.8, 0.8],
    'y_bound': [-70.4, 70.4, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [-1.5, 3.0, 180],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'height_net_conf':
    dict(in_channels=512, mid_channels=512)
}
ida_aug_conf = {
    'final_dim':
    final_dim,
    'H':
    H,
    'W':
    W,
    'bot_pct_lim': (0.0, 0.0),
}

__all__ = ["DiffusionDetDatasetMapper", ]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    # ResizeShortestEdge
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DiffusionDetDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DiffusionDet.

    The callable currently does the following:

    1. Read the image from "file_name"从 "file_name "中读取图像
    2. Applies geometric transforms to the image and annotation对图像和注释进行几何变换
    3. Find and applies suitable cropping to the image and annotation寻找并应用合适的剪裁到图像和注释上
    4. Prepare image and annotation to Tensors准备好图像和注释的张量
    ResizeShortestEdge数据增强
    1.找到短边，根据短边与提供的最短长度列表，计算缩放比例
    2.初步缩放，得到多组h, w
    3.Max_size判断：
     1）若max(h,w)<max_size ， 正常用
     2）若max(h,w)>max——size， 继续缩放--根据max(h,w)缩放到max_size这个比例来缩
    """
    
    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),# RandomCrop是随机裁剪，参数为裁剪的大小
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )
        self.class_names = CLASSES #carlaCLASSES
        self.img_format = cfg.INPUT.FORMAT # RGB
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        
        self.NuscMVDetDataset = NuscMVDetDataset(ida_aug_conf=self.ida_aug_conf,
            classes=self.class_names,
            is_train=True,
            use_cbgs=False,)
 
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("ann_infos", None)           
        
        cam_info = dataset_dict['cam_info'] # dic
        image_data_list= self.NuscMVDetDataset.get_image(cam_info)  # 只获取图像及其尺寸两个参数
        ret_list = list()
        ( 
            zoom_images_whwh,
            # images_path, # 图像路径
            sweep_imgs,
            imgId,
            imgidx
        ) = image_data_list[:4]
        if self.is_train:
            gt_labels,gt_2dboxes,gt_3dboxes,gt_lwhs,gt_P2s,gt_translation_matrixs,gt_rotation_matrixs = self.NuscMVDetDataset.get_gt(dataset_dict) # 
        # Temporary solution for test.
        else:
            gt_labels = sweep_imgs.new_zeros(0, )
            gt_2dboxes = sweep_imgs.new_zeros(0, )
            gt_3dboxes = sweep_imgs.new_zeros(0, )
            gt_lwhs = sweep_imgs.new_zeros(0, )
            gt_P2s = sweep_imgs.new_zeros(0, )
            gt_translation_matrixs = sweep_imgs.new_zeros(0, )
            gt_rotation_matrixs = sweep_imgs.new_zeros(0, )
            
        # # gt_3dboxes的可视化demo
        # demo_dir = "./demo-carla-gt" 
        # image_path = cam_info['filename']
        # demo_name = os.path.basename(image_path)
        # image = cv2.resize(cv2.imread(image_path),(1920,1088))
        # image = test_draw_3d_box_on_image(image, gt_3dboxes.numpy(), gt_P2s.numpy()[0])
        # cv2.imwrite(os.path.join(demo_dir, demo_name),image)
       
        ret_list = [
            imgId, #图片名称
            imgidx,
            zoom_images_whwh,  # 缩放后的长宽 [1,4]
            # images_path, # 图像路径
            sweep_imgs, # 几何变换后的图像 [1,1,3,1088,1920]
            gt_labels,  # 真值标签 [num_gtlabels]
            gt_2dboxes, # 真值2dbox[num_gtlabels,4,2]
            gt_3dboxes, # 真值3dbox[num_gtlabels,8,3]
            gt_lwhs,    # 真值lwh，用于生成pre3dbox在obj坐标系下的坐标[num_gtlabels,3]
            gt_P2s,     # 真值相机投影矩阵[num_gtlabels,3,4]，用于将pre3dbox投影至2d空间中
            gt_translation_matrixs, # 真值平移/旋转矩阵，用于与pre6dof做损失[num_gtlabels,3]
            gt_rotation_matrixs # [num_gtlabels,3]
        ]
            
        return ret_list

    
    
class NuscMVDetDataset(Dataset):
    def __init__(self,
                 ida_aug_conf,
                 classes,
                 is_train,
                 use_cbgs=False,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 return_depth=False,
                 sweep_idxes=list(),
                 key_idxes=list()):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
        """
        super().__init__()
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        
    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []
        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def get_image(self, cam_info):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert cam_info is not None
        sweep_imgs = list()
        zoom_images_whwh = list()
        images_path = list()
        imgs = list()
        imgId = list()
        imgId = cam_info['imgId']
        # PIL.Image: image mode=RGB
        img = Image.open(
            os.path.join(cam_info['filename'])) # size(1920,1080)
        
        img = torchvision.transforms.Resize((1088,1920))(img) 
        # img = img.resize((1088,1920), Image.Resampling.LANCZOS)
        zoom_img_W,zoom_img_H= img._size
        zoom_images_whwh.append(torch.tensor([zoom_img_W,zoom_img_H,zoom_img_W,zoom_img_H]))
        zoom_images_whwh = torch.stack(zoom_images_whwh)
        
        images_path.append(cam_info['filename'])
        
        if self.is_train and random.random() < 0.2:# 对图像色调，饱和度，对比度、亮度数据增强
            img = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5)(img)
                                
        img = mmcv.imnormalize(np.array(img)[:,:,:3], self.img_mean,
                                self.img_std, self.to_rgb)#：这个函数用于对图像进行归一化处理。它接受图像、图像均值 self.img_mean、图像标准差 self.img_std 和 self.to_rgb 参数
        img = torch.from_numpy(img).permute(2, 0, 1) # HWC->CHW
        imgs.append(img) 
        sweep_imgs.append(torch.stack(imgs))

        ret_list = [
            zoom_images_whwh,# 缩放后的长宽
            # images_path, # 图像路径
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),# 数据增强和归一化后的图像
            imgId,
            cam_info["imgidx"],
        ]
        return ret_list

    def get_gt1(self, info):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT labels.  真值标签
            Tensor: GT 2dboxes. 真值2dbox[lenth,4,2]
            Tensor: GT 3dboxes. 真值3dbox[lenth,8,3]
            Tensor: GT lwhs.    真值lwh，用于生成pre3dbox在obj坐标系下的坐标
            Tensor: GT P2s.     真值相机投影矩阵[lenth,3,4]，用于将pre3dbox投影至2d空间中
            Tensor: GT trans/rotate matrixs. 真值平移/旋转矩阵，用于与pre6dof做损失
        """
        gt_labels = list()
        gt_2dboxes = list()
        gt_3dboxes = list()
        gt_lwhs = list()
        gt_P2s = list()
        gt_translation_matrixs = list()
        gt_rotation_matrixs = list()
        for ann_info in info['ann_infos']:
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
            gt_2dbox = np.array(ann_info["box2d"]) 
            gt_2dboxes.append(gt_2dbox)
            gt_3dbox = np.array(ann_info["box3d"])
            gt_3dboxes.append(gt_3dbox)
            gt_lwh = np.array(ann_info["lwh"])
            gt_lwhs.append(gt_lwh)
            gt_P2 = np.array(ann_info["P2"])
            gt_P2s.append(gt_P2)
            gt_translation_matrix = np.array(ann_info["location"])
            gt_translation_matrixs.append(gt_translation_matrix)
            gt_rotation_matrix = np.array(ann_info["rotation"])
            gt_rotation_matrixs.append(gt_rotation_matrix)
            
            # list->array->tensor速度很慢，后期可将其优化为list->tensor
        return torch.tensor(gt_labels),torch.Tensor(gt_2dboxes),torch.Tensor(gt_3dboxes),torch.Tensor(gt_lwhs),torch.Tensor(gt_P2s),torch.Tensor(gt_translation_matrixs),torch.Tensor(gt_rotation_matrixs)

    def get_gt(self, info):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT labels.  真值标签
            Tensor: GT 2dboxes. 真值2dbox[lenth,4,2]
            Tensor: GT 3dboxes. 真值3dbox[lenth,8,3]
            Tensor: GT lwhs.    真值lwh，用于生成pre3dbox在obj坐标系下的坐标
            Tensor: GT P2s.     真值相机投影矩阵[lenth,3,4]，用于将pre3dbox投影至2d空间中
            Tensor: GT trans/rotate matrixs. 真值平移/旋转矩阵，用于与pre6dof做损失
        """
        gt_labels = list()
        gt_2dboxes = list()
        gt_3dboxes = list()
        gt_lwhs = list()
        gt_P2s = list()
        gt_translation_matrixs = list()
        gt_rotation_matrixs = list()
        for ann_info in info['ann_infos']:
            # gt_labels.append(self.classes.index(ann_info['category_name']))
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']])) # 0-9  +1
            gt_2dbox = np.array(ann_info["box2d"]) # abs xyxy
            gt_2dboxes.append(gt_2dbox)
            gt_3dbox = np.array(ann_info["box3d"])
            gt_3dboxes.append(gt_3dbox)
            gt_lwh = np.array(ann_info["lwh"])
            gt_lwhs.append(gt_lwh)
            gt_P2 = np.array(ann_info["P2"])
            gt_P2s.append(gt_P2)
            gt_translation_matrix = np.array(ann_info["location"])
            gt_translation_matrixs.append(gt_translation_matrix)
            gt_rotation_matrix = np.array(ann_info["rotation"]) # (3,3)
            gt_rotation_matrixs.append(gt_rotation_matrix)
            
        GT_labels=torch.tensor(np.array(gt_labels),dtype=torch.int64)
        GT_2dboxes=torch.Tensor(np.array(gt_2dboxes))
        GT_3dboxes=torch.Tensor(np.array(gt_3dboxes))
        GT_lwhs=torch.Tensor(np.array(gt_lwhs))
        GT_P2s=torch.Tensor(np.array(gt_P2s))
        GT_translation_matrixs=torch.Tensor(np.array(gt_translation_matrixs))
        GT_rotation_matrixs=torch.Tensor(np.array(gt_rotation_matrixs))
            # list->array->tensor速度很慢，后期可将其优化为list->tensor
        return GT_labels, GT_2dboxes, GT_3dboxes, GT_lwhs, GT_P2s, GT_translation_matrixs, GT_rotation_matrixs
    
