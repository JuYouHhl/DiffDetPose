# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    # 'output/testhhl''/mnt/c/diffusionpose/1.5k_base'
    cfg.OUTPUT_DIR = 'output/testhhl' 
    # cfg.SOLVER.CHECKPOINT_PERIOD = 40
    cfg.MODEL.DiffusionDet = CN()
    # cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 2
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300 # 最终被yaml中的数值覆盖
    
    # RCNN Head 
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6
    cfg.MODEL.DiffusionDet.POSE_R = 3 # 
    cfg.MODEL.DiffusionDet.POSE_T = 3
    
    #POSE
    cfg.MODEL.DiffusionDet.MODEL_CLASS_NUM = 5
    cfg.MODEL.DiffusionDet.MODEL_NAME = ["tram_001.obj", "car_001.obj", "truck_001.obj", "van_001.obj", "pedestrian_001.obj"]
    cfg.MODEL.DiffusionDet.CLASSES = ['car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',]
    # cfg.MODEL.DiffusionDet.MODEL_ID_MAP = {"1":0, "2":1, "3":2, "4":3, "5":4 }
    cfg.MODEL.DiffusionDet.MODEL_NUM_SAMPLES = 6000
    cfg.MODEL.DiffusionDet.MODEL_FPS_NUM = 8

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0 # ce 10.0 
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0 # giou 2
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 6.0 # bbox
    cfg.MODEL.DiffusionDet.T_WEIGHT = 7.0
    cfg.MODEL.DiffusionDet.R_WEIGHT = 1.0
    cfg.MODEL.DiffusionDet.THD_weight = 5.0
    cfg.MODEL.DiffusionDet.TWODTGT_weight = 20.0
    cfg.MODEL.DiffusionDet.TWOSRC_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_3d_WEIGHT = 1.0
    cfg.MODEL.DiffusionDet.BBox_3d_WEIGHT = 3.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1


    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 50

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # DenseNet Backbone.
    cfg.MODEL.DENSENET = CN()
    cfg.MODEL.DENSENET.DEAPTH = 121
    cfg.MODEL.DENSENET.OUT_FEATURES = ('dens2', 'dens3', 'dens4', 'dens5')
    cfg.MODEL.DENSENET.NUM_CLASSES = 80
    cfg.MODEL.DENSENET.NAME = 'densenet121'
    cfg.MODEL.DENSENET.STEM_OUT_CHANNELS = 64
    
    # VoVNet Backbone.
    cfg.MODEL.VOVNET = CN()
    cfg.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
    cfg.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    # Options: FrozenBN, GN, "SyncBN", "BN"
    cfg.MODEL.VOVNET.NORM = "FrozenBN"
    cfg.MODEL.VOVNET.OUT_CHANNELS = 256
    cfg.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256


    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
