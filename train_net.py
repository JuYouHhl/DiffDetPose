# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.utils.data as torchdata
from fvcore.nn.precise_bn import get_bn_modules
import mmcv
from detectron2.config.config import configurable
from detectron2.data.build import get_detection_dataset_dicts, trivial_batch_collator
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers.distributed_sampler import InferenceSampler
from detectron2.evaluation.testing import print_csv_format

import detectron2.utils.comm as comm
from detectron2.utils.logger import _log_api_usage, setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from diffusiondet.Evaluator import ROPEEvaluator

import torch.multiprocessing as mp

from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model
from detectron2.data import DatasetCatalog, MetadataCatalog

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from read_dataset.read_rope3d_dataset import generate_info_rope3d
from diffusiondet.Evaluator import inference_on_dataset

from detectron2.evaluation import (
    DatasetEvaluator,
    print_csv_format,
    verify_results,
)

class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to DiffusionDet. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2") #getLogger是logging模块的一个方法，用于获取一个logger对象
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())# cfgnode

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)# optimizer params是一个字典，包含了所有的参数
        data_loader = self.build_train_loader(cfg) # 数据集格式转换和数据加载器

        model = create_ddp_model(model, broadcast_buffers=False)# create ddp model分布式训练
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))# may_get_ema_checkpointer是一个函数，返回一个字典，包含了ema的参数
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER # 450000
        self.cfg = cfg
        self.machine_rank = cfg.machine_rank

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)# EMA是指数移动平均，是一种给予近期数据更高权重的平均方法
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif "rope3d"in dataset_name:
            return ROPEEvaluator()
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)# 接收Detectron2数据集格式的数据集dict， 并将其映射为DiffusionDet使用的格式
        # datapkl = mmcv.load(cfg.DATASETS.TRAIN[0]) #, dataset=datapkl
        return build_detection_train_loader(cfg, mapper=mapper)# build_detection_train_loader是一个函数，返回一个迭代器，每次迭代返回一个batch的数据
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = [] 
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True): # key: name of the parameter:'backbone.fpn_lateral2.weight', value: the parameter    
            if not value.requires_grad: # requires_grad=true继续下一个if
                continue
            # if 'pose' in key:
            #     value.requires_grad=False
            # else:
            #     value.requires_grad=True
            # Avoid duplicating parameters
            if value in memo: #  memo.add(value)
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR# 2.5e-05
            weight_decay = cfg.SOLVER.WEIGHT_DECAY# 0.0001
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER# ADAMW
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":# ADAMW是Adam的一种变体，主要是为了解决Adam在训练后期参数更新幅度较大的问题
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
            data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=mapper) #, dataset=datapkl
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),# record the time of each iteration
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook 
            hooks.LRScheduler(),# adjust the learning rate
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=40))
        return ret

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # detectron2 config
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.machine_rank = args.machine_rank
    
    # cfg.DATASETS.TRAIN = ("rope3d_train",) # 训练数据集名称，修改
    # cfg.DATASETS.TEST = ("rope3d_val",) # 训练数据集名称，修改
    
    cfg.freeze()#冻结默认.yaml里的参数
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    try:
        DatasetCatalog.register("rope3d_train", lambda : mmcv.load("data/rope3d/rope3d_all_train.pkl")) # data/carla/carla_train.pkl
        MetadataCatalog.get("rope3d_train" )
        DatasetCatalog.register("rope3d_val", lambda : mmcv.load("data/rope3d/rope3d_all_val.pkl")) 
        MetadataCatalog.get("rope3d_val" )
    except Exception as err:
        pass 
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    launch( 
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
