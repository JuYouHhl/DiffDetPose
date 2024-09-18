# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import mmcv
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from diffusiondet.predictor import VisualizationDemo
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from read_dataset.read_rope3d_dataset import generate_info_rope3d

# constants
WINDOW_NAME = "Rope3D Pose Estimation"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/diffdet.coco.res50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")#是否使用摄像头
    parser.add_argument("--video-input", help="Path to video file.")    #视频输入
    parser.add_argument(    #图片输入
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'"
    )
    parser.add_argument(    #输出的路径
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(    #置信度阈值
        "--confidence-threshold",
        type=float,
        default=0.5, # 0.5
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(# 其余没指定的参数
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
# 参数列表过长：ulimit -s 65536

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

if __name__ == "__main__":
    
    DatasetCatalog.register("rope3d_train", lambda : mmcv.load("data/rope3d/rope3d_3k_train.pkl"))
    MetadataCatalog.get("rope3d_train" )
    DatasetCatalog.register("rope3d_val", lambda : mmcv.load("data/rope3d/rope3d_3k_val.pkl"))
    MetadataCatalog.get("rope3d_val" )
    
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")# 日志记录
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST, mapper=mapper)

    if args.input:
        for per_data in tqdm.tqdm(data_loader.dataset, disable= args.output):
            
            img_path = ["validation-image_2", "training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
            for a in img_path:
                if os.path.exists(os.path.join("F:\Rope3D", a, per_data[0] + ".jpg")):
                    img_file = os.path.join("F:\Rope3D", a, per_data[0] + ".jpg")
                    break 
                
            img = read_image(img_file,  format="BGR") # 读取图片 img: img.shape = (480, 640, 3) img是numpy数组 
            start_time = time.time()# time.time()返回当前时间的时间戳
            predictions, visualized_output = demo.run_on_image(per_data, img) #demo.run_on_image(img)返回预测结果和可视化结果 demo.run_on_image(img)调用DefaultPredictor调用模型和BGR图像预测 -- visualizer画图
            logger.info(
                "{}: {} in {:.2f}s".format(# in {：.2f}s 表示小数点后保留两位
                    per_data[1],# 图片路径
                    "detected {} instances".format(len(predictions["instances"]))# len(predications) = 4 instances是预测结果的key，len(predictions["instances"])是预测结果的数量
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )# [04/06 19:27:42 detectron2]: ./datasets/coco/train2017/000000000009.jpg: detected 4 instances in 108785.20s

            if args.output:# 如果有输出路径
                if os.path.isdir(args.output):# 如果输出路径是一个文件夹
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))# args.output为输出路径 path为图片路径os.path.basename()返回文件名  
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)# 保存可视化结果
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)# 创建一个窗口namewindow cv2.WINDOW_NORMAL表示窗口大小可调整
                cv2.resizeWindow(WINDOW_NAME, 1920, 1088)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])# 显示图片 get_image()返回可视化结果的图片[:, :, ::-1]表示将BGR转为RGB
                # cv2.waitKey(0)
                if cv2.waitKey(0) == 27:# 等待按键，27为esc键
                    break  # esc to quit
    elif args.webcam:# 如果是摄像头
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:  # 如果是视频
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input) 
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
