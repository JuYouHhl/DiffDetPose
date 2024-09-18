# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import colorsys
import bisect
import math
import multiprocessing as mp
from collections import deque
import cv2
import torch
from detectron2.structures import Boxes
import numpy as np
from detectron2.structures import Instances
import pycocotools.mask as mask_util
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from detectron2.utils.colormap import random_color
from read_dataset.visual_utils import draw_box_3d, project_to_image, test_draw_3d_box_on_image
from .util import box_ops
from diffusiondet.loss import box3d, projection, xyzs_to_xys, matrix2angle

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
_KEYPOINT_THRESHOLD = 0.05
_SMALL_OBJECT_AREA_THRESH = 1000

class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")
    
class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

class Visualizer:
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions,sem_seg,panoptic_seg_predictions,dataset_dict}`
    that draw composite data in some pre-defined style.

    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according
    to different heuristics, as long as the results still look visually reasonable.

    To obtain a consistent style, you can implement custom drawing functions with the
    abovementioned primitive methods instead. If you need more customized visualization
    styles, you can process the data yourself following their format documented in
    tutorials (:doc:`/tutorials/models`, :doc:`/tutorials/datasets`). This class does not
    intend to satisfy everyone's preference on drawing styles.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    # TODO implement a fast, rasterized version using OpenCV

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        if metadata is None:
            metadata = MetadataCatalog.get("__nonexist__")
        self.metadata = metadata
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._instance_mode = instance_mode
        self.keypoint_threshold = _KEYPOINT_THRESHOLD

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None # pred
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, CLASSES)
        poses = predictions.pred_poses if predictions.has("pred_poses") else None
        lwhs = predictions.lwhs if predictions.has("lwhs") else None
        gt_P2s = predictions.gt_P2s if predictions.has("gt_P2s") else None
        
        gt_t = predictions.gt_t if predictions.has("gt_t") else None
        gt_r = predictions.gt_r if predictions.has("gt_r") else None
        # gt_pose = predictions.gt_pose if predictions.has("gt_pose") else None
        
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        
        
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else: # this
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            poses=poses,
            lwhs=lwhs,
            gt_P2s=gt_P2s,
            
            gt_t=gt_t,
            gt_r=gt_r,
            # gt_pose=gt_pose,
            
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
    
    # pose demo
    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        poses=None,
        lwhs=None,
        gt_P2s=None,
        
        gt_t=None,
        gt_r=None,
        # gt_pose=None,
        
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        # 生成3dbox(8,3)
        # posetest=torch.cat([gt_t, poses[:,3:]], dim=-1)
        
        trans_factor = torch.tensor([[62,42,200]]) # 31,21,100  carla  33,27,53
        rotate_factor = torch.tensor([[90,40,90,]]) # 90,40,90 carla  90,30,90
        posetest=poses*torch.cat((trans_factor,rotate_factor),dim=-1) # [gtnums,6]
        src_3dboxes_xyz = box3d(posetest, lwhs) # [gtnums,8,3]
        src_3dboxes = xyzs_to_xys(gt_P2s, src_3dboxes_xyz) # (_xy) [gtnums,8,2]
        if src_3dboxes is not None:
            src_3dboxes = self._convert_boxes(src_3dboxes) #
            num_instances = len(src_3dboxes)
        if boxes is not None:
            boxes = self._convert_boxes(boxes) #
        #     num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        # if boxes is not None and boxes.shape[1] == 5:
        #     return self.overlay_rotated_instances(
        #         boxes=boxes, labels=labels, assigned_colors=assigned_colors
        #     )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            src_3dboxes = [src_3dboxes[k] for k in sorted_idxs] if src_3dboxes is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None
            
        # color = assigned_colors[1]
        # self.draw_3dbox(src_3dboxes[1], edge_color=color)
        # self.draw_box(boxes[1], edge_color=color)
        
        for i in range(num_instances): # 4 for i in range(num_instances):
            if labels[i].split()[0] in ['traffic_cone',]:
                continue
            
            color = assigned_colors[i]
            # color = np.array([0,0.88,0.22]) 
            
            if src_3dboxes is not None:
                self.draw_3dbox(src_3dboxes[i], edge_color=color)
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)
            # labels = None
            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output
    
    def overlay_instances1(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output
    
    def _change_color_brightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return tuple(np.clip(modified_color, 0.0, 1.0))
    
    def draw_3dbox(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        
        face_idx = [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        linewidth = 2 #max(self._default_font_size / 4, 1)
        for ind_f in [3, 2, 1, 0]: 
            f = face_idx[ind_f] 
            for j in [0, 1, 2, 3]: 
                line = mpl.lines.Line2D(
                    [int(box_coord[f[j],0]), int(box_coord[f[(j+1)%4], 0])],
                    [int(box_coord[f[j],1]), int(box_coord[f[(j+1)%4], 1])],
                    linewidth=linewidth * self.output.scale, alpha=alpha,
                    linestyle=line_style, color=edge_color,)
                self.output.ax.add_line(line)
            if ind_f == 0:
                line = mpl.lines.Line2D(
                    [int(box_coord[f[0],0]), int(box_coord[f[2], 0])],
                    [int(box_coord[f[0],1]), int(box_coord[f[2], 1])],
                    linewidth=linewidth * self.output.scale, alpha=alpha,
                    linestyle=line_style, color=edge_color,)
                self.output.ax.add_line(line)
                line = mpl.lines.Line2D(
                    [int(box_coord[f[1],0]), int(box_coord[f[3], 0])],
                    [int(box_coord[f[1],1]), int(box_coord[f[3], 1])],
                    linewidth=linewidth * self.output.scale, alpha=alpha,
                    linestyle=line_style, color=edge_color,)
                self.output.ax.add_line(line)
      
        return self.output
    
    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output
    
    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output
    
    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.detach().numpy()
        else:
            return np.asarray(boxes)

    def _convert_masks(self, masks_or_polygons):
        """
        Convert different format of masks or polygons to a tuple of masks and polygons.

        Returns:
            list[GenericMask]:
        """

        m = masks_or_polygons
        if isinstance(m, PolygonMasks):
            m = m.polygons
        if isinstance(m, BitMasks):
            m = m.tensor.numpy()
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        ret = []
        for x in m:
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                ret.append(GenericMask(x, self.output.height, self.output.width))
        return ret

    def _draw_text_in_mask(self, binary_mask, text, color):
        """
        Find proper places to draw text given a binary mask.
        """
        # TODO sometimes drawn on wrong objects. the heuristics here can improve.
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
        if stats[1:, -1].size == 0:
            return
        largest_component_id = np.argmax(stats[1:, -1]) + 1

        # draw text on the largest component, as well as other very large components.
        for cid in range(1, _num_cc):
            if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                # median is more stable than centroid
                # center = centroids[largest_component_id]
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                self.draw_text(text, center, color=color)

    def _convert_keypoints(self, keypoints):
        if isinstance(keypoints, Keypoints):
            keypoints = keypoints.tensor
        keypoints = np.asarray(keypoints)
        return keypoints

    def get_output(self):
        """
        Returns:
            output (VisImage): the image output containing the visualizations added
            to the image.
        """
        return self.output
    
def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels
class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            # if self.input_format == "RGB":
            #     # whether the model expects BGR inputs or RGB
            #     original_image = original_image[:, :, ::-1]
            # height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            # inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([original_image])[0]
            return predictions

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(# MetadataCatalog.get(name) 从元数据集中获取元数据
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused" 
        )# self.metadata 为Metadata(evaluator_type='coco', image_root='datasets/coco/val2017', json_file='datasets/coco/annotations/instances_val2017.json', name='coco_2017_val', thing_classes=['person', 'bicycle', 'car', 'motorcycle',
        self.cpu_device = torch.device("cpu")#cpu设备
        self.instance_mode = instance_mode

        self.parallel = parallel # false
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)#创建预测器
        
        self.threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST  # workaround

    def run_on_image(self, per_image, img):#运行图像
        """
        Args:
            # image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            #     This is the format used by OpenCV.
            per_image: 真值
            img(np.ndarray): 以BGR读取图像

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        #per_image 缩放后的宽高，图像，标签，2d真值，3d真值，lwh真值，内参，平移和旋转
        predictions = self.predictor(per_image) # 调用DefaultPredictor.__call__(self, original_image)  'instances': Instances(num_instances=95, image_height=480, image_width=640, fields=[pred_boxes: Boxes(tensor([[3.0941e+02, 1.6541e+00, 6.2857e+02, 2.3873e+02],
        # Filter
        instances = predictions['instances']  # Instances(num_instances=95, image_heigh
        # new_instances = instances[instances.scores > self.threshold] # 取出分数大于阈值的实例0.5
        a_instances = instances[instances.scores > 0.5].to(self.cpu_device)
        # 通过预测的2dbox的位置来与真值的每个2dbox做giou，然后取giou最大的真值对应的lwh作为预测值的lwh
        pred_boxes = a_instances.pred_boxes.tensor # xyxy abs
        gt_boxes = per_image[5]#.cuda() #  xyxy abs
        # 匈牙利算法实现最优分配;匈牙利匹配算法用于执行最优匹配，使得每个预测与一个目标（或未匹配的情况）关联
        cost_ious = 1-box_ops.box_iou(gt_boxes, pred_boxes)[0] # [num_gts,num_dts]
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost_ious) # gtindex, dtindex
        mask = cost_ious[row, col] != 1 # 过滤iou为0的部分
        mask = mask.numpy()
        row_ind, col_ind = row[mask], col[mask]
        max_index = np.argsort(col_ind) 
        dt_index = col_ind[max_index]
        gt_index = row_ind[max_index]
        
        # gious = box_ops.generalized_box_iou(pred_boxes, gt_boxes) # 22*32 gious[N, M];N = len(pred_boxes) M = len(gt_boxes)
        # max_value, max_index = torch.max(gious,dim=-1) # [22] max_index.shape
        # index = max_index
        # dt_index = max_value>0
        # index = max_index[max_value>0] # giou>0
        
        new_instances = Instances(instances.image_size) 
        new_instances.pred_boxes = Boxes(a_instances.pred_boxes.tensor[dt_index]) #[dt_index]) 
        new_instances.scores = a_instances.scores[dt_index] #[dt_index]
        new_instances.pred_classes = a_instances.pred_classes[dt_index] #[dt_index]
        new_instances.pred_poses = a_instances.pred_poses[dt_index] #[dt_index]
        new_instances.to(self.cpu_device)
        
        new_instances.lwhs = per_image[-4][gt_index] # .cuda()
        new_instances.gt_P2s = per_image[-3][gt_index]      
        new_instances.gt_t = per_image[-2][gt_index] # [numgts,3]
        new_instances.gt_r = per_image[-1][gt_index]
        new_instances.gt_boxes = Boxes(per_image[5][gt_index]) 
        
        
        # new_instances.gt_r = matrix2angle(per_image[-1].cuda()[index]) # [numgts,3]
        # new_instances.gt_pose = torch.cat([gt_t, gt_r], dim=-1)
        predictions = {'instances': new_instances}
       
        img = img[:, :, ::-1]
        # img = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        # gt_3dboxes的可视化demo
        # demo_dir = "./demo-pre" 
        # image_path = os.path.join("datasets/rope3d/validation-image_2", per_image[0] + ".jpg") #
        # demo_name = os.path.basename(image_path)
        # image = cv2.imread(image_path)
        # gt_P2s = new_instances.gt_P2s # [23,3,4]
        # gt_lwh = new_instances.lwhs # [23,3]
        # pred_poses = predictions['instances'].get('pred_poses') # 取预测值 [23,6]
        # pre_3dboxes = box3d(pred_poses, gt_lwh)  # tensor
        # box_2d = project_to_image(pre_3dboxes[0].cpu().numpy(), gt_P2s[0].cpu().numpy()) 
        # image = draw_box_3d(image, box_2d, c=(0,0,255))
        # # for pre_3dbox, gt_P2 in zip(pre_3dboxes, gt_P2s):
        # #     box_2d = project_to_image(pre_3dbox.cpu().numpy(), gt_P2.cpu().numpy()) 
        # #     image = draw_box_3d(image, box_2d, c=(0,0,255))
        # cv2.imwrite(os.path.join(demo_dir, demo_name),image)
        
        visualizer = Visualizer(img, self.metadata, instance_mode=self.instance_mode) # 创建可视化器
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device) # 将实例转移到cpu设备
                vis_output = visualizer.draw_instance_predictions(predictions=instances)# 画出实例预测

        return predictions, vis_output
    
    def box3d(self, pose6dof, lwh):
        box_center = pose6dof[:3]
        rotation_matrix = self.angle2matrix(pose6dof[3:])
        l, w, h = lwh[0], lwh[1], lwh[2]
        # 3d框在目标物体坐标系下的中心坐标为[0,-h/2,0]，3d框的8个顶点在x、y、z轴上的坐标
        x_corners = torch.tensor([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], device=rotation_matrix.device)
        y_corners = torch.tensor([0,0,0,0,-h,-h,-h,-h], device=rotation_matrix.device)
        z_corners = torch.tensor([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], device=rotation_matrix.device)
        corners = torch.stack([x_corners, y_corners, z_corners],dim=1) # [8,3]
        corners_3d = torch.matmul(corners, rotation_matrix.t()) + box_center # [8,3]
        return corners_3d
    
    def angle2matrix(self, angle):
        # 将角度转换为弧度
        bz_r_pose_rad = angle * (math.pi / 180.0)

        # 计算欧拉角的余弦和正弦值
        cos_x = torch.cos(bz_r_pose_rad[0])
        sin_x = torch.sin(bz_r_pose_rad[0])
        cos_y = torch.cos(bz_r_pose_rad[1])
        sin_y = torch.sin(bz_r_pose_rad[1])
        cos_z = torch.cos(bz_r_pose_rad[2])
        sin_z = torch.sin(bz_r_pose_rad[2])

        # 构建旋转矩阵
        rotation_matrices = torch.zeros(angle.size(0), 3, dtype=torch.float32, device=angle.device)
        rotation_matrices[0, 0] = cos_y * cos_z
        rotation_matrices[0, 1] = cos_x * sin_z + sin_x * sin_y * cos_z
        rotation_matrices[0, 2] = sin_x * sin_z - cos_x * sin_y * cos_z
        rotation_matrices[1, 0] = -cos_y * sin_z
        rotation_matrices[1, 1] = cos_x * cos_z - sin_x * sin_y * sin_z
        rotation_matrices[1, 2] = sin_x * cos_z + cos_x * sin_y * sin_z
        rotation_matrices[2, 0] = sin_y
        rotation_matrices[2, 1] = -sin_x * cos_y
        rotation_matrices[2, 2] = cos_x * cos_y
        return rotation_matrices

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

