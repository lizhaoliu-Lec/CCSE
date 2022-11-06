#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.
# Adopted from https://github.com/Megvii-BaseDetection/cvpods, but transfer it into detectron2 style

import logging
import math
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from torch import nn, Tensor

from detectron2.layers import ShapeSpec, cat

from module.instance.common import Scale
from module.instance.layers import generalized_batched_nms
from module.instance.modeling.losses import iou_loss
from module.instance.modeling.box_regression import Shift2BoxTransform
from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils import comm

from module.instance.comm import all_reduce
from module.instance.modeling.shift_generator import build_shift_generator

logger = logging.getLogger(__name__)


def permute_all_cls_and_box_to_N_HWA_K_and_concat(
        box_cls, box_delta, box_center, num_classes=80
):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)
    return box_cls, box_delta, box_center


@META_ARCH_REGISTRY.register()
class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1708.02002).
    """

    @configurable
    def __init__(self,
                 *,
                 backbone,
                 head,
                 shift_generator,
                 shift2box_transform,
                 pixel_mean,
                 pixel_std,
                 num_classes,
                 head_in_features,
                 fpn_strides,
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0,
                 iou_loss_type="iou",
                 center_sampling_radius=0.0,
                 object_sizes_of_interest=([-1, 64],
                                           [64, 128],
                                           [128, 256],
                                           [256, 512],
                                           [512, float("inf")]),
                 test_score_thresh=0.05,
                 test_topk_candidates=1000,
                 test_nms_thresh=0.6,
                 max_detections_per_image=100,
                 test_nms_type='normal',
                 vis_period=0,
                 input_format='BGR'):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        if len(self.backbone.output_shape()) != len(self.head_in_features):
            logger.warning("[FCOS] Backbone produces unused features.")
        # Shifts
        self.shift_generator = shift_generator
        self.shift2box_transform = shift2box_transform

        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss_type = iou_loss_type
        self.center_sampling_radius = center_sampling_radius
        self.object_sizes_of_interest = object_sizes_of_interest
        # Inference parameters:
        self.test_score_threshold = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_threshold = test_nms_thresh
        self.test_nms_type = test_nms_type
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.FCOS.IN_FEATURES]
        head = FCOSHead(cfg, feature_shapes)
        shift_generator = build_shift_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "shift_generator": shift_generator,
            "shift2box_transform": Shift2BoxTransform(weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.FCOS.NUM_CLASSES,
            "head_in_features": cfg.MODEL.FCOS.IN_FEATURES,
            "fpn_strides": cfg.MODEL.FCOS.FPN_STRIDES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA,
            "iou_loss_type": cfg.MODEL.FCOS.IOU_LOSS_TYPE,
            "center_sampling_radius": cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS,
            "object_sizes_of_interest": cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.FCOS.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.FCOS.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "test_nms_type": cfg.MODEL.NMS_TYPE_TEST,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
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
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10)
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)

        if self.training:
            gt_classes, gt_shifts_reg_deltas, gt_centerness = self.get_ground_truth(
                shifts, gt_instances)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        box_cls, box_delta, box_center, shifts, images
                    )
                    self.visualize_training(batched_inputs, results)
            return self.losses(gt_classes, gt_shifts_reg_deltas, gt_centerness,
                               box_cls, box_delta, box_center)
        else:
            results = self.inference(box_cls, box_delta, box_center, shifts,
                                     images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_classes, gt_shifts_deltas, gt_centerness,
               pred_class_logits, pred_shift_deltas, pred_centerness):
        """
        Args:
            For `gt_classes`, `gt_shifts_deltas` and `gt_centerness` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_centerness`, see
                :meth:`FCOSHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_shift_deltas, pred_centerness = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat(
                pred_class_logits, pred_shift_deltas, pred_centerness,
                self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = all_reduce(num_foreground) / float(comm.get_world_size())
        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        num_targets = all_reduce(num_foreground_centerness) / float(comm.get_world_size())

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="sum",
        ) / max(1.0, num_targets)

        # centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / max(1, num_foreground)

        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
            gt_centerness (Tensor):
                An float tensor (0, 1) of shape (N, R) whose values in [0, 1]
                storing ground-truth centerness for each shift.

        """
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        for shifts_per_image, targets_per_image in zip(shifts, targets):
            object_sizes_of_interest = torch.cat([
                shifts_i.new_tensor(size).unsqueeze(0).expand(
                    shifts_i.size(0), -1) for shifts_i, size in zip(
                    shifts_per_image, self.object_sizes_of_interest)
            ], dim=0)

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            if self.center_sampling_radius > 0:
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                        torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            max_deltas = deltas.max(dim=-1).values
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                (max_deltas <= object_sizes_of_interest[None, :, 1])

            gt_positions_area = gt_boxes.area().unsqueeze(1).repeat(
                1, shifts_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = math.inf
            gt_positions_area[~is_cared_in_the_level] = math.inf

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

            # ground truth box regression
            gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes[gt_matched_idxs].tensor)

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Shifts with area inf are treated as background.
                gt_classes_i[positions_min_area == math.inf] = self.num_classes
            else:
                gt_classes_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes

            # ground truth centerness
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

        return torch.stack(gt_classes), torch.stack(
            gt_shifts_deltas), torch.stack(gt_centerness)

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        """
        Arguments:
            box_cls: Same as the output of :meth:`FCOSHead.forward`
            box_delta: Same as the output of :meth:`FCOSHead.forward`
            box_center: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_ctr_per_image = [
                box_ctr_per_level[img_idx] for box_ctr_per_level in box_center
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, box_ctr_per_image,
                shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_center, shifts,
                               image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(
                box_cls, box_delta, box_center, shifts):
            # (HxWxK,)
            # Since box_ctr_i depends on the sigmoided box_cls_i
            # do not make in palce on it, otherwise, the gradient will become unreachable
            # box_cls_i = box_cls_i.flatten().sigmoid_()
            box_cls_i = box_cls_i.flatten().sigmoid()

            # Keep top k top scoring indices only.
            num_topk = min(self.test_topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.test_score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            # do not take inplace sigmoid as mentioned aboe
            # box_ctr_i = box_ctr_i.flatten().sigmoid_()[shift_idxs]
            box_ctr_i = box_ctr_i.flatten().sigmoid()[shift_idxs]
            predicted_prob = torch.sqrt(predicted_prob * box_ctr_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = generalized_batched_nms(
            boxes_all, scores_all, class_idxs_all,
            self.test_nms_threshold, nms_type=self.test_nms_type
        )
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _inference_for_ms_test(self, batched_inputs):
        """
        function used for multiscale test, will be refactor in the future.
        The same input with `forward` function.
        """
        assert not self.training, "inference mode with training=True"
        assert len(batched_inputs) == 1, "inference image number > 1"
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)

        results = self.inference(box_cls, box_delta, box_center, shifts, images)
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results = detector_postprocess(results_per_image, height, width)
        return processed_results


class FCOSHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
            self,
            *,
            input_shape: List[ShapeSpec],
            num_classes,
            num_convs,
            num_shifts,
            fpn_strides,
            centerness_on_reg=False,
            norm_reg_targets=False,
            prior_prob=0.01):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels

        self.fpn_strides = fpn_strides
        self.centerness_on_reg = centerness_on_reg
        self.norm_reg_targets = norm_reg_targets

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_shifts * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_shifts * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.centerness = nn.Conv2d(in_channels,
                                    num_shifts * 1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score,
            self.bbox_pred, self.centerness
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_shifts = build_shift_generator(cfg, input_shape).num_cell_shifts
        assert (
                len(set(num_shifts)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_shifts = num_shifts[0]

        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.FCOS.NUM_CLASSES,
            "num_convs": cfg.MODEL.FCOS.NUM_CONVS,
            "prior_prob": cfg.MODEL.FCOS.PRIOR_PROB,
            "num_shifts": num_shifts,
            "fpn_strides": cfg.MODEL.FCOS.FPN_STRIDES,
            "centerness_on_reg": cfg.MODEL.FCOS.CENTERNESS_ON_REG,
            "norm_reg_targets": cfg.MODEL.FCOS.NORM_REG_TARGETS,
        }

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            centerness (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness at each spatial position.
        """
        logits = []
        bbox_reg = []
        centerness = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            if self.centerness_on_reg:
                centerness.append(self.centerness(bbox_subnet))
            else:
                centerness.append(self.centerness(cls_subnet))

            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[level])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness
