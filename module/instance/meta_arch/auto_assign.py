#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.
# Adopted from https://github.com/Megvii-BaseDetection/AutoAssign, but transfer it into detectron2 style

import logging
import math
from typing import List
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
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils import comm
from module.instance.comm import all_reduce

from module.instance.modeling.shift_generator import build_shift_generator

logger = logging.getLogger(__name__)


def positive_bag_loss(logits, mask, gaussian_probs):
    # bag_prob = Mean-max(logits)
    weight = (3 * logits).exp() * gaussian_probs * mask
    w = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
    bag_prob = (w * logits).sum(dim=1)
    return F.binary_cross_entropy(bag_prob,
                                  torch.ones_like(bag_prob),
                                  reduction='none')


def negative_bag_loss(logits, gamma):
    return logits ** gamma * F.binary_cross_entropy(
        logits, torch.zeros_like(logits), reduction='none')


def normal_distribution(x, mu=0, sigma=1):
    return (-(x - mu) ** 2 / (2 * sigma ** 2)).exp()


def normalize(x):
    return (x - x.min() + 1e-12) / (x.max() - x.min() + 1e-12)


@META_ARCH_REGISTRY.register()
class AutoAssign(nn.Module):
    """
    Implement AutoAssign (https://arxiv.org/abs/2007.03496).
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
                 iou_loss_type="giou",
                 reg_weight=5.0,
                 test_score_thresh=0.05,
                 test_topk_candidates=1000,
                 test_nms_thresh=0.6,
                 max_detections_per_image=100,
                 test_nms_type='normal',
                 vis_period=0,
                 input_format='BGR'):
        super(AutoAssign, self).__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        if len(self.backbone.output_shape()) != len(self.head_in_features):
            logger.warning("[AutoAssign] Backbone produces unused features.")
        # Shifts
        self.shift_generator = shift_generator
        self.shift2box_transform = shift2box_transform

        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weight = reg_weight
        self.mu = nn.Parameter(torch.zeros(num_classes, 2))
        self.sigma = nn.Parameter(torch.ones(num_classes, 2))
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
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.AUTO_ASSIGN.IN_FEATURES]
        head = AutoAssignHead(cfg, feature_shapes)
        shift_generator = build_shift_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "shift_generator": shift_generator,
            "shift2box_transform": Shift2BoxTransform(weights=cfg.MODEL.AUTO_ASSIGN.BBOX_REG_WEIGHTS),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.AUTO_ASSIGN.NUM_CLASSES,
            "head_in_features": cfg.MODEL.AUTO_ASSIGN.IN_FEATURES,
            "fpn_strides": cfg.MODEL.AUTO_ASSIGN.FPN_STRIDES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.AUTO_ASSIGN.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.AUTO_ASSIGN.FOCAL_LOSS_GAMMA,
            "iou_loss_type": cfg.MODEL.AUTO_ASSIGN.IOU_LOSS_TYPE,
            "reg_weight": cfg.MODEL.AUTO_ASSIGN.REG_WEIGHT,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.AUTO_ASSIGN.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.AUTO_ASSIGN.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.AUTO_ASSIGN.NMS_THRESH_TEST,
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
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10)
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)

        if self.training:
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        box_cls, box_delta, box_center, shifts, images
                    )
                    self.visualize_training(batched_inputs, results)
            return self.losses(shifts, gt_instances, box_cls, box_delta,
                               box_center)
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

    def losses(self, shifts, gt_instances, box_cls, box_delta, box_center):
        box_cls_flattened = [
            permute_to_N_HWA_K(x, self.num_classes) for x in box_cls
        ]
        box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
        pred_class_logits = cat(box_cls_flattened, dim=1)
        pred_shift_deltas = cat(box_delta_flattened, dim=1)
        pred_obj_logits = cat(box_center_flattened, dim=1)

        pred_class_probs = pred_class_logits.sigmoid()
        pred_obj_probs = pred_obj_logits.sigmoid()
        pred_box_probs = []
        num_foreground = pred_class_logits.new_zeros(1)
        num_background = pred_class_logits.new_zeros(1)
        positive_losses = []
        gaussian_norm_losses = []

        for shifts_per_image, gt_instances_per_image, \
            pred_class_probs_per_image, pred_shift_deltas_per_image, \
            pred_obj_probs_per_image in zip(shifts, gt_instances, pred_class_probs, pred_shift_deltas, pred_obj_probs):
            locations = torch.cat(shifts_per_image, dim=0)
            labels = gt_instances_per_image.gt_classes
            gt_boxes = gt_instances_per_image.gt_boxes

            target_shift_deltas = self.shift2box_transform.get_deltas(
                locations, gt_boxes.tensor.unsqueeze(1))
            is_in_boxes = target_shift_deltas.min(dim=-1).values > 0

            foreground_idxs = torch.nonzero(is_in_boxes, as_tuple=True)

            with torch.no_grad():
                # predicted_boxes_per_image: a_{j}^{loc}, shape: [j, 4]
                predicted_boxes_per_image = self.shift2box_transform.apply_deltas(
                    pred_shift_deltas_per_image, locations)
                # gt_pred_iou: IoU_{ij}^{loc}, shape: [i, j]
                gt_pred_iou = pairwise_iou(
                    gt_boxes, Boxes(predicted_boxes_per_image)).max(
                    dim=0, keepdim=True).values.repeat(
                    len(gt_instances_per_image), 1)

                # pred_box_prob_per_image: P{a_{j} \in A_{+}}, shape: [j, c]
                pred_box_prob_per_image = torch.zeros_like(
                    pred_class_probs_per_image)
                box_prob = 1 / (1 - gt_pred_iou[foreground_idxs]).clamp_(1e-12)
                for i in range(len(gt_instances_per_image)):
                    idxs = foreground_idxs[0] == i
                    if idxs.sum() > 0:
                        box_prob[idxs] = normalize(box_prob[idxs])
                pred_box_prob_per_image[foreground_idxs[1],
                                        labels[foreground_idxs[0]]] = box_prob
                pred_box_probs.append(pred_box_prob_per_image)

            normal_probs = []
            for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                gt_shift_deltas = self.shift2box_transform.get_deltas(
                    shifts_i, gt_boxes.tensor.unsqueeze(1))
                distances = (gt_shift_deltas[..., :2] - gt_shift_deltas[..., 2:]) / 2
                normal_probs.append(
                    normal_distribution(distances / stride,
                                        self.mu[labels].unsqueeze(1),
                                        self.sigma[labels].unsqueeze(1)))
            normal_probs = torch.cat(normal_probs, dim=1).prod(dim=-1)

            composed_cls_prob = pred_class_probs_per_image[:, labels] * pred_obj_probs_per_image

            # matched_gt_shift_deltas: P_{ij}^{loc}
            loss_box_reg = iou_loss(pred_shift_deltas_per_image.unsqueeze(0),
                                    target_shift_deltas,
                                    box_mode="ltrb",
                                    loss_type=self.iou_loss_type,
                                    reduction="none") * self.reg_weight
            pred_reg_probs = (-loss_box_reg).exp()

            # positive_losses: { -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) }
            positive_losses.append(
                positive_bag_loss(composed_cls_prob.permute(1, 0) * pred_reg_probs,
                                  is_in_boxes.float(), normal_probs))

            num_foreground += len(gt_instances_per_image)
            num_background += normal_probs[foreground_idxs].sum().item()

            gaussian_norm_losses.append(
                len(gt_instances_per_image) / normal_probs[foreground_idxs].sum().clamp_(1e-12))

        num_foreground = all_reduce(num_foreground) / float(comm.get_world_size())
        num_background = all_reduce(num_background) / float(comm.get_world_size())

        # positive_loss: \sum_{i}{ -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) } / ||B||
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_foreground)

        # pred_box_probs: P{a_{j} \in A_{+}}
        pred_box_probs = torch.stack(pred_box_probs, dim=0)
        # negative_loss: \sum_{j}{ FL( (1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}) ) } / n||B||
        negative_loss = negative_bag_loss(
            pred_class_probs * pred_obj_probs * (1 - pred_box_probs),
            self.focal_loss_gamma).sum() / max(1, num_background)

        loss_pos = positive_loss * self.focal_loss_alpha
        loss_neg = negative_loss * (1 - self.focal_loss_alpha)
        loss_norm = torch.stack(gaussian_norm_losses).mean() * (1 - self.focal_loss_alpha)

        return {
            "loss_pos": loss_pos,
            "loss_neg": loss_neg,
            "loss_norm": loss_norm,
        }

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        """
        Arguments:
            box_cls: Same as the output of :meth:`AutoAssignHead.forward`
            box_delta: Same as the output of :meth:`AutoAssignHead.forward`
            box_center: Same as the output of :meth:`AutoAssignHead.forward`
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
            box_cls_i = (box_cls_i.sigmoid() * box_ctr_i.sigmoid()).flatten()

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

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class AutoAssignHead(nn.Module):
    """
    The head used in AutoAssign for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(self,
                 *,
                 input_shape: List[ShapeSpec],
                 num_classes,
                 num_convs,
                 num_shifts,
                 fpn_strides,
                 norm_reg_targets=True,
                 prior_prob=0.02):
        super(AutoAssignHead, self).__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        self.fpn_strides = fpn_strides
        self.norm_reg_targets = norm_reg_targets
        # fmt: on
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
        self.obj_score = nn.Conv2d(in_channels,
                                   num_shifts * 1,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score,
            self.bbox_pred, self.obj_score
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
        torch.nn.init.constant_(self.bbox_pred.bias, 4.0)

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
            "num_classes": cfg.MODEL.AUTO_ASSIGN.NUM_CLASSES,
            "num_convs": cfg.MODEL.AUTO_ASSIGN.NUM_CONVS,
            "prior_prob": cfg.MODEL.AUTO_ASSIGN.PRIOR_PROB,
            "num_shifts": num_shifts,
            "fpn_strides": cfg.MODEL.AUTO_ASSIGN.FPN_STRIDES,
            "norm_reg_targets": cfg.MODEL.AUTO_ASSIGN.NORM_REG_TARGETS,
        }

    def forward(self, features):
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
        """
        logits = []
        bbox_reg = []
        obj_logits = []
        for feature, stride, scale in zip(features, self.fpn_strides,
                                          self.scales):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            obj_logits.append(self.obj_score(bbox_subnet))

            bbox_pred = scale(self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * stride)
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, obj_logits
