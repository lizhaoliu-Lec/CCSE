#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess, build_mask_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from torch import nn

from .head import DynamicHead, ReferenceDynamicHead
from .loss import SetCriterion, HungarianMatcher
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (nested_tensor_from_tensor_list)

__all__ = ["ReferenceSparseRCNN"]


@META_ARCH_REGISTRY.register()
class ReferenceSparseRCNN(nn.Module):
    """
    Implement ReferenceSparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ReferenceSparseRCNN.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_proposals = cfg.MODEL.ReferenceSparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.ReferenceSparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.ReferenceSparseRCNN.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        self.input_shape = self.backbone.output_shape()

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # additional prop feats and head for reference
        self.reference_init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.reference_head = ReferenceDynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Build Mask Head.
        self.mask_pooler = None
        self.mask_head = None
        self.mask_in_features = None
        if self.mask_on:
            # fmt: off
            in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
            pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            pooler_scales = tuple(1.0 / self.input_shape[k].stride for k in in_features)
            sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
            # fmt: on
            self.mask_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            ) if pooler_type else None
            in_channels = [self.input_shape[f].channels for f in in_features][0]

            if pooler_type:
                shape = ShapeSpec(
                    channels=in_channels, width=pooler_resolution, height=pooler_resolution
                )
            else:
                shape = {f: self.input_shape[f] for f in in_features}
            self.mask_head = build_mask_head(cfg=cfg, input_shape=shape)
            self.mask_in_features = in_features

        # Loss parameters:
        class_weight = cfg.MODEL.ReferenceSparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.ReferenceSparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.ReferenceSparseRCNN.L1_WEIGHT
        mask_weight = cfg.MODEL.ReferenceSparseRCNN.MASK_WEIGHT
        no_object_weight = cfg.MODEL.ReferenceSparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.ReferenceSparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.ReferenceSparseRCNN.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.mask_on:
            weight_dict["loss_mask"] = mask_weight
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        # for visualization
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        # for loss
        self.matcher = matcher

    def _calculate_mask_loss(self, images, features, output, targets):
        # (1) perform matching between the pred_boxes and gt_boxes to get the correspond gt_masks
        indices = self.matcher(output, targets)

        pred_boxes = output['pred_boxes']

        # (2) construct proposal for mask head
        proposals = []
        for n, image_size in enumerate(images.image_sizes):
            # print("===> indices[n][0]: {}".format(indices[n][0]))
            # print("===> indices[n][1]: {}".format(indices[n][1]))
            # print("===> indices[n][0].size(): {}".format(indices[n][0].size()))
            # print("===> indices[n][1].size(): {}".format(indices[n][1].size()))
            # print("===> pred_boxes.size(): {}".format(pred_boxes.size()))
            # print("===> len(targets[n]): {}".format(len(targets[n])))
            res = Instances(image_size=image_size)
            res.proposal_boxes = Boxes(pred_boxes[n, indices[n][0]].detach())
            # res.objectness_logits = ... TODO this may not be necessary for mask rcnn head
            res.gt_masks = targets[n]['gt_masks'][indices[n][1]]
            res.gt_classes = targets[n]['labels'][indices[n][1]]

            # print("===> len(res.proposal_boxes): {}".format(len(res.proposal_boxes)))
            # print("===> len(res.gt_masks): {}".format(len(res.gt_masks)))
            # print("===> len(res.gt_classes): {}".format(len(res.gt_classes)))

            proposals.append(res)

        # print("===> len(proposals): {}".format(len(proposals)))

        # (3) prepare mask in features
        if self.mask_pooler is not None:
            mask_in_features = [features[f] for f in self.mask_in_features]
            mask_in_features = self.mask_pooler(mask_in_features, [_.proposal_boxes for _ in proposals])
            # print("===> mask_in_features.size(): {}".format(mask_in_features.size()))
        else:
            mask_in_features = {f: features[f] for f in self.mask_in_features}

        # (4) calculate mask loss
        return self.mask_head(mask_in_features, proposals)

    def _calculate_detection_loss(self, batched_inputs, images, targets, output, outputs_class, outputs_coord,
                                  prefix=None):

        if self.deep_supervision:
            output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                     for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                results = self.inference(
                    box_cls, box_pred, images.image_sizes
                )
                self.visualize_training(batched_inputs, results, prefix=prefix)

        return loss_dict

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
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        reference_images, reference_images_whwh = self.preprocess_image(batched_inputs, prefix='reference')

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        if isinstance(reference_images, (list, torch.Tensor)):
            reference_images = nested_tensor_from_tensor_list(reference_images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        reference_src = self.backbone(reference_images.tensor)
        features = list()
        reference_features = list()
        for f in self.in_features:
            feature = src[f]
            reference_feature = reference_src[f]
            features.append(feature)
            reference_features.append(reference_feature)

        # Prepare Proposals.
        # (1) original proposal boxes
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
        # (2) reference proposal boxes
        reference_proposal_boxes = self.init_proposal_boxes.weight.clone()
        reference_proposal_boxes = box_cxcywh_to_xyxy(reference_proposal_boxes)
        reference_proposal_boxes = reference_proposal_boxes[None] * reference_images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord, reference_output_class, reference_output_coord = self.reference_head(
            features, proposal_boxes, self.init_proposal_features.weight,
            reference_features, reference_proposal_boxes, self.reference_init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        reference_output = {'pred_logits': reference_output_class[-1], 'pred_boxes': reference_output_coord[-1]}

        if self.training:
            # prepare targets
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            reference_gt_instances = [x["reference_instances"].to(self.device) for x in batched_inputs]
            reference_targets = self.prepare_targets(reference_gt_instances)

            # calculate loss for main stream
            loss_dict = self._calculate_detection_loss(batched_inputs, images,
                                                       targets, output,
                                                       outputs_class, outputs_coord)
            # calculate loss for reference
            reference_loss_dict = self._calculate_detection_loss(batched_inputs, reference_images,
                                                                 reference_targets, reference_output,
                                                                 reference_output_class, reference_output_coord,
                                                                 prefix='reference')
            # update the reference loss dict into regular loss dict for detectron2 engine to handle
            loss_dict.update({
                'reference_' + k: v for k, v in reference_loss_dict.items()
            })

            if self.mask_on:
                loss_dict.update(self._calculate_mask_loss(images, src, output, targets))
                loss_dict.update({
                    'reference_' + k: v for k, v in
                    self._calculate_mask_loss(reference_images, reference_src, reference_output,
                                              reference_targets).items()
                })

            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)

            # prepare mask in features
            if self.mask_pooler is not None:
                mask_in_features = [src[f] for f in self.mask_in_features]
                mask_in_features = self.mask_pooler(mask_in_features, [_.pred_boxes for _ in results])
                # print("===> mask_in_features.size(): {}".format(mask_in_features.size()))
            else:
                mask_in_features = {f: src[f] for f in self.mask_in_features}

            # after predicting the box
            # we perform mask prediction
            results = self.mask_head(mask_in_features, results)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results

    def visualize_training(self, batched_inputs, results, prefix=None):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
            prefix: a str to denoted whether it is a reference image
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        if prefix is not None:
            img = batched_inputs[image_index]["{}_image".format(prefix)]
        else:
            img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        if prefix is not None:
            v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["{}_instances".format(prefix)].gt_boxes)
        else:
            v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        # In case index error
        max_boxes = min(predicted_boxes.shape[0], max_boxes)

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        if prefix is not None:
            vis_name = f"{prefix} Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        else:
            vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)

            # add for mask
            if self.mask_on:
                target["gt_masks"] = targets_per_image.gt_masks

            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs, prefix=None):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image" if prefix is None else "{}_image".format(prefix)].to(self.device)) for x in
                  batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            if prefix is not None:
                h, w = bi["{}_image".format(prefix)].shape[-2:]
            else:
                h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
