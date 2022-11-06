import inspect
from typing import List, Optional

import torch
import numpy as np
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import StandardROIHeads, ROI_HEADS_REGISTRY, build_box_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.structures import Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from torch import nn

from module.instance.modeling.matcher import Top2Matcher
from module.instance.modeling.roi_heads.fast_rcnn import RepulsionFastRCNNOutputLayers, \
    NMSGeneralizedFastRCNNOutputLayers
from module.instance.modeling.roi_heads.maskiou_head import build_maskiou_head, maskiou_loss, mask_iou_inference, \
    mask_rcnn_loss_with_maskiou


@ROI_HEADS_REGISTRY.register()
class RepulsionROIHeads(StandardROIHeads):
    """
    RepulsionROIHeads that supports batch maintaining after roi pool and top 2 matcher to mine the GT_{repl}.
    """

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # Use RepulsionFastRCNNOutputLayers instead of FastRCNNOutputLayers
        box_predictor = RepulsionFastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        # Use the Top2Matcher instead of the Matcher
        ret.update({
            "proposal_matcher": Top2Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
        })
        return ret

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                - gt_box_inds - index in targets of each box in gt_boxes
                - gt_rep_boxes - repulsion target for each proposal

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            has_repel_target = len(targets_per_image) > 1

            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels, second_matched_idxs, second_matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            if has_repel_target:
                assert second_matched_idxs is not None, 'second_matched_idxs is None when repulsion target available'
                assert second_matched_labels is not None, 'second_matched_labels is None when repulsion target available'

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            sampled_targets = matched_idxs[sampled_idxs]

            # also store the index of the target box - useful for creating subsets later
            # e.g., calculating RepBox_loss
            proposals_per_image.set("gt_box_inds", sampled_targets)

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            if has_repel_target:
                # Since the second best are bond with the best, we do not need to
                # acquire the sampled_idxs for _sample_proposals
                sampled_repels = second_matched_idxs[sampled_idxs]
                assert "gt_boxes" in targets_per_image.get_fields()
                proposals_per_image.set("gt_rep_boxes", targets_per_image.gt_boxes[sampled_repels])

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt


@ROI_HEADS_REGISTRY.register()
class NMSGeneralizedROIHeads(StandardROIHeads):
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)

        if "box_predictor" in ret and "box_head" in ret:
            box_head = ret["box_head"]
            ret.update({
                "box_predictor": NMSGeneralizedFastRCNNOutputLayers(cfg, box_head.output_shape)
            })

        return ret


@ROI_HEADS_REGISTRY.register()
class MaskIOUROIHeads(StandardROIHeads):
    """
    Build upon StandardROIHeads, with mask iou head added
    """

    @configurable
    def __init__(
            self,
            *,
            box_in_features: List[str],
            box_pooler: ROIPooler,
            box_head: nn.Module,
            box_predictor: nn.Module,
            mask_in_features: Optional[List[str]] = None,
            mask_pooler: Optional[ROIPooler] = None,
            mask_head: Optional[nn.Module] = None,
            keypoint_in_features: Optional[List[str]] = None,
            keypoint_pooler: Optional[ROIPooler] = None,
            keypoint_head: Optional[nn.Module] = None,
            train_on_pred_boxes: bool = False,
            maskiou_head: Optional[nn.Module] = None,
            maskiou_weight: Optional[float] = None,
            vis_period: int = 0,
            **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
            maskiou_head (nn.Module): transform features from mask predictions and roi to predict mask iou
            maskiou_weight (float): optimization strength on maskiou loss
            vis_period (int): the period to visualize training statics

        """
        super().__init__(box_in_features=box_in_features,
                         box_pooler=box_pooler,
                         box_head=box_head,
                         box_predictor=box_predictor,
                         mask_in_features=mask_in_features,
                         mask_pooler=mask_pooler,
                         mask_head=mask_head,
                         keypoint_in_features=keypoint_in_features,
                         keypoint_pooler=keypoint_pooler,
                         keypoint_head=keypoint_head,
                         train_on_pred_boxes=train_on_pred_boxes,
                         **kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        self.maskiou_on = maskiou_head is not None
        self.maskiou_head = maskiou_head
        self.maskiou_weight = maskiou_weight

        self.vis_period = vis_period

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if inspect.ismethod(cls._init_maskiou_head):
            ret.update(cls._init_maskiou_head(cfg))
        # update the vis period
        ret.update({
            "vis_period": cfg.VIS_PERIOD,
        })
        return ret

    @classmethod
    def _init_maskiou_head(cls, cfg):
        # only utilize mask iou head when both maskiou_on and mask_on are set
        if not (cfg.MODEL.MASKIOU_ON and cfg.MODEL.MASK_ON):

            return {}

        return {
            "maskiou_head": build_maskiou_head(cfg),
            "maskiou_weight": cfg.MODEL.MASKIOU_LOSS_WEIGHT,
        }

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            if self.maskiou_on:
                loss, mask_features, selected_mask, labels, maskiou_targets = self._forward_mask(features,
                                                                                                 proposals)
                losses.update(loss)
                losses.update(self._forward_maskiou(mask_features, proposals, selected_mask, labels, maskiou_targets))
            else:
                losses.update(self._forward_mask(features, proposals))

            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        if self.maskiou_on:
            instances, mask_features = self._forward_mask(features, instances)
            instances = self._forward_maskiou(mask_features, instances)
        else:
            instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}

        if self.training:
            # The loss is only defined on positive proposals.
            mask_logits = self.mask_head.layers(features)
            if self.maskiou_on:
                loss, selected_mask, labels, maskiou_targets = mask_rcnn_loss_with_maskiou(mask_logits, instances,
                                                                                           self.maskiou_on,
                                                                                           self.vis_period)
                return {"loss_mask": loss}, features, selected_mask, labels, maskiou_targets
            else:
                return {
                    "loss_mask": mask_rcnn_loss_with_maskiou(mask_logits, instances, self.maskiou_on, self.vis_period)}

        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head.layers(features)
            mask_rcnn_inference(mask_logits, instances)

            if self.maskiou_on:
                return instances, mask_features
            else:
                return instances

    def _forward_maskiou(self, mask_features, instances, selected_mask=None, labels=None, maskiou_targets=None):
        """
        Forward logic of the mask iou prediction branch.

        Args:
            mask_features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, calibrate instances' scores.
        """
        if not self.maskiou_on:
            return {} if self.training else instances

        if self.training:
            pred_maskiou = self.maskiou_head(mask_features, selected_mask)
            return {"loss_maskiou": maskiou_loss(labels, pred_maskiou, maskiou_targets, self.maskiou_weight)}

        else:
            pred_maskiou = self.maskiou_head(mask_features, torch.cat([i.pred_masks for i in instances], 0))
            mask_iou_inference(instances, pred_maskiou)
            return instances
