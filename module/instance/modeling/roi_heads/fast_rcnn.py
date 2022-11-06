import torch
import torch.nn as nn

from typing import Union, Dict, Tuple, List
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, cat, cross_entropy
from detectron2.modeling import FastRCNNOutputLayers
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.structures import Boxes, Instances

from module.instance.modeling.losses.repulsion_loss import RepGT_loss, RepBox_loss
from module.instance.layers.nms import generalized_batched_nms


def nms_generalized_fast_rcnn_inference(
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        nms_score_threshold: float,
        nms_type: str,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
        nms_score_threshold (float): Used in soft nms. The pred boxes / proposals with det score >=
            `nms_score_threshold` will be processed.
        nms_type (str): The type of nms to used. Options are ['normal', 'softnms-linear',
            'softnms-gaussian', 'cluster'].


    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        nms_generalized_fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape,
            score_thresh, nms_thresh, topk_per_image,
            nms_score_threshold, nms_type,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def nms_generalized_fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        nms_score_threshold: float,
        nms_type: str,
):
    """
    NMS generalized fast rcnn inference that leverage the generalized batch nms instead
    of batch nms.
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = generalized_batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh,
                                   nms_score_threshold, nms_type)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class RepulsionFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Repulsion FastRCNNOutputLayers, which support the loss type of repulsion loss
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            rep_gt_sigma: float = 0.9,
            rep_box_sigma: float = 0.1,
    ):
        super().__init__(input_shape=input_shape,
                         box2box_transform=box2box_transform,
                         num_classes=num_classes,
                         test_score_thresh=test_score_thresh,
                         test_nms_thresh=test_nms_thresh,
                         test_topk_per_image=test_topk_per_image,
                         cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
                         smooth_l1_beta=smooth_l1_beta,
                         box_reg_loss_type=box_reg_loss_type,
                         loss_weight=loss_weight)

        self.rep_gt_sigma = rep_gt_sigma
        self.rep_box_sigma = rep_box_sigma

    @classmethod
    def from_config(cls, cfg, input_shape):
        _config = super().from_config(cfg, input_shape)

        rep_gt_sigma = cfg.MODEL.REPULSION_LOSS.REP_GT_SIGMA
        rep_box_sigma = cfg.MODEL.REPULSION_LOSS.REP_BOX_SIGMA

        # Repulsion loss hyper parameters are added here
        _config.update({
            "loss_weight": {
                "loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
                "loss_rep_gt": cfg.MODEL.REPULSION_LOSS.REP_GT_LOSS_WEIGHT,
                "loss_rep_box": cfg.MODEL.REPULSION_LOSS.REP_BOX_LOSS_WEIGHT,
            },
            "rep_gt_sigma": rep_gt_sigma,
            "rep_box_sigma": rep_box_sigma,
        })
        return _config

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` and ``gt_box_inds`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
            repl_gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_rep_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            repl_gt_boxes = proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes),
            "loss_rep_gt": self.rep_gt_loss(proposal_boxes, repl_gt_boxes, proposal_deltas, gt_classes),
            "loss_rep_box": self.rep_box_loss(predictions, proposals),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def rep_gt_loss(self, proposal_boxes, repl_gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            repl_gt_boxes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        fg_pred_boxes = self.box2box_transform.apply_deltas(
            fg_pred_deltas, proposal_boxes[fg_inds]
        )

        # bad box will occur. e.g., x2 > x1
        try:
            loss_rep_gt = RepGT_loss(fg_pred_boxes, repl_gt_boxes[fg_inds], sigma=self.rep_gt_sigma, reduction='sum')
        except:
            loss_rep_gt = torch.zeros((1,), device=pred_deltas.device)

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_rep_gt / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def rep_box_loss(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` and ``gt_box_inds`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        # List[gt_box_inds]
        list_gt_box_inds = [p.gt_box_inds for p in proposals]
        list_gt_classes = [p.gt_classes for p in proposals]
        list_gt_boxes = [p.gt_boxes.tensor for p in proposals]
        list_proposal_boxes = [p.proposal_boxes.tensor for p in proposals]

        list_num_gt = []
        # assert the box indexes, gt classes, gt boxes has the same length
        for bidi, ci, bi, pbi in zip(list_gt_box_inds,
                                     list_gt_classes,
                                     list_gt_boxes,
                                     list_proposal_boxes):
            assert bidi.shape[0] == ci.shape[0] == bi.shape[0] == pbi.shape[0]
            list_num_gt.append(bidi.shape[0])

        # chunk the proposal_deltas into batch-wise style
        list_proposal_deltas = torch.split(proposal_deltas,
                                           split_size_or_sections=list_num_gt, dim=0)

        loss_rep_box = torch.zeros((1,), device=proposal_deltas.device)
        num_pair = 0.0
        # iterate over every image in the batch
        # TODO gt_boxes is not used here
        for gt_box_inds, gt_classes, gt_boxes, per_image_deltas, per_image_proposal_boxes in zip(
                list_gt_box_inds, list_gt_classes, list_gt_boxes, list_proposal_deltas, list_proposal_boxes
        ):
            # filter out the background and only operate on the foreground target
            box_dim = per_image_proposal_boxes.shape[1]  # 4 or 5
            # Regression loss is only computed for foreground proposals (those matched to a GT)
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            if per_image_deltas.shape[1] == box_dim:  # cls-agnostic regression
                fg_pred_deltas = per_image_deltas[fg_inds]
            else:
                fg_pred_deltas = per_image_deltas.view(-1, self.num_classes, box_dim)[
                    fg_inds, gt_classes[fg_inds]
                ]

            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, per_image_proposal_boxes[fg_inds]
            )

            fg_gt_box_inds = gt_box_inds[fg_inds]

            # find the uniques in fg_gt_box_inds
            unique_fg_gt_box_inds = torch.unique(fg_gt_box_inds)

            # if the number of unique is one, no loss will be performed
            if unique_fg_gt_box_inds.size(0) > 1:
                # iterative over each unique subset
                # note that we randomly sample one box in the set
                per_set_fg_pred_boxes = []
                for set_id in unique_fg_gt_box_inds:
                    _set_fg_pred_boxes = fg_pred_boxes[fg_gt_box_inds == set_id]
                    sample_idx = torch.randperm(_set_fg_pred_boxes.size(0), device=_set_fg_pred_boxes.device)[:1][0]
                    per_set_fg_pred_boxes.append(
                        # randomly sample one box in the set
                        _set_fg_pred_boxes[sample_idx]
                    )
                # cat the set pred boxes in one image to calculate rep_box_loss
                per_set_fg_pred_boxes = torch.stack(per_set_fg_pred_boxes, dim=0)
                # bad box will occur. e.g., x2 > x1
                try:
                    _loss_rep_box, _num_pair = RepBox_loss(per_set_fg_pred_boxes, sigma=self.rep_box_sigma,
                                                           reduction='sum')
                    # cumulate the loss and num
                    loss_rep_box += _loss_rep_box
                    num_pair += _num_pair
                except:
                    pass

        return loss_rep_box / max(num_pair, 1.0)  # return 0 if empty


class NMSGeneralizedFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
        NMS Generalized Two linear layers for predicting Fast R-CNN outputs:
        which leverages the generalized nms instead of batch nms
        1. proposal-to-detection box regression deltas
        2. classification scores
        """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            nms_score_threshold: float = 0.001,
            nms_type: str = "normal",
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
            nms_score_threshold (float): Used in soft nms. The pred boxes / proposals with det score >=
                `nms_score_threshold` will be processed.
            nms_type (str): The type of nms to used. Options are ['normal', 'softnms-linear',
                'softnms-gaussian', 'cluster'].
        """
        super().__init__(
            input_shape=input_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=test_score_thresh,
            test_nms_thresh=test_nms_thresh,
            test_topk_per_image=test_topk_per_image,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            loss_weight=loss_weight,
        )
        self.nms_score_threshold = nms_score_threshold
        self.nms_type = nms_type

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)

        ret.update({
            "nms_score_threshold": cfg.MODEL.NMS_SCORE_THRESHOLD,
            "nms_type": cfg.MODEL.NMS_TYPE,
        })

        return ret

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return nms_generalized_fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.nms_score_threshold,
            self.nms_type,
        )
