import copy
import os.path
from typing import List, Tuple, Union, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY, build_anchor_generator, RPN_HEAD_REGISTRY
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator import RPN
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from module.instance.modeling.proposal_generator.proposal_utils import nms_generalized_find_top_rpn_proposals


@PROPOSAL_GENERATOR_REGISTRY.register()
class NMSGeneralizedRPN(RPN):
    """
    GeneralizedRPN that uses the generalized batch nms for nms operation.
    """

    @configurable
    def __init__(
            self,
            *,
            in_features: List[str],
            head: nn.Module,
            anchor_generator: nn.Module,
            anchor_matcher: Matcher,
            box2box_transform: Box2BoxTransform,
            batch_size_per_image: int,
            positive_fraction: float,
            pre_nms_topk: Tuple[float, float],
            post_nms_topk: Tuple[float, float],
            nms_thresh: float = 0.7,
            min_box_size: float = 0.0,
            anchor_boundary_thresh: float = -1.0,
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            box_reg_loss_type: str = "smooth_l1",
            smooth_l1_beta: float = 0.0,
            nms_score_threshold: float = 0.001,
            nms_type: str = "normal",
    ):
        super().__init__(
            in_features=in_features,
            head=head,
            anchor_generator=anchor_generator,
            anchor_matcher=anchor_matcher,
            box2box_transform=box2box_transform,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            pre_nms_topk=pre_nms_topk,
            post_nms_topk=post_nms_topk,
            nms_thresh=nms_thresh,
            min_box_size=min_box_size,
            anchor_boundary_thresh=anchor_boundary_thresh,
            loss_weight=loss_weight,
            box_reg_loss_type=box_reg_loss_type,
            smooth_l1_beta=smooth_l1_beta
        )
        self.nms_score_threshold = nms_score_threshold
        self.nms_type = nms_type

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            "nms_score_threshold": cfg.MODEL.NMS_SCORE_THRESHOLD,
            "nms_type": cfg.MODEL.NMS_TYPE,
        })
        return ret

    def predict_proposals(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return nms_generalized_find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
                self.nms_score_threshold,
                self.nms_type,
            )


@PROPOSAL_GENERATOR_REGISTRY.register()
class GroupRPN(RPN):
    class SimpleCombination(nn.Module):
        # TODO, maybe als register this module
        def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4, num_classes: int = 1):
            """
            Args:
                in_channels (int): number of input feature channels. When using multiple
                    input features, they must have the same number of channels.
                num_anchors (int): number of anchors to predict for *each spatial position*
                    on the feature map. The total number of anchors for each
                    feature map will be `num_anchors * H * W`.
                box_dim (int): dimension of a box, which is also the number of box regression
                    predictions to make for each anchor. An axis aligned box has
                    box_dim=4, while a rotated box has box_dim=5.
            """
            super().__init__()
            # 3x3 conv for the hidden representation
            # original features + logit features + delta features
            conv_in_channels = in_channels + 2 * (num_anchors * num_classes + num_anchors * box_dim)
            # print("===> conv_in_channels: {}".format(conv_in_channels))
            # print("===> in_channels: {}".format(in_channels))
            # print("===> num_anchors: {}".format(num_anchors))
            # print("===> box_dim: {}".format(box_dim))
            self.conv = nn.Conv2d(conv_in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            # 1x1 conv for predicting objectness logits
            self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
            # 1x1 conv for predicting box2box transform deltas
            self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

            for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
                nn.init.normal_(l.weight, std=0.01)
                nn.init.constant_(l.bias, 0)

        def forward(
                self,
                features: List[torch.Tensor],
                logit_features1: List[torch.Tensor],
                delta_features1: List[torch.Tensor],
                logit_features2: List[torch.Tensor],
                delta_features2: List[torch.Tensor]
        ):
            """
            Args:
                features (list[Tensor]): list of feature maps
                logit_features1 (list[Tensor]): list of logits feature maps from original
                delta_features1 (list[Tensor]): list of deltas feature maps from original
                logit_features2 (list[Tensor]): list of logits feature maps from group
                delta_features2 (list[Tensor]): list of deltas feature maps from group

            Returns:
                list[Tensor]: A list of L elements.
                    Element i is a tensor of shape (N, A, Hi, Wi) representing
                    the predicted objectness logits for all anchors. A is the number of cell anchors.
                list[Tensor]: A list of L elements. Element i is a tensor of shape
                    (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                    to proposals.
            """
            pred_objectness_logits = []
            pred_anchor_deltas = []
            for x, lx1, dx1, lx2, dx2 in zip(features,
                                             logit_features1, delta_features1,
                                             logit_features2, delta_features2):
                t = F.relu(self.conv(cat([x, lx1, dx1, lx2, dx2], dim=1)))
                pred_objectness_logits.append(self.objectness_logits(t))
                pred_anchor_deltas.append(self.anchor_deltas(t))
            return pred_objectness_logits, pred_anchor_deltas

    @configurable
    def __init__(
            self,
            *,
            in_features: List[str],
            head: nn.Module,
            anchor_generator: nn.Module,
            anchor_matcher: Matcher,
            box2box_transform: Box2BoxTransform,
            batch_size_per_image: int,
            positive_fraction: float,
            pre_nms_topk: Tuple[float, float],
            post_nms_topk: Tuple[float, float],
            nms_thresh: float = 0.7,
            min_box_size: float = 0.0,
            anchor_boundary_thresh: float = -1.0,
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            box_reg_loss_type: str = "smooth_l1",
            smooth_l1_beta: float = 0.0,
            group_iou_threshold: float = 0.1,
            use_group_rpn_head: bool = False,
            use_multi_class_rpn_head: bool = False,
    ):
        """
        Args:
            group_iou_threshold: iou threshold (e.g., when >= group_iou_threshold) to combine multiple boxes
                in one image into a group
            use_group_rpn_head: whether to apply extrac rpn head for group proposal extractions. Note that, this will also
                trigger the fuse_rpn_head module that fuses the proposals from both original and the group rpn head.
        """
        super().__init__(
            in_features=in_features,
            head=head,
            anchor_generator=anchor_generator,
            anchor_matcher=anchor_matcher,
            box2box_transform=box2box_transform,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            pre_nms_topk=pre_nms_topk,
            post_nms_topk=post_nms_topk,
            nms_thresh=nms_thresh,
            min_box_size=min_box_size,
            anchor_boundary_thresh=anchor_boundary_thresh,
            loss_weight=loss_weight,
            box_reg_loss_type=box_reg_loss_type,
            smooth_l1_beta=smooth_l1_beta
        )
        self.group_iou_threshold = group_iou_threshold
        self.use_group_rpn_head = use_group_rpn_head
        self.use_multi_class_rpn_head = use_multi_class_rpn_head

        if self.use_multi_class_rpn_head:
            # if we use multi class rpn head
            # then the output can not be processed by standard RCNN head
            # thus we must enable group rpn
            self.use_group_rpn_head = True

        self.group_rpn_head = None
        self.fuse_rpn_head = None
        self.num_classes = 1
        self.num_anchors = None
        num_classes = self.num_classes
        if use_group_rpn_head:
            self.group_rpn_head = copy.deepcopy(self.rpn_head)
            in_channels = self.rpn_head.conv.in_channels

            # num_classes + 1
            if self.use_multi_class_rpn_head:
                self.num_classes = self.rpn_head.num_classes
                num_classes = self.num_classes + 1

            num_anchors = self.rpn_head.objectness_logits.out_channels // num_classes
            box_dim = self.anchor_generator.box_dim
            self.fuse_rpn_head = GroupRPN.SimpleCombination(in_channels=in_channels,
                                                            num_anchors=num_anchors,
                                                            box_dim=box_dim,
                                                            num_classes=num_classes)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            "group_iou_threshold": cfg.MODEL.GROUP_IOU_THRESHOLD,
            "use_group_rpn_head": cfg.MODEL.USE_GROUP_RPN_HEAD,
            "use_multi_class_rpn_head": cfg.MODEL.RPN.HEAD_NAME == 'MultiClassRPNHead',
        })
        return ret

    @torch.jit.unused
    def losses(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            gt_labels: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            gt_boxes: List[torch.Tensor],
            loss_tag: str = '',
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            loss_tag (str): Used to distinguish group loss and group pos neg in tensorboard.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai*L))
        prefix = '{}_'.format(loss_tag) if loss_tag != '' else ''

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/{}num_pos_anchors".format(prefix), num_pos_anchors / num_images)
        storage.put_scalar("rpn/{}num_neg_anchors".format(prefix), num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],  # (N, Hi*Wi*Ai*L)
            gt_labels[valid_mask].to(torch.float32),  # (N, Hi*Wi*Ai*L)
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        if prefix != '':
            # update the losses key
            losses = {k.replace('rpn', '{}rpn'.format(prefix)): v for k, v in losses.items()}
        return losses

    @torch.jit.unused
    @torch.no_grad()
    def multi_class_label_and_sample_anchors(
            self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; >= 1 positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        gt_classes = [x.gt_classes for x in gt_instances]
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        matched_gt_classes = []
        for image_size_i, gt_boxes_i, gt_classes_i in zip(image_sizes, gt_boxes, gt_classes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            gt_classes_i: ground-truth classes for i-th image, maybe have one or two label
            """

            # get number of label from gt_classes_i
            # 1 or 2
            num_label_for_gt_classes_i = gt_classes_i.size(-1)

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            # first fill the matched_gt_classes_i with background class
            # anchors size: (num_anchor, 4)
            matched_gt_classes_i = torch.zeros_like(anchors.tensor[:, :0 + num_label_for_gt_classes_i])
            matched_gt_classes_i = matched_gt_classes_i + self.num_classes + 1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
                matched_gt_classes_i = gt_classes_i[matched_idxs]

            # reshape the matched_gt_classes_i to ensure it has a unified number of dimension
            matched_gt_classes_i = torch.reshape(matched_gt_classes_i, [matched_gt_classes_i.size(0), -1])

            gt_labels.append(gt_labels_i)  # N,AHWL
            matched_gt_boxes.append(matched_gt_boxes_i)  # N,AHWL,4
            matched_gt_classes.append(matched_gt_classes_i)  # N,AHWL,1 or N,AHWL,2
        return gt_labels, matched_gt_boxes, matched_gt_classes

    @torch.jit.unused
    def multi_class_losses(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            gt_labels: List[torch.Tensor],
            gt_classes: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            gt_boxes: List[torch.Tensor],
            loss_tag: str = ''
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A, C) representing. C = num_class + 1
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`multi_class_label_and_sample_anchors`.
            gt_classes (list[Tensor]): Output of :meth:`multi_class_label_and_sample_anchors`. Note that for background
                class, its label is num_class + 1
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            loss_tag (str): Used to distinguish group loss and group pos neg in tensorboard.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        # let L = num_level
        # [(L*Hi*Wi*Ai)] => (N, L*Hi*Wi*Ai)
        gt_labels = torch.stack(gt_labels)
        prefix = '{}_'.format(loss_tag) if loss_tag != '' else ''

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("multi_class_rpn/{}num_pos_anchors".format(prefix), num_pos_anchors / num_images)
        storage.put_scalar("multi_class_rpn/{}num_neg_anchors".format(prefix), num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        # (N, sum(L*Hi*Wi*Ai)) => (N*sum(L*Hi*Wi*Ai))
        gt_labels = gt_labels.reshape([-1])
        # [(sum(L*Hi*Wi*Ai), 1 or 2)] => (N, sum(L*Hi*Wi*Ai), 1 or 2) => (N*sum(L*Hi*Wi*Ai), 1 or 2)
        gt_classes = torch.stack(gt_classes).reshape([-1, gt_classes[0].size(-1)])

        # compute multi-class cross-entropy loss
        valid_mask = gt_labels >= 0  # (N, sum(L*Hi*Wi*Ai))
        mixup = gt_classes.size(-1) > 1
        C = self.num_classes + 1

        # [(N, Hi*Wi*A, C)]_{l=1}^{L} => (N, L*Hi*Wi*A, C) => (N*L*Hi*Wi*A, C)
        # for _pred_objectness_logits in pred_objectness_logits:
        #     print("===> _pred_objectness_logits.size(): {}".format(_pred_objectness_logits.size()))
        pred_objectness_logits = cat(pred_objectness_logits, dim=1).reshape([-1, C])[valid_mask]
        # (N, L*Hi*Wi*Ai) => (N*L*Hi*Wi*Ai)
        gt_classes = gt_classes[valid_mask].long()

        # print("===> pred_objectness_logits.shape: {}".format(pred_objectness_logits.size()))
        # print("===> gt_classes.shape: {}".format(gt_classes.size()))
        # print("===> torch.unique(gt_classes): {}".format(torch.unique(gt_classes)))

        if not mixup:
            objectness_loss = F.cross_entropy(
                pred_objectness_logits,
                gt_classes[:, 0],
                reduction="sum",
            )
        else:
            objectness_loss_class1 = F.cross_entropy(
                pred_objectness_logits,
                gt_classes[:, 0],
                reduction="sum",
            )
            objectness_loss_class2 = F.cross_entropy(
                pred_objectness_logits,
                gt_classes[:, 1],
                reduction="sum",
            )
            objectness_loss = 0.5 * objectness_loss_class1 + 0.5 * objectness_loss_class2

        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        if prefix != '':
            # update the losses key
            losses = {k.replace('rpn', '{}_multi_class_rpn'.format(prefix)): v for k, v in losses.items()}
        return losses

    @staticmethod
    def _reshape_logits(_pred_objectness_logits, _multi_class=False, _num_class=None):
        if not _multi_class:
            # Transpose the Hi*Wi*A dimension to the middle:
            return [
                # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                score.permute(0, 2, 3, 1).flatten(1)
                for score in _pred_objectness_logits
            ]
        else:
            assert _num_class is not None
            # Transpose the Hi*Wi*A dimension to the middle:
            # while preserving the class dimension
            # denote C = num_class + 1
            ret = []
            for score in _pred_objectness_logits:
                N, AC, Hi, Wi = score.size()
                # print("===> N, AC, Hi, Wi: {}, {}, {}, {}".format(N, AC, Hi, Wi))
                C = _num_class
                A = AC // _num_class
                # print("===> A, C: {}, {}".format(A, C))

                # (N, A*C, Hi, Wi) -> (N, Hi, Wi, A*C) -> (N, Hi, Wi, A, C) -> (N, Hi*Wi*A, C)
                score = score.permute(0, 2, 3, 1).reshape([N, Hi, Wi, A, C]).reshape([N, Hi * Wi * A, C])
                ret.append(score)
            return ret

    def _reshape_deltas(self, _pred_anchor_deltas):
        return [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
            for x in _pred_anchor_deltas
        ]

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        # for _ in features:
        # print("===> _.size(): {}".format(_.size()))
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        reshaped_pred_objectness_logits = self._reshape_logits(pred_objectness_logits,
                                                               _multi_class=self.use_multi_class_rpn_head,
                                                               _num_class=self.num_classes + 1)
        reshaped_pred_anchor_deltas = self._reshape_deltas(pred_anchor_deltas)

        if not self.use_group_rpn_head:
            # grouped_pred_objectness_logits = pred_objectness_logits
            # grouped_pred_anchor_deltas = pred_anchor_deltas
            reshaped_grouped_pred_objectness_logits = reshaped_pred_objectness_logits
            reshaped_grouped_pred_anchor_deltas = reshaped_pred_anchor_deltas
            reshaped_fused_pred_objectness_logits = reshaped_pred_objectness_logits
            reshaped_fused_pred_anchor_deltas = reshaped_pred_anchor_deltas
        else:
            assert self.group_rpn_head is not None
            assert self.fuse_rpn_head is not None
            # group rpn head
            grouped_pred_objectness_logits, grouped_pred_anchor_deltas = self.group_rpn_head(features)
            reshaped_grouped_pred_objectness_logits = self._reshape_logits(grouped_pred_objectness_logits,
                                                                           _multi_class=self.use_multi_class_rpn_head,
                                                                           _num_class=self.num_classes + 1)
            reshaped_grouped_pred_anchor_deltas = self._reshape_deltas(grouped_pred_anchor_deltas)

            # fuse rpn head
            fused_pred_objectness_logits, fused_pred_anchor_deltas = self.fuse_rpn_head(
                features,
                pred_objectness_logits, pred_anchor_deltas,
                grouped_pred_objectness_logits, grouped_pred_anchor_deltas,
            )
            # fuse rpn always not using multi-class fpn
            reshaped_fused_pred_objectness_logits = self._reshape_logits(fused_pred_objectness_logits)
            reshaped_fused_pred_anchor_deltas = self._reshape_deltas(fused_pred_anchor_deltas)

        use_multi_class_rpn = self.use_multi_class_rpn_head

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"

            if not use_multi_class_rpn:
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                losses = self.losses(
                    anchors,
                    reshaped_pred_objectness_logits, gt_labels,
                    reshaped_pred_anchor_deltas, gt_boxes
                )
            else:
                gt_labels, gt_boxes, gt_classes = self.multi_class_label_and_sample_anchors(anchors, gt_instances)
                losses = self.multi_class_losses(
                    anchors,
                    reshaped_pred_objectness_logits, gt_labels, gt_classes,
                    reshaped_pred_anchor_deltas, gt_boxes
                )

            # (1)
            # perform group combination
            grouped_gt_instances = self.group_combination(gt_instances)
            if not use_multi_class_rpn:
                grouped_gt_labels, grouped_gt_boxes = self.label_and_sample_anchors(anchors, grouped_gt_instances)
                # perform group loss
                group_losses = self.losses(
                    anchors,
                    reshaped_grouped_pred_objectness_logits, grouped_gt_labels,
                    reshaped_grouped_pred_anchor_deltas, grouped_gt_boxes,
                    loss_tag='group',
                )
            else:
                grouped_gt_labels, grouped_gt_boxes, grouped_gt_classes = self.multi_class_label_and_sample_anchors(
                    anchors, grouped_gt_instances)
                # perform group loss
                group_losses = self.multi_class_losses(
                    anchors,
                    reshaped_grouped_pred_objectness_logits, grouped_gt_labels, grouped_gt_classes,
                    reshaped_grouped_pred_anchor_deltas, grouped_gt_boxes,
                    loss_tag='group',
                )
            losses.update(group_losses)

            # (2)
            # perform fusion between original and grouped features
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            if self.use_group_rpn_head:
                fuse_losses = self.losses(
                    anchors,
                    reshaped_fused_pred_objectness_logits, gt_labels,
                    reshaped_fused_pred_anchor_deltas, gt_boxes,
                    loss_tag='fuse',
                )
                losses.update(fuse_losses)

        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, reshaped_fused_pred_objectness_logits, reshaped_fused_pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    def group_combination(self, gt_instances: List[Instances]):
        """
        Given a list (batch) of instances, for every instances, we combine the boxes that
        IoU overlap >=  self.group_iou_threshold
        Args:
            gt_instances: the ground-truth instances for each image.

        Returns:
            grouped_gt_instances: the grouped boxes and classes for each image.
            # grouped_gt_boxes: the grouped ground-truth boxes for each image.
            # grouped_gt_classes: the grouped ground-truth classes for each image.
        """
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        img_sizes = [x.image_size for x in gt_instances]
        del gt_instances
        grouped_gt_instances = []

        for img_size, gt_classes_i, gt_boxes_i in zip(img_sizes, gt_classes, gt_boxes):
            # retrieve pairwise iou
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, gt_boxes_i)

            # retrieve indexes of boxes to group
            box_js, box_ks = [], []
            for j in range(match_quality_matrix.size(0)):
                for k in range(j, match_quality_matrix.size(0)):
                    if match_quality_matrix[j, k] >= self.group_iou_threshold:
                        box_js.append(j)
                        box_ks.append(k)

            # group boxes according to indexes
            # construct boxes first
            boxes_i1 = Boxes(gt_boxes_i.tensor[box_js, :])
            boxes_i2 = Boxes(gt_boxes_i.tensor[box_ks, :])
            grouped_boxes = self.union_boxes(boxes_i1, boxes_i2)

            # group classes according to indexes
            classes_i1 = gt_classes_i[box_js]
            classes_i2 = gt_classes_i[box_ks]
            grouped_classes = self.union_classes(classes_i1, classes_i2)

            # construct new instances
            grouped_gt_instances_i = Instances(image_size=img_size,
                                               gt_boxes=grouped_boxes,
                                               gt_classes=grouped_classes)

            grouped_gt_instances.append(grouped_gt_instances_i)

        return grouped_gt_instances

    @classmethod
    def union_boxes(cls, boxes1: Boxes, boxes2: Boxes):
        def _choose_min(_x, _y):
            return torch.where(_x < _y, _x, _y)

        def _choose_max(_x, _y):
            return torch.where(_x >= _y, _x, _y)

        assert len(boxes1) == len(boxes2)
        tensor1 = boxes1.tensor
        tensor2 = boxes2.tensor
        new_tensor = torch.zeros_like(boxes1.tensor)
        new_tensor[:, 0] = _choose_min(tensor1[:, 0], tensor2[:, 0])
        new_tensor[:, 1] = _choose_min(tensor1[:, 1], tensor2[:, 1])
        new_tensor[:, 2] = _choose_max(tensor1[:, 2], tensor2[:, 2])
        new_tensor[:, 3] = _choose_max(tensor1[:, 3], tensor2[:, 3])
        return Boxes(new_tensor)

    @classmethod
    def union_classes(cls, classes1, classes2):
        assert classes1.size() == classes2.size()
        return torch.stack([classes1, classes2], dim=1)


@RPN_HEAD_REGISTRY.register()
class MultiClassRPNHead(nn.Module):
    """
    Multi class RPN classification and regression heads.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, num_classes: int = 1, box_dim: int = 4):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            num_classes (int): number of classes of the dataset
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        self.num_classes = num_classes
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        # we plus one here for background class
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        box_dim = anchor_generator.box_dim
        assert (
                len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {"in_channels": in_channels,
                "num_anchors": num_anchors[0],
                "num_classes": num_classes,
                "box_dim": box_dim}

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A*num_classes, Hi, Wi) representing
                the predicted class logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas
