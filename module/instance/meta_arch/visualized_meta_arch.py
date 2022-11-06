from typing import Tuple, Dict

import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, RetinaNet, detector_postprocess
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.utils.events import get_event_storage
from torch import Tensor

from module.instance.meta_arch._utils import visualize_featuremaps


@META_ARCH_REGISTRY.register()
class VisualizedRetinaNet(RetinaNet):
    """
    Implement the visualized version of RetinaNet.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            head,
            head_in_features,
            anchor_generator,
            box2box_transform,
            anchor_matcher,
            num_classes,
            focal_loss_alpha=0.25,
            focal_loss_gamma=2.0,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
            test_score_thresh=0.05,
            test_topk_candidates=1000,
            test_nms_thresh=0.5,
            max_detections_per_image=100,
            pixel_mean,
            pixel_std,
            vis_period=0,
            input_format="BGR",
            backbone_name="build_retinanet_resnet_fpn_backbone",
            channel_indexes=(0, 7, 15, 31),
    ):
        super().__init__(backbone=backbone,
                         head=head,
                         head_in_features=head_in_features,
                         anchor_generator=anchor_generator,
                         box2box_transform=box2box_transform,
                         anchor_matcher=anchor_matcher,
                         num_classes=num_classes,
                         focal_loss_alpha=focal_loss_alpha,
                         focal_loss_gamma=focal_loss_gamma,
                         smooth_l1_beta=smooth_l1_beta,
                         box_reg_loss_type=box_reg_loss_type,
                         test_score_thresh=test_score_thresh,
                         test_topk_candidates=test_topk_candidates,
                         test_nms_thresh=test_nms_thresh,
                         max_detections_per_image=max_detections_per_image,
                         pixel_mean=pixel_mean,
                         pixel_std=pixel_std,
                         vis_period=vis_period,
                         input_format=input_format)
        self.channel_indexes = channel_indexes
        self.backbone_name = backbone_name

    @classmethod
    def from_config(cls, cfg):
        ret = RetinaNet.from_config(cfg)
        ret.update({
            "channel_indexes": cfg.CHANNEL_INDEXES,
            "backbone_name": cfg.MODEL.BACKBONE.NAME,
        })
        return ret

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
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
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        backbone_features = self.backbone(images.tensor)
        features = [backbone_features[f] for f in self.head_in_features]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)
                    # visualize the feature map here
                    # get the height and width
                    height = batched_inputs[0].get("height", images.image_sizes[0][0])
                    width = batched_inputs[0].get("width", images.image_sizes[0][1])
                    visualize_featuremaps(featuremap_dict={k: backbone_features[k] for k in self.head_in_features},
                                          backbone_name=self.backbone_name,
                                          height=height, width=width,
                                          channel_indexes=self.channel_indexes)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
