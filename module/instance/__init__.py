"""
Add the newly added modules' default parameter to detectron2's default config
Also, the newly added modules themselves for registry
"""
import copy

from detectron2.config.defaults import _C, CN
from .modeling import build_shift_generator, SHIFT_GENERATOR_REGISTRY
from .meta_arch import FCOS, AutoAssign, VisualizedRetinaNet
from .modeling.roi_heads import MaskRCNNConvAttentionUpsampleHead
from .modeling.roi_heads import RepulsionROIHeads, NMSGeneralizedROIHeads, MaskIOUROIHeads
from .modeling.proposal_generator.rpn import NMSGeneralizedRPN, GroupRPN, MultiClassRPNHead

# ---------------------------------------------------------------------------- #
# Shift generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.SHIFT_GENERATOR = CN()
# The generator can be any name in the SHIFT_GENERATOR registry
_C.MODEL.SHIFT_GENERATOR.NAME = "ShiftGenerator"
# NUm shifts in shift gnerator
_C.MODEL.SHIFT_GENERATOR.NUM_SHIFTS = 1
# Relative offset between the center of the first anchor and the top-left corner of the image
# Value has to be in [0, 1). Recommend to use 0.5, which means half stride.
# The value is not expected to affect model accuracy.
_C.MODEL.SHIFT_GENERATOR.OFFSET = 0.0

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.NMS_TYPE_TEST = "normal"
_C.MODEL.FCOS = CN()
# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80

_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.FCOS.NUM_CONVS = 4

# The strides from resnet backbone features
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
# _C.MODEL.FCOS.IOU_THRESHOLDS = [0.4, 0.5]
# _C.MODEL.FCOS.IOU_LABELS = [0, -1, 1]

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.FCOS.PRIOR_PROB = 0.01

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.FCOS.SCORE_THRESH_TEST = 0.05
# Select topk candidates before NMS
_C.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.FCOS.NMS_THRESH_TEST = 0.6

# Weights on (dx, dy, dw, dh) for normalizing FCOS anchor regression targets
_C.MODEL.FCOS.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Loss parameters
_C.MODEL.FCOS.CENTERNESS_ON_REG = False
_C.MODEL.FCOS.NORM_REG_TARGETS = False
_C.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.FCOS.IOU_LOSS_TYPE = "iou"
_C.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 0.0
_C.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST = ([-1, 64],
                                          [64, 128],
                                          [128, 256],
                                          [256, 512],
                                          [512, float("inf")],)
# Options are: "smooth_l1", "giou"
_C.MODEL.FCOS.BBOX_REG_LOSS_TYPE = "smooth_l1"

# One of BN, SyncBN, FrozenBN, GN
# Only supports GN until unshared norm is implemented
# FCOS only only uses GN
# _C.MODEL.FCOS.NORM = ""

# ---------------------------------------------------------------------------- #
# AutoAssign Options
# ---------------------------------------------------------------------------- #
_C.MODEL.NMS_TYPE_TEST = "normal"
_C.MODEL.AUTO_ASSIGN = CN()
# This is the number of foreground classes.
_C.MODEL.AUTO_ASSIGN.NUM_CLASSES = 80

_C.MODEL.AUTO_ASSIGN.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.AUTO_ASSIGN.NUM_CONVS = 4

# The strides from resnet backbone features
_C.MODEL.AUTO_ASSIGN.FPN_STRIDES = [8, 16, 32, 64, 128]

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
# _C.MODEL.AUTO_ASSIGN.IOU_THRESHOLDS = [0.4, 0.5]
# _C.MODEL.AUTO_ASSIGN.IOU_LABELS = [0, -1, 1]

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.AUTO_ASSIGN.PRIOR_PROB = 0.02

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.AUTO_ASSIGN.SCORE_THRESH_TEST = 0.05
# Select topk candidates before NMS
_C.MODEL.AUTO_ASSIGN.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.AUTO_ASSIGN.NMS_THRESH_TEST = 0.6

# Weights on (dx, dy, dw, dh) for normalizing AUTO_ASSIGN anchor regression targets
_C.MODEL.AUTO_ASSIGN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Loss parameters
_C.MODEL.AUTO_ASSIGN.NORM_REG_TARGETS = True
_C.MODEL.AUTO_ASSIGN.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.AUTO_ASSIGN.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.AUTO_ASSIGN.IOU_LOSS_TYPE = "giou"
_C.MODEL.AUTO_ASSIGN.REG_WEIGHT = 5.0

# ---------------------------------------------------------------------------- #
# Shared Visualized Version
# ---------------------------------------------------------------------------- #
_C.CHANNEL_INDEXES = (0, 7, 15, 31)

# ---------------------------------------------------------------------------- #
# Visualized Retinanet Options (Same as the Retinanet, thus we just copy)
# ---------------------------------------------------------------------------- #
_C.MODEL.VISUALIZED_RETINANET = copy.deepcopy(_C.MODEL.RETINANET)

# For iterative prediction
_C.MODEL.VISUALIZED_RETINANET.ITERATIVE_MAX_ITER = 5
_C.MODEL.VISUALIZED_RETINANET.ITERATIVE_TOP_K = 5

# ---------------------------------------------------------------------------- #
# Attention
# ---------------------------------------------------------------------------- #
_C.MODEL.ATTENTION = CN()
_C.MODEL.ATTENTION.NAME = "Identity"
# for SAM Attention
_C.MODEL.ATTENTION.SAM = CN()
_C.MODEL.ATTENTION.SAM.KERNEL_SIZE = 3
# for Non local Attention
_C.MODEL.ATTENTION.NON_LOCAL = CN()
_C.MODEL.ATTENTION.NON_LOCAL.IN_CHANNEL = 0
_C.MODEL.ATTENTION.NON_LOCAL.KERNEL_SIZE = 1
_C.MODEL.ATTENTION.NON_LOCAL.REDUCTION = 1
_C.MODEL.ATTENTION.NON_LOCAL.NORM = "bn"

# ---------------------------------------------------------------------------- #
# Repulsion Loss
# ---------------------------------------------------------------------------- #
_C.MODEL.REPULSION_LOSS = CN()
_C.MODEL.REPULSION_LOSS.REP_GT_SIGMA = 0.5
_C.MODEL.REPULSION_LOSS.REP_BOX_SIGMA = 0.5
_C.MODEL.REPULSION_LOSS.REP_GT_LOSS_WEIGHT = 0.1
_C.MODEL.REPULSION_LOSS.REP_BOX_LOSS_WEIGHT = 0.9

# ---------------------------------------------------------------------------- #
# Generalized NMS
# ---------------------------------------------------------------------------- #
_C.MODEL.NMS_SCORE_THRESHOLD = 0.001
_C.MODEL.NMS_TYPE = "normal"

# ---------------------------------------------------------------------------- #
# MaskIOUROIHeads
# ---------------------------------------------------------------------------- #
_C.MODEL.MASKIOU_LOSS_WEIGHT = 1.0
_C.MODEL.MASKIOU_ON = False
_C.MODEL.ROI_MASKIOU_HEAD = CN()
_C.MODEL.ROI_MASKIOU_HEAD.NAME = "MaskIoUHead"

# ---------------------------------------------------------------------------- #
# Group RPN
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_IOU_THRESHOLD = 0.1
_C.MODEL.USE_GROUP_RPN_HEAD = False
