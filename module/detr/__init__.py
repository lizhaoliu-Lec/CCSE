"""
Add the newly added modules' default parameter to detectron2's default config
Also, the newly added modules themselves for registry
"""
from detectron2.config.defaults import _C, CN
from .d2.detr import Detr

# ---------------------------------------------------------------------------- #
# DETR options
# ---------------------------------------------------------------------------- #
_C.MODEL.DETR = CN()
_C.MODEL.DETR.NUM_CLASSES = 80

# For Segmentation
_C.MODEL.DETR.FROZEN_WEIGHTS = ''

# LOSS
_C.MODEL.DETR.GIOU_WEIGHT = 2.0
_C.MODEL.DETR.L1_WEIGHT = 5.0
_C.MODEL.DETR.DEEP_SUPERVISION = True
_C.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1

# TRANSFORMER
_C.MODEL.DETR.NHEADS = 8
_C.MODEL.DETR.DROPOUT = 0.1
_C.MODEL.DETR.DIM_FEEDFORWARD = 2048
_C.MODEL.DETR.ENC_LAYERS = 6
_C.MODEL.DETR.DEC_LAYERS = 6
_C.MODEL.DETR.PRE_NORM = False

_C.MODEL.DETR.HIDDEN_DIM = 256
_C.MODEL.DETR.NUM_OBJECT_QUERIES = 100

_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.BACKBONE_MULTIPLIER = 0.1
