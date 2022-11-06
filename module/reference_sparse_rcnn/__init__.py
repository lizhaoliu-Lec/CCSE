"""
Add the newly added modules' default parameter to detectron2's default config
Also, the newly added modules themselves for registry
"""
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy

from .detector import ReferenceSparseRCNN
from .dataset_mapper import ReferenceSparseRCNNDatasetMapper
from detectron2.config.defaults import _C, CN

# ---------------------------------------------------------------------------- #
# ReferenceSparseRCNN options
# ---------------------------------------------------------------------------- #
_C.MODEL.ReferenceSparseRCNN = CN()
_C.MODEL.ReferenceSparseRCNN.NUM_CLASSES = 80
_C.MODEL.ReferenceSparseRCNN.NUM_PROPOSALS = 300

# ReferenceSparseRCNN Head. (e.g., Transformer)
_C.MODEL.ReferenceSparseRCNN.NHEADS = 8
_C.MODEL.ReferenceSparseRCNN.DROPOUT = 0.0
_C.MODEL.ReferenceSparseRCNN.DIM_FEEDFORWARD = 2048
_C.MODEL.ReferenceSparseRCNN.ACTIVATION = 'relu'
_C.MODEL.ReferenceSparseRCNN.HIDDEN_DIM = 256
_C.MODEL.ReferenceSparseRCNN.NUM_CLS = 1
_C.MODEL.ReferenceSparseRCNN.NUM_REG = 3
_C.MODEL.ReferenceSparseRCNN.NUM_HEADS = 6

# Dynamic Conv.
_C.MODEL.ReferenceSparseRCNN.NUM_DYNAMIC = 2
_C.MODEL.ReferenceSparseRCNN.DIM_DYNAMIC = 64

# Loss.
_C.MODEL.ReferenceSparseRCNN.CLASS_WEIGHT = 2.0
_C.MODEL.ReferenceSparseRCNN.GIOU_WEIGHT = 2.0
_C.MODEL.ReferenceSparseRCNN.L1_WEIGHT = 5.0
_C.MODEL.ReferenceSparseRCNN.MASK_WEIGHT = 2.0
_C.MODEL.ReferenceSparseRCNN.DEEP_SUPERVISION = True
_C.MODEL.ReferenceSparseRCNN.NO_OBJECT_WEIGHT = 0.1

# Focal Loss.
_C.MODEL.ReferenceSparseRCNN.USE_FOCAL = True
_C.MODEL.ReferenceSparseRCNN.ALPHA = 0.25
_C.MODEL.ReferenceSparseRCNN.GAMMA = 2.0
_C.MODEL.ReferenceSparseRCNN.PRIOR_PROB = 0.01

# Optimizer.
_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.BACKBONE_MULTIPLIER = 1.0
