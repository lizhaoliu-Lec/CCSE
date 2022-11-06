"""
Add the newly added modules' default parameter to detectron2's default config
Also, the newly added modules themselves for registry
"""
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy

from .detector import SparseRCNN
from .dataset_mapper import SparseRCNNDatasetMapper
from detectron2.config.defaults import _C, CN

# ---------------------------------------------------------------------------- #
# SparseRCNN options
# ---------------------------------------------------------------------------- #
_C.MODEL.SparseRCNN = CN()
_C.MODEL.SparseRCNN.NUM_CLASSES = 80
_C.MODEL.SparseRCNN.NUM_PROPOSALS = 300

# SparseRCNN Head. (e.g., Transformer)
_C.MODEL.SparseRCNN.NHEADS = 8
_C.MODEL.SparseRCNN.DROPOUT = 0.0
_C.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048
_C.MODEL.SparseRCNN.ACTIVATION = 'relu'
_C.MODEL.SparseRCNN.HIDDEN_DIM = 256
_C.MODEL.SparseRCNN.NUM_CLS = 1
_C.MODEL.SparseRCNN.NUM_REG = 3
_C.MODEL.SparseRCNN.NUM_HEADS = 6

# Dynamic Conv.
_C.MODEL.SparseRCNN.NUM_DYNAMIC = 2
_C.MODEL.SparseRCNN.DIM_DYNAMIC = 64

# Loss.
_C.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0
_C.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0
_C.MODEL.SparseRCNN.L1_WEIGHT = 5.0
_C.MODEL.SparseRCNN.MASK_WEIGHT = 2.0
_C.MODEL.SparseRCNN.DEEP_SUPERVISION = True
_C.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.1

# Focal Loss.
_C.MODEL.SparseRCNN.USE_FOCAL = True
_C.MODEL.SparseRCNN.ALPHA = 0.25
_C.MODEL.SparseRCNN.GAMMA = 2.0
_C.MODEL.SparseRCNN.PRIOR_PROB = 0.01

# Optimizer.
_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.BACKBONE_MULTIPLIER = 1.0
