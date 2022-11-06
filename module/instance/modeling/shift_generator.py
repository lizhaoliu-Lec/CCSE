import copy
from typing import List

import torch
import torch.nn as nn
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import _create_grid_offsets
from detectron2.utils.registry import Registry

SHIFT_GENERATOR_REGISTRY = Registry("SHIFT_GENERATOR")
SHIFT_GENERATOR_REGISTRY.__doc__ = """
Registry for modules that creates object detection shifts for feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@SHIFT_GENERATOR_REGISTRY.register()
class ShiftGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of shifts.
    """

    @configurable
    def __init__(self, *, num_shifts, strides, offset):
        super().__init__()
        # fmt: off
        self.num_shifts = num_shifts
        self.strides = strides
        self.offset = offset
        assert 0.0 <= self.offset < 1.0, self.offset
        # fmt: on

        self.num_features = len(self.strides)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "num_shifts": cfg.MODEL.SHIFT_GENERATOR.NUM_SHIFTS,
            "strides": [x.stride for x in input_shape],
            "offset": cfg.MODEL.SHIFT_GENERATOR.OFFSET,
        }

    @property
    def num_cell_shifts(self):
        return [self.num_shifts for _ in self.strides]

    def grid_shifts(self, grid_sizes, device):
        shifts_over_all = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, device)
            shifts = torch.stack((shift_x, shift_y), dim=1)

            shifts_over_all.append(shifts.repeat_interleave(self.num_shifts, dim=0))

        return shifts_over_all

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate shifts.

        Returns:
            list[list[Tensor]]: a list of #image elements. Each is a list of #feature level tensors.
                The tensors contains shifts of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all = self.grid_shifts(grid_sizes, features[0].device)

        shifts = [copy.deepcopy(shifts_over_all) for _ in range(num_images)]
        return shifts


def build_shift_generator(cfg, input_shape):
    """
    Built an shift generator from `cfg.MODEL.SHIFT_GENERATOR.NAME`.
    """
    shift_generator = cfg.MODEL.SHIFT_GENERATOR.NAME
    return SHIFT_GENERATOR_REGISTRY.get(shift_generator)(cfg, input_shape)
