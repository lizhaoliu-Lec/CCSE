from typing import Tuple

import torch
import numpy as np

from detectron2.layers import nonzero_tuple


def smooth_ln(x, sigma=0.5):
    """
    Smooth ln loss (Xinlong Wang et. al)
    https://arxiv.org/abs/1711.07752

    Smooth ln loss that maximize x between [0, 1)

    Args:
        x (Scalar or Tensor): the var to maximize between [0, 1]
        sigma (Scalar): between [0, 1), the larger, the more penalization for x when x is close to 1
    """
    assert torch.min(x) >= 0.0 and torch.max(x) <= 1.0, "bad x: x not in [0.0, 1.0]"
    assert 0.0 <= sigma < 1.0, "bad sigma: sigma not in [0.0, 1.0)"
    return torch.where(torch.le(x, sigma), -torch.log(1 - x), ((x - sigma) / (1 - sigma)) - np.log(1 - sigma))


def iog(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        eps: float = 1e-7,
) -> torch.Tensor:
    """
    Intersection over Groundtruth (Xinlong Wang et. al)
    https://arxiv.org/abs/1711.07752

    IoG: the pred bbox and corresponding gt box e.g., 1 - intersect(bbox, gt) / area(gt).

    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    # calculate intersection with the overlapped bboxes
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    # In IoG, the union is the area of the gt
    unionk = (x2g - x1g) * (y2g - y1g)
    # Add a small eps to avoid zero
    iogk = intsctk / (unionk + eps)

    return iogk


def RepGT_loss(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-7,
        sigma: float = 0.5,
) -> torch.Tensor:
    """
    RepGT_loss Loss (Xinlong Wang et. al)
    https://arxiv.org/abs/1711.07752

    RepGT loss that penalizes the pred bbox not close to other GT boxes other than the matched ones.

    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
        sigma (float): see smooth_ln:sigma for details
    """

    _iog = iog(boxes1, boxes2, eps)

    loss = smooth_ln(_iog, sigma=sigma)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def pairwise_intersection(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Tensor): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


def boxes_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of all the boxes.

    Returns:
        torch.Tensor: a vector with areas of each box.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def pairwise_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes_area(boxes1)  # [N]
    area2 = boxes_area(boxes2)  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        torch.gt(inter, 0),
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def RepBox_loss(
        boxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-7,
        sigma: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RepBox_loss Loss (Xinlong Wang et. al)
    https://arxiv.org/abs/1711.07752

    RepBox loss that repels each pred boxes from others with different designated targets..
    Also return the number of
    Args:
        boxes (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
        sigma (float): see smooth_ln:sigma for details
    """

    _pair_wise_iou = pairwise_iou(boxes, boxes)
    # filter out the diagonal and the symmetry ones
    _pair_wise_iou = torch.triu(_pair_wise_iou, diagonal=1)

    loss = smooth_ln(_pair_wise_iou, sigma=sigma)

    num_non_zero = torch.count_nonzero(_pair_wise_iou.detach())
    # num_non_zero = nonzero_tuple(_pair_wise_iou)[0]

    if reduction == "mean":
        loss = loss.sum() / (num_non_zero + eps) if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss, num_non_zero
