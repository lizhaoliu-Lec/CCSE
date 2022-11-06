import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from detectron2.utils.registry import Registry
from fvcore.nn import weight_init

ATTENTION_REGISTRY = Registry("ATTENTION")
ATTENTION_REGISTRY.__doc__ = """
Registry for attention module, which improves the performance with minimal overhead.

The registered object will be called with `obj(cfg)`.
"""


@ATTENTION_REGISTRY.register()
class Identity(nn.Module):
    @configurable
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    @classmethod
    def from_config(cls, cfg):
        return {}


@ATTENTION_REGISTRY.register()
class SAM(nn.Module):
    """
    Spatial Attention Module depicted in CenterMask: https://arxiv.org/pdf/1911.06667.pdf
    """

    @configurable
    def __init__(self, kernel_size=3):
        super(SAM, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        weight_init.c2_msra_fill(self.conv)

    @classmethod
    def from_config(cls, cfg):
        return {
            "kernel_size": cfg.MODEL.ATTENTION.SAM.KERNEL_SIZE,
        }

    def forward(self, x):
        try:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            scale = torch.cat([avg_out, max_out], dim=1)
            scale = self.conv(scale)
            out = x * self.sigmoid(scale)
        except:
            out = x

        return out


def permute_to_N_HW_C(tensor):
    """
    Transpose/reshape a tensor from (N, C, H, W) to (N, (HxW), C)
    """
    assert tensor.dim() == 4, tensor.shape
    N, C, H, W = tensor.shape
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor.reshape(N, H * W, C)  # Size=(N,HW,C)
    return tensor


def N_HW_C_to_N_C_H_W(tensor, HW):
    """
    Transpose/reshape a tensor from (N, (HxW), C) to (N, C, H, W)
    """
    assert tensor.dim() == 3, tensor.shape
    assert len(HW) == 2, HW
    N, C = tensor.size(0), tensor.size(2)
    H, W = HW
    tensor = tensor.permute(0, 2, 1)
    tensor = tensor.reshape(N, C, H, W)  # Size=(N,C,H,W)
    return tensor


@ATTENTION_REGISTRY.register()
class NonLocal(nn.Module):
    """
    Non local Attention Module depicted in BCNet: https://arxiv.org/abs/2103.12340
    """

    @configurable
    def __init__(self, in_channel, kernel_size=1, reduction=1, norm="bn"):
        super(NonLocal, self).__init__()

        assert kernel_size in (1, 3), 'kernel size must be 1 or 3'
        padding = 1 if kernel_size == 3 else 0

        self.query = nn.Conv2d(in_channel, in_channel // reduction,
                               kernel_size=kernel_size, stride=1,
                               padding=padding, bias=False)
        self.key = nn.Conv2d(in_channel, in_channel // reduction,
                             kernel_size=kernel_size, stride=1,
                             padding=padding, bias=False)
        self.value = nn.Conv2d(in_channel, in_channel // reduction,
                               kernel_size=kernel_size, stride=1,
                               padding=padding, bias=False)
        self.output = nn.Conv2d(in_channel // reduction, in_channel,
                                kernel_size=kernel_size, stride=1,
                                padding=padding, bias=False)

        self.norm = None
        if norm == "bn":
            self.norm = nn.BatchNorm2d(in_channel, eps=1e-04)

        # To make the attention score after softmax more stable, according to Transformer paper
        self.scale = 1.0 / ((in_channel // reduction) ** 0.5)

        weight_init.c2_msra_fill(self.query)
        weight_init.c2_msra_fill(self.key)
        weight_init.c2_msra_fill(self.value)
        weight_init.c2_msra_fill(self.output)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channel": cfg.MODEL.ATTENTION.NON_LOCAL.IN_CHANNEL,
            "kernel_size": cfg.MODEL.ATTENTION.NON_LOCAL.KERNEL_SIZE,
            "reduction": cfg.MODEL.ATTENTION.NON_LOCAL.REDUCTION,
            "norm": cfg.MODEL.ATTENTION.NON_LOCAL.NORM,
        }

    def forward(self, x):
        H, W = x.size()[-2:]
        
        q, k, v = self.query(x), self.key(x), self.value(x)
        # Note that the C below is multiplied by reduction e.g., C * self.reduction
        # q: (N, C, H, W) ==> (N, H*W, C)
        # k: (N, C, H, W) ==> (N, H*W, C) ==> (N, C, H*W)
        # bmm(q, k) ==> (N, H*W, H*W)
        # v: (N, C, H, W) ==> (N, H*W, C)
        q = permute_to_N_HW_C(q)
        k = permute_to_N_HW_C(k).transpose(2, 1)
        v = permute_to_N_HW_C(v)

        dot_product = torch.matmul(q, k) * self.scale  # (N, H*W, H*W)
        attention_score = F.softmax(dot_product, dim=2)
        out = torch.matmul(attention_score, v)  # (N, H*W, C)
        out = N_HW_C_to_N_C_H_W(out, HW=(H, W))

        out = self.output(out)

        if self.norm is not None:
            out = self.norm(out)

        return x + out  # residual connection


def build_attention(cfg):
    """
    Build a attention defined by `cfg.MODEL.ATTENTION.NAME`.
    """
    name = cfg.MODEL.ATTENTION.NAME
    return ATTENTION_REGISTRY.get(name)(cfg)
