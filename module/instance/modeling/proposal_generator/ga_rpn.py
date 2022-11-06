import torch.nn as nn
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY


@PROPOSAL_GENERATOR_REGISTRY.register()
class GuidedAnchorProposalNetwork(nn.Module):
    """
    Region Proposal by Guided Anchoring. https://arxiv.org/abs/1901.03278.pdf
    """
    ...
