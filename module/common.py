import torch


def kld_loss(pred, target):
    """
    KL divergence Loss
    """
    pred = pred.log_softmax(dim=-1)
    target = target.softmax(dim=-1)
    return torch.nn.KLDivLoss(reduction='batchmean')(pred, target)
