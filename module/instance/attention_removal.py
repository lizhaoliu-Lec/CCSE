from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool


class ReversedSEAttention(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature: torch.Tensor, attentions: List[torch.Tensor]):
        """
        Perform SE attention from the given feature and list of attention vectors
        Args:
            feature: Tensor[N, C, H, W]
            attentions: list per Box ROI vectors List[Tensor[L, C]], note that the list length is equal to batchsize N

        Returns:
        The SE attended feature
        """
        N, C, _, _ = feature.size()
        assert N == len(attentions)
        # first perform attention on all ROI vectors
        attentions = [self.mlp(att) for att in attentions]  # List[[L, C]] => List[[L, C]]

        # multiply the attention vectors on the feature
        # (1) first aggregate the attention from multiple roi
        batch_aggregated_attention = []  # List[Tensor[C, 1, 1]]
        for batch_id, att in enumerate(attentions):
            # att [L, C]
            L, C = att.size()
            att = self.mlp(att).mean(dim=0).reshape([C, 1, 1])  # [L, C] => [L, C] => [C] => [C, 1, 1]
            batch_aggregated_attention.append(att)
        aggregated_attention = torch.stack(batch_aggregated_attention, dim=0)  # Tensor[N, C, 1, 1]

        aggregated_attention = 1 - aggregated_attention  # reverse the attention score to achieve attention removal

        return F.relu(feature + self.conv(feature) * aggregated_attention)  # a little residual connection here


class AttentionRemoval(nn.Module):
    def __init__(self, scales, in_channels_list):
        super(AttentionRemoval, self).__init__()
        self.scales = scales
        self.in_channels_list = in_channels_list
        assert len(scales) == len(in_channels_list), 'Inconsistent length of scales ({}), in_channels_list ({})'.format(
            len(self.scales), len(in_channels_list))
        self.se_attention_list = nn.ModuleList([ReversedSEAttention(in_channels=c) for c in self.in_channels_list])

    def forward(self, features: List[torch.Tensor], boxes: List[torch.Tensor]):
        """
        Perform attention removal on multiple level's features and corresponding boxes
        Args:
            features: List of per lvl features Tensor[N, C, H, W]
            boxes: List of per lvl boxes Tensor[K, 5], where the first column denotes the batch index

        Returns:
        The deAttended features
        """
        assert len(self.scales) == len(features), 'Inconsistent length of scales ({}), features ({})'.format(
            len(self.scales), len(features))

        # construct roi for each scales
        attention_removed_features = []
        for scale, feature, _boxes, se_attention in zip(self.scales, features, boxes, self.se_attention_list):
            # get the roi features Tensor[K, C, 1, 1]
            roi_features = roi_pool(feature, boxes=_boxes, output_size=(1, 1), spatial_scale=scale)
            # reshape roi_features to List[Tensor[L, C]]
            N, C, _, _ = feature.size()
            reshaped_roi_features = [[] for _ in range(N)]
            for box, roi_feature in zip(boxes, roi_features):
                reshaped_roi_features[box[0]].append(roi_feature)
            reshaped_roi_features = [torch.stack(roi, dim=0) for roi in reshaped_roi_features]

            # perform reverse se attention on feature
            attention_removed_features.append(se_attention(feature, reshaped_roi_features))

        return attention_removed_features
