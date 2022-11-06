import cv2
import torch
import numpy as np
from detectron2.utils.events import get_event_storage
import torch.nn.functional as F


def get_principle_components(embedding, num_components=3):
    """Calculates the principal components given the embedding features.
    Args:
        embedding: A 2-D float tensor with shape `[batch x h x w, embedding_dim]`.
        num_components: The number of principal components to return.
    Returns:
        A 2-D float tensor with principal components in the last dimension.
    """
    embedding -= torch.mean(embedding, dim=0, keepdim=True)
    sigma = torch.matmul(embedding.transpose(0, 1), embedding)
    u, _, _ = torch.svd(sigma)
    return u[:, :num_components]


def pca(embedding, num_components=3):
    """Conducts principal component analysis on the embedding features.
    This function is used to reduce the dimensionality of the embedding, so that
    we can visualize the embedding as an RGB image.
    Args:
        embedding: A 4-D float tensor with shape `[batch, embedding_dim, h, w]`.
        num_components: The number of principal components to be reduced to.
    Returns:
        A 4-D float tensor with shape [batch, num_components, height, width].
    """
    N, c, h, w = embedding.size()
    embedding = embedding.permute([0, 2, 3, 1]).reshape([-1, c])

    pc = get_principle_components(embedding, num_components)
    embedding = torch.matmul(embedding, pc)
    embedding = embedding.reshape([N, h, w, -1]).permute([0, 3, 1, 2])
    return embedding


def decode_featuremap(featuremap):
    """Decode depth image to pseudo colorful image
    Args:
        featuremap (np.ndarray): a (M, N) array of float values denoting activation
    Returns:
        color (np.ndarray): a (M, N, 3) the resulting decoded color image.
    """
    color = cv2.applyColorMap(cv2.convertScaleAbs(featuremap.astype(np.uint8), alpha=60), cv2.COLORMAP_JET)
    return color


def visualize_featuremaps(featuremap_dict,
                          backbone_name,
                          height=None,
                          width=None,
                          channel_indexes=(0, 7, 15, 31)):
    """
    Visualize the featuremaps
    Args:
        featuremap_dict: the featuremap dict contains multiple featuremaps from the output of different backbone stage
            with a format of {K:v} e.g., {'res1': feat1, 'res2': feat2, 'res3': feat3, 'res4': feat4} for resnet
        backbone_name: the name of the backbone
        height: the height of the input image to interpolate the featuremap, if None, show the original size
        width: the width of the input image to interpolate the featuremap, if None, show the original size
        channel_indexes: the channel indexes to visualize, if None, then mean all channels
    Returns:
        None
    """
    storage = get_event_storage()
    # Since there are multiple channels for each featuremap
    # we only sample some channel from it
    # to visualize the training process consistently, we keep visualize the same channels
    for name, featuremap in featuremap_dict.items():
        # only visualize the featuremap from the first image
        featuremap = featuremap.detach()
        # interpolate the featuremap image to image size if height and width provided
        if height is not None and width is not None:
            featuremap = F.interpolate(featuremap, size=(height, width), mode='bilinear')
        if channel_indexes is not None:
            single_featuremap = featuremap.cpu()[0]
            C, H, W = single_featuremap.size()  # shape (C, H, W)
            for channel_index in channel_indexes:
                if channel_index <= C:
                    _featuremap_img = single_featuremap[channel_index]  # shape (H, W)
                    # normalize the img by projecting from [min, max] -> [0, 1] -> [0, 255]
                    _featuremap_img = (_featuremap_img - _featuremap_img.min()) / (
                            _featuremap_img.max() - _featuremap_img.min())
                    _featuremap_img = _featuremap_img * 255
                    colored_img = decode_featuremap(_featuremap_img.numpy())
                    # make it to torch format e.g., channel first
                    colored_img = torch.from_numpy(colored_img).permute(2, 0, 1)
                    storage.put_image('{}/{} channel {}'.format(backbone_name, name, channel_index), colored_img)
        else:
            # print("===> Visualize using PCA")
            # featuremap = pca(embedding=featuremap)
            # # only visualize the first featuremap
            # _featuremap_img = featuremap[0]
            # storage.put_image('{}/{}'.format(backbone_name, name), _featuremap_img)
            print("===> Visualize using Mean")
            _featuremap_img = featuremap[0]
            _featuremap_img = torch.mean(_featuremap_img, dim=0, keepdim=True)
            storage.put_image('{}/{}'.format(backbone_name, name), _featuremap_img)
