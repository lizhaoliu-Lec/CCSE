import argparse
import os.path
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur, ColorJitter

from env import stroke_normal, stroke_draw


def play1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--d2', action='store_true')

    args = parser.parse_args()
    print("===> args.pretrained: {}".format(args.pretrained))
    print("===> type(args.pretrained): {}".format(type(args.pretrained)))

    print("===> args.d2: {}".format(args.d2))
    print("===> type(args.d2): {}".format(type(args.d2)))

    import torch

    t = torch.Tensor([True])

    f = torch.Tensor([False])

    print(t, f)

    if t:
        print("=== t")
    else:
        print("=== nt")

    if f:
        print("=== f")
    else:
        print("=== nf")


def binarize_img(img):
    # black part=0: background
    # color part!=0: stroke
    black_part = img == 0
    color_part = img != 0
    img[black_part] = 1
    img[color_part] = 0
    img *= 255
    return img


def invert_img(img):
    return 1 - img / 255


def binaryMaskIOU(mask1, mask2):
    mask1_area = torch.count_nonzero(mask1)
    mask2_area = torch.count_nonzero(mask2)
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou


def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union


def l1_loss(mask1, mask2):
    return F.l1_loss(mask1, mask2, reduction='mean')


def play2():
    import cv2 as cv
    import os
    import numpy as np

    root1 = 'mean'
    root2 = '2'
    width = 128
    concat_root = 'data/font_concat/{}'.format(root2)
    img_list = os.listdir(concat_root)
    add = img_list[0]

    # 1_1.png
    img_id = int(add.split('.')[0].split('_')[0])
    stroke = int(add.split('.')[0].split('_')[1])
    stroke -= 1

    point_src = np.load('data/font_10_stroke/{}_point.npy'.format(root1))
    point_tgt = np.load('data/font_10_stroke/{}_point.npy'.format(root2))

    p = 'data/font_concat/{}/'.format(root2) + add
    print("===> p: {}".format(p))
    print("===> os.path.exists(p): {}".format(os.path.exists(p)))
    img = cv.imread(p)
    img = cv.resize(img, (width * 4, width))

    img_ref = img[:, 1 * width:2 * width, :]
    img_tgt = img[:, 3 * width:4 * width, :]
    img_point = np.zeros((width, width, 3))
    img_point[0:10, 0:2, 0] = stroke_normal(point_src[img_id][stroke])
    img_point[0:10, 0:2, 1] = point_tgt[img_id][stroke] / 64 - 1
    img_point[0:10, 0:2, 2] = point_src[img_id][stroke] / 64 - 1
    img_src = stroke_draw(stroke_normal(point_src[img_id][stroke]))
    img_canvas = stroke_draw(stroke_normal(point_src[img_id][stroke]))

    # img_src = img_src.transpose((2, 0, 1)).reshape(3, width, width) / 255.
    # img_canvas = img_canvas.transpose((2, 0, 1)).reshape(3, width, width) / 255.
    # img_ref = img_ref.transpose((2, 0, 1)).reshape(3, width, width) / 255.
    # img_point = img_point.transpose((2, 0, 1)).reshape(3, width, width)

    print("===> img_src.shape: {}".format(img_src.shape))
    print("===> img_canvas.shape: {}".format(img_canvas.shape))
    print("===> img_ref.shape: {}".format(img_ref.shape))
    print("===> img_tgt.shape: {}".format(img_tgt.shape))
    print("===> img_point.shape: {}".format(img_point.shape))

    # make testImage dir
    base_dir = 'testImage'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    cv.imwrite(os.path.join(base_dir, 'img_src.png'), img_src)
    cv.imwrite(os.path.join(base_dir, 'img_canvas.png'), img_canvas)
    cv.imwrite(os.path.join(base_dir, 'img_ref.png'), img_ref)
    cv.imwrite(os.path.join(base_dir, 'img_tgt.png'), img_tgt)
    cv.imwrite(os.path.join(base_dir, 'img_point.png'), img_point)
    cv.imwrite(os.path.join(base_dir, 'img_ref_binarized.png'), binarize_img(img_ref))
    cv.imwrite(os.path.join(base_dir, 'img_tgt_binarized.png'), binarize_img(img_tgt))

    b_ref, b_tgt = invert_img(binarize_img(img_ref)), invert_img(binarize_img(img_tgt))

    b_ref, b_tgt = torch.Tensor(b_ref), torch.Tensor(b_tgt)
    print("===> binaryMaskIOU(b_ref, b_tgt): {}".format(binaryMaskIOU(b_ref, b_tgt)))
    print("===> iou(b_ref, b_tgt): {}".format(iou(b_ref, b_tgt)))
    print("===> l1_loss(b_ref, b_tgt): {}".format(l1_loss(b_ref, b_tgt)))


def get_gt_img(path, width=128):
    import os
    import cv2 as cv

    add = os.path.basename(path)
    img_id = int(add.split('.')[0].split('_')[0])
    stroke = int(add.split('.')[0].split('_')[1])

    img = cv.imread(path)
    img = cv.resize(img, (width * 4, width))
    img_src = stroke_draw(stroke_normal(point_src[img_id][stroke]))


def play_transformation():
    a = torch.randn((5, 3, 224, 224)).cuda()
    _a = GaussianBlur(kernel_size=5)(a)
    print("===> _a.size(): {}".format(_a.size()))
    print("===> _a.device: {}".format(_a.device))
    _b = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(_a)
    print("===> _b.size(): {}".format(_b.size()))
    print("===> _b.device: {}".format(_b.device))


if __name__ == '__main__':
    # point_src = np.load('data/font_10_stroke/{}_point.npy'.format(2))
    # play2()
    play_transformation()
