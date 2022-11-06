import copy
from functools import reduce

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur, ColorJitter
from tqdm import tqdm
import torchvision.transforms.functional as TF

from DRL.actor import ResNet
from DRL.ddpg import decode, move
from DRL.mover import ResNet_mover
from env import stroke_normal, stroke_draw
from meta_data import train_list_775, train_list, RESNET50_CONVERT_MAP
import os
import numpy as np
import cv2 as cv
import torch

from utils.util import t_print


def final_process_pretrained_weights(_weights):
    # conv1_weight = _weights['conv1.weight']
    # new_conv1_weight = torch.zeros((64, 9, 7, 7))  # (64, 3, 7, 7)
    # t_print("===> conv1_weight.size(): {}".format(conv1_weight.size()))
    # t_print("===> new_conv1_weight.size(): {}".format(new_conv1_weight.size()))
    # for _ in range(3):
    #     left = _*3
    #     right = (_+1) * 3
    #     t_print("===> left: {}, right: {}".format(left, right))
    #     new_conv1_weight[:, left:right, :, :] = conv1_weight.clone()
    # _weights['conv1.weight'] = new_conv1_weight
    for _ in ['conv1.weight', 'fc.weight', 'fc.bias']:
        if _ in _weights:
            _weights.pop(_)

    return _weights


def load_mover_weights(_mover, pretrained_path, is_d2):
    if pretrained_path is not None:
        assert os.path.exists(pretrained_path), '{} not exist'.format(pretrained_path)
        t_print("Loading pretrained weight from {}...".format(pretrained_path))
        pretrained_weights = None
        if is_d2:
            t_print("Converting pretrained weights from d2...")
            # weight for CCSE-HW pretrained resnet50
            # /home/liulizhao/dataset/SIS_exp/liulizhao/mask_rcnn_R_50_FPN_3x_handwritten/20220805.001734/model_0239999.pth
            d2_model = torch.load(pretrained_path)['model']
            converted_d2_model = {}
            for k, v in d2_model.items():
                if k in RESNET50_CONVERT_MAP:
                    converted_d2_model[RESNET50_CONVERT_MAP[k]] = v
            pretrained_weights = converted_d2_model
        else:
            # weight for imagenet pretrained resnet50
            # /home/liulizhao/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
            pretrained_weights = torch.load(pretrained_path)

        t_print("Final processing pretrained weights")
        pretrained_weights = final_process_pretrained_weights(pretrained_weights)
        _mover.load_state_dict(pretrained_weights, strict=False)
        t_print("Done loading pretrained weights")
    return _mover


def filter_train_img(img_list):
    _img_list = []
    for add in img_list:
        img_id = int(add.split('.')[0].split('_')[0])

        if img_id not in train_list_775 and img_id not in train_list[:100]:
            continue

        _img_list.append(add)

    return _img_list


class FontDataset(Dataset):
    def __init__(self, img_path, point_src_path, point_tgt_path, width=128, num_limit=100, train=False):
        self.width = width
        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        self.train = train
        if train:
            self.img_list = filter_train_img(self.img_list)
            num_limit = -1
        self.num_limit = num_limit
        self.img2fontID, self.fontID2cnt = self.filter_by_num()
        self.point_src = np.load(point_src_path)
        self.point_tgt = np.load(point_tgt_path)
        self.len = len(self.img_list)
        self.col = []
        cc = [128, 192, 255]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.col.append((cc[i], cc[j], cc[k]))
        self.col.append((64, 192, 255))
        self.col.append((255, 192, 64))
        self.col.append((192, 64, 255))

    def filter_by_num(self):
        if self.num_limit > 0:
            _img_list = []
            for img in self.img_list:
                font_id = int(img.split('.')[0].split('_')[0])
                if font_id <= self.num_limit:
                    _img_list.append(img)
            self.img_list = _img_list

        img2fontID = {}
        fontID2cnt = {}
        for img in self.img_list:
            font_id = int(img.split('.')[0].split('_')[0])
            img2fontID[img] = font_id
            if font_id not in fontID2cnt:
                fontID2cnt[font_id] = 0
            fontID2cnt[font_id] += 1

        return img2fontID, fontID2cnt

    def __getitem__(self, index):
        width = self.width
        add = self.img_list[index]
        img_id = int(add.split('.')[0].split('_')[0])
        stroke = int(add.split('.')[0].split('_')[1])
        stroke -= 1

        img = cv.imread(os.path.join(self.img_path, add))

        img_ref = img[:, 1 * width:2 * width, :]
        img_tgt = img[:, 3 * width:4 * width, :]
        img_point = np.zeros((width, width, 3))
        img_point[0:10, 0:2, 0] = stroke_normal(self.point_src[img_id][stroke])
        img_point[0:10, 0:2, 1] = self.point_tgt[img_id][stroke] / 64 - 1
        img_point[0:10, 0:2, 2] = self.point_src[img_id][stroke] / 64 - 1
        img_src = stroke_draw(stroke_normal(self.point_src[img_id][stroke]))
        img_canvas = stroke_draw(stroke_normal(self.point_src[img_id][stroke]))

        img_src = img_src.transpose((2, 0, 1)) / 255.
        img_canvas = img_canvas.transpose((2, 0, 1)) / 255.
        img_ref = img_ref.transpose((2, 0, 1)) / 255.
        img_point = img_point.transpose((2, 0, 1))

        src = torch.tensor(img_src).float()
        ref = torch.tensor(img_ref).float()
        tgt = torch.tensor(img_tgt).float()
        canvas = torch.tensor(img_canvas).float()
        point = torch.tensor(img_point).float()

        if not self.train:
            COL = torch.tensor(self.col[stroke]).float().view(3, 1, 1)
            return src, ref, canvas, point, COL, img_id, stroke, add, tgt
        else:
            COL = torch.tensor(self.col[stroke]).float().view(3)
            point = torch.tensor(img_point[0, 0:10, 0:2]).float()
            tgt_point = torch.tensor(img_point[1, 0:10, 0:2]).float()
            ref_point = torch.tensor(img_point[2, 0:10, 0:2]).float()
            return src, ref, canvas, point, COL, img_id, stroke, add, tgt, ref_point, tgt_point

    def __len__(self):
        return self.len


def binarize_img(img):
    # black part=0: background
    # color part!=0: stroke
    color_part = img != 0
    img[color_part] = 1
    img *= 255
    return img


def invert_img(img):
    return 1 - img / 255


def binaryMaskIOU(mask1, mask2):
    mask1_area = torch.count_nonzero(mask1)
    mask2_area = torch.count_nonzero(mask2)
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    _iou = intersection / (mask1_area + mask2_area - intersection)
    return _iou


def iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union


def l1_loss(mask1, mask2):
    return F.l1_loss(mask1, mask2, reduction='mean')


def merge_and_eval(fontID2ret, fontID2GT, fontID2cnt, IoU, L1_Loss, total):
    for k, v in fontID2cnt.items():
        if v != 0:
            continue

        ret = fontID2ret[k]
        gt = fontID2GT[k]

        # merge ret
        # strokes = list(ret.values())
        # strokes.append(np.zeros_like(gt))
        # pred = 255 * reduce(np.multiply, strokes)
        strokes = [_ / 255 for _ in ret.values()]
        # strokes.append(np.ones_like(strokes[0]))
        pred = 255 * reduce(np.multiply, strokes)

        # pred = np.ones_like(gt)
        # # t_print("===> pred.shape: {}".format(pred.shape))
        # for stroke in ret.values():
        #     # t_print("===> ret[stroke_idx].shape: {}".format(ret[stroke_idx].shape))
        #     print("===> stroke.mean(): {}".format(np.mean(stroke)))
        #     print("===> stroke.unique(): {}".format(np.unique(stroke)))
        #     pred *= stroke / 255
        # pred = pred * 255
        # print("===>11111111111 font_id: {}, pred_mean: {}, pred_unique: {}".format(k, np.mean(pred), np.unique(pred)))
        #
        # base = "/home/liulizhao/projects/HWCQA/third_party/fontRL"
        # b = os.path.join(base, 'valIOU')
        # if not os.path.exists(b):
        #     os.makedirs(b)
        # p_pred = '{}-AAAAA-Pred.png'.format(k)
        # p_gt = '{}-AAAAA-GT.png'.format(k)
        # p_pred = os.path.join(b, p_pred)
        # p_gt = os.path.join(b, p_gt)
        # t_print("Saving {}".format(p_pred))
        # t_print("Saving {}".format(p_gt))
        # cv.imwrite(p_pred, pred)
        # cv.imwrite(p_gt, gt)
        # exit()

        # pred: white background, black stroke
        # gt: black background, colorful pixel
        # make pred black background first
        pred = 255 - pred  # 0 background, 255 stroke
        gt = binarize_img(gt)  # 0 background, 255 stroke
        # print("===> np.unique(pred): {}".format(np.unique(pred)))
        # print("===> np.mean(pred): {}".format(np.mean(pred)))
        # print("===> font_id: {}, b_pred_mean: {}, b_gt_mean: {}".format(k, np.mean(pred), np.mean(gt)))
        b_pred, b_gt = pred / 255, gt / 255
        # print("===> font_id: {}, b_pred_mean: {}, b_gt_mean: {}".format(k, np.mean(b_pred), np.mean(b_gt)))
        # exit()
        # p_pred = '{}-BBBBB-Pred.png'.format(k)
        # p_gt = '{}-BBBBB-GT.png'.format(k)
        # p_pred = os.path.join(b, p_pred)
        # p_gt = os.path.join(b, p_gt)
        # t_print("Saving {}".format(p_pred))
        # t_print("Saving {}".format(p_gt))
        # cv.imwrite(p_pred, b_pred * 255)
        # cv.imwrite(p_gt, b_gt * 255)
        # exit()

        b_pred, b_gt = torch.Tensor(b_pred), torch.Tensor(b_gt)
        IoU += binaryMaskIOU(b_pred, b_gt)
        L1_Loss += l1_loss(b_pred, b_gt)
        total += 1
        # print("===> IoU: {}".format(IoU))
        # print("===> L1_Loss: {}".format(L1_Loss))
        # print("===> total: {}".format(total))

        # base = "/home/liulizhao/projects/HWCQA/third_party/fontRL"
        # b = os.path.join(base, 'valIOU')
        # if not os.path.exists(b):
        #     os.makedirs(b)
        # p_pred = '{}-FFFFFFFF-Pred.png'.format(k)
        # p_gt = '{}-FFFFFFFF-GT.png'.format(k)
        # p_pred = os.path.join(b, p_pred)
        # p_gt = os.path.join(b, p_gt)
        # t_print("Saving {}".format(p_pred))
        # cv.imwrite(p_pred, pred * 255)
        # cv.imwrite(p_gt, gt)
        # return

    # update to pop completed evaluation
    fontID2cnt = {k: v for k, v in fontID2cnt.items() if v != 0}
    fontID2ret = {k: v for k, v in fontID2ret.items() if k in fontID2cnt}
    fontID2GT = {k: v for k, v in fontID2GT.items() if k in fontID2cnt}

    # t_print("===> Mean IoU: {}".format(IoU / total))
    # t_print("===> Mean L1_Loss: {}".format(L1_Loss / total))

    return fontID2ret, fontID2GT, fontID2cnt, IoU, L1_Loss, total


@torch.no_grad()
def run_valset(actor_path, mover_path, num_limit=10):
    base = "/home/liulizhao/projects/HWCQA/third_party/fontRL"
    join = os.path.join
    d = FontDataset(img_path=join(base, 'data/font_concat/{}'.format(2)),
                    point_src_path=join(base, 'data/font_10_stroke/{}_point.npy'.format('mean')),
                    point_tgt_path=join(base, 'data/font_10_stroke/{}_point.npy'.format(2)), num_limit=num_limit)
    t_print(len(d))

    channels = 3
    width = 128
    batch_size = 512
    maxstep = 1
    IoU = 0.0
    L1_Loss = 0.0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = ResNet(2 * channels + 3, 18, 50)
    actor.load_state_dict(torch.load(actor_path))
    mover = ResNet_mover(3 * channels, 50, 3)
    mover.load_state_dict(torch.load(mover_path))
    actor = actor.to(device).eval()
    mover = mover.to(device).eval()

    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
    coord = torch.zeros([1, 2, width, width]).float().to(device)
    coord[0, 0, :, :] = torch.linspace(-1, 1, 128)
    coord[0, 1, :, :] = torch.linspace(-1, 1, 128)[..., None]

    dataloader = DataLoader(dataset=d, batch_size=batch_size, shuffle=False, num_workers=8)

    fontID2cnt = d.fontID2cnt
    fontID2ret = {_: {} for _ in fontID2cnt.keys()}
    fontID2GT = {}

    for iter, (src, ref, canvas, point, COL, img_id, stroke_id, add, tgt) in enumerate(tqdm(dataloader)):
        src, ref, canvas, point, COL = src.to(device), ref.to(device), canvas.to(device), point.to(device), COL.to(
            device)

        for i in range(maxstep):
            step = T.float() * i / maxstep
            state = torch.cat((canvas, ref, step.repeat(src.shape[0], 1, 1, 1), coord.repeat(src.shape[0], 1, 1, 1)), 1)
            action = actor(state.float())
            if i == maxstep - 1:
                canvas, _ = decode(point[:, 0, 0:10, 0:2], action, False, point[:, 2, 0:10, 0:2])
                _, point[:, 0, 0:10, 0:2] = decode(point[:, 0, 0:10, 0:2], action, False)
            else:
                canvas, point[:, 0, 0:10, 0:2] = decode(point[:, 0, 0:10, 0:2], action)
            canvas = canvas / 255.

        canvas = canvas * COL / 255.
        state = torch.cat((canvas, src, ref), 1)
        action = mover(state.float())
        canvas, point[:, 0, 0:10, 0:2] = move(point[:, 0, 0:10, 0:2], action, W=width)

        canvas = canvas.detach().cpu().numpy()
        canvas = canvas.transpose((0, 2, 3, 1))
        for i in range(canvas.shape[0]):
            # record gt
            _img_id = img_id[i].item()
            _stroke_id = stroke_id[i].item()
            if _img_id not in fontID2GT:
                fontID2GT[_img_id] = tgt[i].numpy()
            # record pred
            fontID2ret[_img_id][_stroke_id] = 255 - canvas[i]
            # update count
            fontID2cnt[_img_id] -= 1
            # t_print(canvas[i].shape)
            # b = os.path.join(base, 'valIOU')
            # if not os.path.exists(b):
            #     os.makedirs(b)
            # p = '{}-XXXX.png'.format(add[i])
            # p = os.path.join(b, p)
            # t_print("Saving {}".format(p))
            # cv.imwrite(p, 255 - canvas[i])
            # print("\n===> fontID2cnt: {}".format(fontID2cnt))
        fontID2ret, fontID2GT, fontID2cnt, IoU, L1_Loss, total = merge_and_eval(fontID2ret, fontID2GT, fontID2cnt, IoU,
                                                                                L1_Loss, total)

    t_print("===> Mean IoU: {}".format(IoU / total))
    t_print("===> Mean L1_Loss: {}".format(L1_Loss / total))


@torch.no_grad()
def bbox_validation(val_dataset, val_dataloader, actor, mover, device, maxstep, width=128):
    IoU = 0.0
    L1_Loss = 0.0
    total = 0

    d = val_dataset
    dataloader = val_dataloader

    fontID2cnt = copy.deepcopy(d.fontID2cnt)
    fontID2ret = {_: {} for _ in fontID2cnt.keys()}  # TODO
    fontID2GT = {}

    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
    coord = torch.zeros([1, 2, width, width]).float().to(device)
    coord[0, 0, :, :] = torch.linspace(-1, 1, 128)
    coord[0, 1, :, :] = torch.linspace(-1, 1, 128)[..., None]

    for iter, (src, ref, canvas, point, COL, img_id, stroke_id, add, tgt) in enumerate(tqdm(dataloader)):
        src, ref, canvas, point, COL = src.to(device), ref.to(device), canvas.to(device), point.to(device), COL.to(
            device)

        for i in range(maxstep):
            step = T.float() * i / maxstep
            state = torch.cat((canvas, ref, step.repeat(src.shape[0], 1, 1, 1), coord.repeat(src.shape[0], 1, 1, 1)), 1)
            action = actor(state.float())
            if i == maxstep - 1:
                canvas, _ = decode(point[:, 0, 0:10, 0:2], action, False, point[:, 2, 0:10, 0:2])
                _, point[:, 0, 0:10, 0:2] = decode(point[:, 0, 0:10, 0:2], action, False)
            else:
                canvas, point[:, 0, 0:10, 0:2] = decode(point[:, 0, 0:10, 0:2], action)
            canvas = canvas / 255.

        canvas = canvas * COL / 255.
        state = torch.cat((canvas, src, ref), 1)
        action = mover(state.float())
        # print("=======> action: {}".format(action))
        canvas, point[:, 0, 0:10, 0:2] = move(point[:, 0, 0:10, 0:2], action, W=width)

        canvas = canvas.detach().cpu().numpy()
        canvas = canvas.transpose((0, 2, 3, 1))
        # print("=======> np.sum(canvas): {}".format(np.sum(canvas)))
        # print("=======> np.mean(canvas): {}".format(np.mean(canvas)))
        # break
        for i in range(canvas.shape[0]):
            _img_id = img_id[i].item()
            _stroke_id = stroke_id[i].item()
            # record gt
            if _img_id not in fontID2GT:
                fontID2GT[_img_id] = tgt[i].numpy()
            # record pred
            fontID2ret[_img_id][_stroke_id] = 255 - canvas[i]
            # update count
            fontID2cnt[_img_id] -= 1

        fontID2ret, fontID2GT, fontID2cnt, IoU, L1_Loss, total = merge_and_eval(fontID2ret, fontID2GT, fontID2cnt, IoU,
                                                                                L1_Loss, total)

    # t_print("===> Mean IoU: {}".format(IoU / total))
    # t_print("===> Mean L1_Loss: {}".format(L1_Loss / total))
    if total == 0:
        return 0., 0.
    return IoU / total, L1_Loss / total


class ParallelGaussianBlur(GaussianBlur):
    def forward(self, img1, img2, img3):
        sigma = self.get_params(self.sigma[0], self.sigma[1])

        img1 = TF.gaussian_blur(img1, self.kernel_size, [sigma, sigma])
        img2 = TF.gaussian_blur(img2, self.kernel_size, [sigma, sigma])
        img3 = TF.gaussian_blur(img3, self.kernel_size, [sigma, sigma])

        return img1, img2, img3


class ParallelColorJitter(ColorJitter):
    def forward(self, img1, img2, img3):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img1 = TF.adjust_brightness(img1, brightness_factor)
                img2 = TF.adjust_brightness(img2, brightness_factor)
                img3 = TF.adjust_brightness(img3, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img1 = TF.adjust_contrast(img1, contrast_factor)
                img2 = TF.adjust_contrast(img2, contrast_factor)
                img3 = TF.adjust_contrast(img3, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img1 = TF.adjust_saturation(img1, saturation_factor)
                img2 = TF.adjust_saturation(img2, saturation_factor)
                img3 = TF.adjust_saturation(img3, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img1 = TF.adjust_hue(img1, hue_factor)
                img2 = TF.adjust_hue(img2, hue_factor)
                img3 = TF.adjust_hue(img3, hue_factor)

        return img1, img2, img3


@torch.no_grad()
def augmentation(img1, img2, img3):
    # print("===> img1.size(): {}, img2.size(): {}, img3.size(): {}".format(
    #     img1.size(), img2.size(), img3.size()
    # ))
    img1, img2, img3 = ParallelGaussianBlur(kernel_size=5)(img1, img2, img3)
    img1, img2, img3 = ParallelColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(img1, img2, img3)
    return img1, img2, img3


if __name__ == '__main__':
    base = "/home/liulizhao/projects/HWCQA/third_party/fontRL"
    join = os.path.join

    # _actor_path = 'model/{}/Paint-run1/actor.pkl'.format(2)
    # _mover_path = "model/2/Paint-run-hwBS96/mover/mover-10728.pkl"
    # _mover_path = "model/2/Paint-run-hwBS96/mover/mover-7272.pkl"
    # _mover_path = "model/2/Paint-run-ImageNetBS98Val/mover/mover-149.pkl"

    # formal exp below
    # only on actor path
    _actor_path = "model/2/Paint-run3/actor-4801.pkl"

    # for Paint-run-ImageNetBS98Val-123
    # for Paint-run-noBS96Val-139
    # for Paint-run-hwBS96ValRecover-118
    _mover_paths = [
        "model/2/Paint-run-ImageNetBS98Val/mover/mover-123.pkl",
        "model/2/Paint-run-noBS96Val/mover/mover-139.pkl",
        "model/2/Paint-run-hwBS96ValRecover/mover/mover-118.pkl",
    ]

    """ sealed with the following results
    ssh://liulizhao@gpu022.scut-smil.cn:22/mnt/cephfs/home/liulizhao/anaconda3/envs/torch1.6/bin/python -u /home/liulizhao/projects/HWCQA/third_party/fontRL/mover_dataset.py
    ===> running with actor=model/2/Paint-run3/actor-4801.pkl, mover=model/2/Paint-run-ImageNetBS98Val/mover/mover-123.pkl
    2022-08-15 19:27:20 71826
    100%|█████████████████████████████████████████| 141/141 [11:53<00:00,  5.06s/it]
    2022-08-15 19:39:18 ===> Mean IoU: 0.43109527230262756
    2022-08-15 19:39:18 ===> Mean L1_Loss: 0.12012417614459991
    ===> running with actor=model/2/Paint-run3/actor-4801.pkl, mover=model/2/Paint-run-noBS96Val/mover/mover-139.pkl
    2022-08-15 19:39:18 71826
    100%|█████████████████████████████████████████| 141/141 [11:43<00:00,  4.99s/it]
    2022-08-15 19:51:09 ===> Mean IoU: 0.41227149963378906
    2022-08-15 19:51:09 ===> Mean L1_Loss: 0.12573352456092834
    ===> running with actor=model/2/Paint-run3/actor-4801.pkl, mover=model/2/Paint-run-hwBS96ValRecover/mover/mover-118.pkl
    2022-08-15 19:51:09 71826
    100%|█████████████████████████████████████████| 141/141 [10:31<00:00,  4.48s/it]
    2022-08-15 20:01:44 ===> Mean IoU: 0.4130800664424896
    2022-08-15 20:01:44 ===> Mean L1_Loss: 0.12536656856536865
    """

    for _mover_path in _mover_paths:
        print("===> running with actor={}, mover={}".format(_actor_path, _mover_path))
        run_valset(actor_path=join(base, _actor_path),
                   mover_path=join(base, _mover_path),
                   num_limit=-1)
