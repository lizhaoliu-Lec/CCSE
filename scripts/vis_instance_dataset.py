import logging
import os
import cv2
import random
import numpy as np
from detectron2.structures import BoxMode

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog

from common.cmd_parser import parse_cmd_arg
from common.utils import plt_show, join

from initializer.instance_initializer import InstanceInitializer


def draw_dataset_dict(_visualizer, dic):
    """
    Draw annotations/segmentaions in Detectron2 Dataset format.

    Args:
        _visualizer (Visualizer): Visualizer in detectron2
        dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

    Returns:
        output (VisImage): image object with visualizations.
    """
    annos = dic.get("annotations", None)
    if annos:
        if "segmentation" in annos[0]:
            masks = [x["segmentation"] for x in annos]
        else:
            masks = None
        if "keypoints" in annos[0]:
            keypts = [x["keypoints"] for x in annos]
            keypts = np.array(keypts).reshape(len(annos), -1, 3)
        else:
            keypts = None

        boxes = [
            BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
            if len(x["bbox"]) == 4
            else x["bbox"]
            for x in annos
        ]

        colors = None
        category_ids = [x["category_id"] for x in annos]
        if _visualizer._instance_mode == ColorMode.SEGMENTATION and _visualizer.metadata.get("thing_colors"):
            colors = [
                _visualizer._jitter([x / 255 for x in _visualizer.metadata.thing_colors[c]])
                for c in category_ids
            ]
        _visualizer.overlay_instances(
            labels=None, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
        )

    return _visualizer.output


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def visualize_dataset(dataset_name, dataset_metadata, OUTPUT_DIR, logger, num_vis=10):
    visualize_dataset_path = join(OUTPUT_DIR, 'visualize_dataset')
    logger.info("Saving dataset visualization results in {}".format(visualize_dataset_path))
    if not os.path.exists(visualize_dataset_path):
        os.makedirs(visualize_dataset_path)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # expected_name = '0000001486.jpg'
    # expected_name = '0000001645.jpg'
    # expected_name = '0000003568.jpg'
    expected_name = '0000000288.jpg'
    # expected_name = 'ChineseStroke_train_000000002088.jpg'

    _dataset_dicts = [_ for _ in dataset_dicts if expected_name in _["file_name"]]

    # for d in random.sample(dataset_dicts, num_vis):
    for d in _dataset_dicts:
        img = cv2.imread(d["file_name"])

        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=7.0)
        # save original image first
        plt_show(visualizer.get_output().get_image()[:, :, ::-1],
                 join(visualize_dataset_path, 'original_' + os.path.basename(d['file_name'])))

        # the we save the annotated one
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=7.0)
        # out = visualizer.draw_dataset_dict(d)
        out = draw_dataset_dict(visualizer, d)
        plt_show(out.get_image()[:, :, ::-1], join(visualize_dataset_path, os.path.basename(d['file_name'])))


def main(init: InstanceInitializer):
    config = init.config
    logger = logging.getLogger('detectron2')

    # visualize the dataset
    visualize_dataset(init.train_set_name,
                      init.dataset_metadata,
                      config.OUTPUT_DIR,
                      logger,
                      num_vis=config.NUM_VIS)


if __name__ == '__main__':
    args = parse_cmd_arg()

    initializer = InstanceInitializer(args.config)
    main(initializer)
