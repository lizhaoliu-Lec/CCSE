import logging
import os
import random
from os.path import basename

import cv2
from detectron2.data import DatasetCatalog
from module.instance.predictor import DefaultPredictorWithProposal as DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

from common.cmd_parser import parse_cmd_arg
from common.utils import plt_show, join, mkdirs_if_not_exist
from initializer.instance_initializer import InferenceInstanceInitializer
from module.instance.evaluator import EnhancedCOCOEvaluator
from module.sparse_rcnn.trainer import Trainer as TrainerWithoutHorizontalFlip
from pre_process.pre_process import read_to_gray_scale

from detectron2 import model_zoo


def main(init: InferenceInstanceInitializer):
    # this need to set in config yaml file
    # path to the model we just trained
    # config.MODEL.WEIGHTS = join('output/debug/20210608.232202', "model_final.pth")
    # set a custom testing threshold
    # config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    logger = logging.getLogger('detectron2')

    config = init.config
    dataset_metadata = init.dataset_metadata
    predictor = DefaultPredictor(config)
    visualize_prediction_path = join(config.OUTPUT_DIR, 'visualize_prediction')
    visualize_input_path = join(config.OUTPUT_DIR, 'visualize_input')

    mkdirs_if_not_exist(visualize_prediction_path)
    mkdirs_if_not_exist(visualize_input_path)

    # predict and visualize the image provided in image paths
    if config.IMAGE_PATHS is not None:
        for image_path in tqdm(config.IMAGE_PATHS):
            im = read_to_gray_scale(image_path)
            plt_show(im[:, :, ::-1], save_filename=join(visualize_input_path, basename(image_path)))

            outputs = predictor(im)

            # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for
            # specification
            # print(outputs["instances"].pred_classes)
            # print(outputs["instances"].pred_boxes)

            # We can use `Visualizer` to draw the predictions on the image.
            v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=5.0)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt_show(out.get_image()[:, :, ::-1], save_filename=join(visualize_prediction_path, basename(image_path)))

    # evaluate the model and get bbox, segm metrics
    evaluate(init=init, config=config)

    if config.VIS_DATASET_RESULT:
        # visualize prediction in dataset
        visualize_prediction_in_datasets(config=config,
                                         dataset_name=init.val_set_name,
                                         dataset_metadata=dataset_metadata,
                                         num_vis=None,
                                         logger=logger)


def evaluate(init, config):
    evaluator = EnhancedCOCOEvaluator(init.val_set_name, config.TASKS, False, output_dir=config.OUTPUT_DIR)

    trainer = TrainerWithoutHorizontalFlip(config)
    trainer.resume_or_load()
    trainer.test(config, model=trainer.model, evaluators=[evaluator])


def visualize_prediction(predictor, dataset_name, dataset_metadata, OUTPUT_DIR, logger, num_vis=10,
                         num_vis_proposal=20):
    visualize_prediction_path = join(OUTPUT_DIR, 'visualize_dataset_prediction')
    logger.info("Saving prediction visualization results in {}".format(visualize_prediction_path))
    if not os.path.exists(visualize_prediction_path):
        os.makedirs(visualize_prediction_path)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    if num_vis is None:
        vis_collect = dataset_dicts
    else:
        vis_collect = random.sample(dataset_dicts, num_vis)
    for d in tqdm(vis_collect):
        im = cv2.imread(d["file_name"])
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=5.0,
                       # instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt_show(out.get_image()[:, :, ::-1], join(visualize_prediction_path,
                                                   os.path.basename(d['file_name'])))


def visualize_prediction_in_datasets(config, dataset_name, dataset_metadata, logger, num_vis=10):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    config.MODEL.WEIGHTS = config.MODEL.WEIGHTS  # path to the model we just trained
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set a custom testing threshold
    predictor = DefaultPredictor(config)
    visualize_prediction(predictor, dataset_name, dataset_metadata, config.OUTPUT_DIR, logger, num_vis=num_vis,
                         num_vis_proposal=config.NUM_VIS_PROPOSAL)


if __name__ == '__main__':
    args = parse_cmd_arg()

    initializer = InferenceInstanceInitializer(args.config)
    main(initializer)
