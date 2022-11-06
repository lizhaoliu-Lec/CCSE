import logging
import os
import cv2
import random

from detectron2.engine import DefaultPredictor, launch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator

from common.cmd_parser import parse_cmd_arg
from pre_process.pre_process import read_to_gray_scale
from module.instance.trainer import TrainerWithoutHorizontalFlip
from common.utils import plt_show, join

from initializer.instance_initializer import InstanceInitializer


def main(init: InstanceInitializer):
    config = init.config
    logger = logging.getLogger('detectron2')

    # launch again to support multi-process!
    init.launch_calling()

    # visualize the dataset
    visualize_dataset(init.train_set_name,
                      init.dataset_metadata,
                      config.OUTPUT_DIR,
                      logger,
                      num_vis=config.NUM_VIS)

    # train and evaluate the model
    train_and_evaluate(init, config)

    # visualize the prediction
    # predictor = DefaultPredictor(config)
    # visualize_prediction(predictor, init.val_set_name, dataset_metadata=init.dataset_metadata,
    #                      OUTPUT_DIR=config.OUTPUT_DIR, logger=logger, num_vis=config.NUM_VIS)
    visualize(config, init.val_set_name, dataset_metadata=init.dataset_metadata, logger=logger, num_vis=config.NUM_VIS)


def visualize_dataset(dataset_name, dataset_metadata, OUTPUT_DIR, logger, num_vis=10):
    visualize_dataset_path = join(OUTPUT_DIR, 'visualize_dataset')
    logger.info("Saving dataset visualization results in {}".format(visualize_dataset_path))
    if not os.path.exists(visualize_dataset_path):
        os.makedirs(visualize_dataset_path, exist_ok=True)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, num_vis):
        print('===> d["file_name"]: {}'.format(d["file_name"]))
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=5.0)
        out = visualizer.draw_dataset_dict(d)
        plt_show(out.get_image()[:, :, ::-1], join(visualize_dataset_path, os.path.basename(d['file_name'])))


def visualize_prediction(predictor, dataset_name, dataset_metadata, OUTPUT_DIR, logger, num_vis=10):
    visualize_prediction_path = join(OUTPUT_DIR, 'visualize_prediction')
    logger.info("Saving prediction visualization results in {}".format(visualize_prediction_path))
    if not os.path.exists(visualize_prediction_path):
        os.makedirs(visualize_prediction_path, exist_ok=True)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, num_vis):
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
        plt_show(out.get_image()[:, :, ::-1], join(visualize_prediction_path, os.path.basename(d['file_name'])))


def train_and_evaluate(init, config):
    evaluator = COCOEvaluator(init.val_set_name, config.TASKS, False, output_dir=config.OUTPUT_DIR)

    trainer = TrainerWithoutHorizontalFlip(config)
    trainer.resume_or_load(resume=False)
    trainer.train()
    trainer.test(config, model=trainer.model, evaluators=[evaluator])


def visualize(config, dataset_name, dataset_metadata, logger, num_vis=10):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    config.MODEL.WEIGHTS = join(config.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(config)
    visualize_prediction(predictor, dataset_name, dataset_metadata, config.OUTPUT_DIR, logger, num_vis=num_vis)


if __name__ == '__main__':
    args = parse_cmd_arg()

    initializer = InstanceInitializer(args.config)
    initializer.logger = None
    num_gpu = len(initializer.config.GPU_IDS)

    launch(main_func=main, num_gpus_per_machine=num_gpu, dist_url='auto', args=(initializer,))
