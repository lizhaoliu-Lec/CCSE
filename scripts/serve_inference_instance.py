import logging
import os
import random
from os.path import basename

import cv2
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

from common.cmd_parser import parse_cmd_arg
from common.utils import plt_show, join, mkdirs_if_not_exist
from initializer.instance_initializer import InferenceInstanceInitializer
from module.instance.evaluator import EnhancedCOCOEvaluator
from module.instance.trainer import TrainerWithoutHorizontalFlip
from pre_process.pre_process import read_to_gray_scale

from detectron2 import model_zoo

from flask import Flask, request, jsonify

app = Flask(__name__)

predictor: DefaultPredictor = None
dataset_metadata = None


def model_init(init: InferenceInstanceInitializer):
    # this need to set in config yaml file
    # path to the model we just trained
    # config.MODEL.WEIGHTS = join('output/debug/20210608.232202', "model_final.pth")
    # set a custom testing threshold
    # config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    global dataset_metadata, predictor
    config = init.config
    dataset_metadata = init.dataset_metadata

    predictor = DefaultPredictor(config)


def real_check(image_path, output_path):
    im = read_to_gray_scale(image_path)
    # plt_show(im[:, :, ::-1], save_filename=join(visualize_input_path, basename(image_path)))

    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for
    # specification
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=5.0)
    instances = outputs["instances"].to("cpu")
    out = v.draw_instance_predictions(instances)
    plt_show(out.get_image()[:, :, ::-1], save_filename=output_path)

    dict_output = {
        "image_size": instances.image_size,
        "pred_boxes": instances.pred_boxes.tensor.numpy().tolist(),
        "scores": instances.scores.numpy().tolist(),
        "pred_classes": instances.pred_classes.numpy().tolist(),
        "pred_masks": instances.pred_masks.numpy().tolist()
    }
    return dict_output


@app.route('/check', methods=['POST'])
def check():
    logging.info("Receive request")
    if request.method == 'POST':
        # we will get the file from the request
        image_path = request.form['image_path']
        output_path = request.form['output_path']
        logging.info("Request image_path: {0}".format(image_path))
        logging.info("Request output_path: {0}".format(output_path))
        outputs = real_check(image_path, output_path)
        logging.info(outputs)
        return jsonify(outputs)


if __name__ == '__main__':
    args = parse_cmd_arg()

    initializer = InferenceInstanceInitializer(args.config)
    model_init(initializer)
    app.run()
