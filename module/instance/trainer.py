from detectron2.data import build_detection_train_loader, DatasetMapper
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from common.utils import join


class TrainerWithoutHorizontalFlip(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name=dataset_name,
                             tasks=cfg.TASKS,
                             distributed=False,
                             output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_mapper = DatasetMapper(cfg, is_train=True,
                                       augmentations=[
                                           T.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                                sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)
                                       ])
        dataloader = build_detection_train_loader(cfg,
                                                  mapper=dataset_mapper)

        return dataloader
