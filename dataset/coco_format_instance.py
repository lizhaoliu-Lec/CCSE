import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from common.utils import join
from module.reference_sparse_rcnn.datasets import load_coco_with_reference_json


def register_coco_instances_with_reference(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_with_reference_json(json_file, image_root, name))

    MetadataCatalog.get(name).thing_classes = ['wangou', 'na', 'ti', 'pie', 'piezhe', 'piedian', 'xiegouhuowogou',
                                               'heng', 'hengzhe', 'hengzhezhehuohengzhewan', 'hengzhezhezhe',
                                               'hengzhezhezhegouhuohengpiewangou', 'hengzhezhepie', 'hengzheti',
                                               'hengzhegou', 'hengpiehuohenggou', 'hengxiegou', 'dian', 'shu', 'shuwan',
                                               'shuwangou', 'shuzhezhegou', 'shuzhepiehuoshuzhezhe', 'shuti', 'shugou']

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_coco_format_instance_with_reference(data_root,
                                                 dataset_name='chinese_stroke_2021',
                                                 annotation_dir='annotations',
                                                 train_image_dir='train2021',
                                                 val_image_dir='val2021',
                                                 train_json_name='instances_train2021.json',
                                                 val_json_name='instances_val2021.json'):
    # data_root = 'resources/chinese_stroke_2021'
    annotation_root = join(data_root, annotation_dir)
    train_image_root = join(data_root, train_image_dir)
    val_image_root = join(data_root, val_image_dir)
    train_set_name, val_set_name = "{}_train".format(dataset_name), "{}_val".format(dataset_name)
    register_coco_instances_with_reference(train_set_name, {},
                                           join(annotation_root, train_json_name),
                                           train_image_root)
    register_coco_instances_with_reference(val_set_name, {},
                                           join(annotation_root, val_json_name),
                                           val_image_root)
    return train_set_name, val_set_name


def register_coco_format_instance(data_root,
                                  dataset_name='chinese_stroke_2021',
                                  annotation_dir='annotations',
                                  train_image_dir='train2021',
                                  val_image_dir='val2021',
                                  train_json_name='instances_train2021.json',
                                  val_json_name='instances_val2021.json'):
    # data_root = 'resources/chinese_stroke_2021'
    annotation_root = join(data_root, annotation_dir)
    train_image_root = join(data_root, train_image_dir)
    val_image_root = join(data_root, val_image_dir)
    train_set_name, val_set_name = "{}_train".format(dataset_name), "{}_val".format(dataset_name)
    register_coco_instances(train_set_name, {},
                            join(annotation_root, train_json_name),
                            train_image_root)
    register_coco_instances(val_set_name, {},
                            join(annotation_root, val_json_name),
                            val_image_root)
    return train_set_name, val_set_name


def register_coco_format_reference_instance(reference_data_root,
                                            reference_dataset_name='chinese_print_stroke_2021',
                                            reference_annotation_dir='annotations',
                                            reference_image_dir='train2021',
                                            reference_json_name='instances_train2021.json'):
    # data_root = 'resources/chinese_stroke_2021'
    annotation_root = join(reference_data_root, reference_annotation_dir)
    image_root = join(reference_data_root, reference_image_dir)
    register_coco_instances(reference_dataset_name, {},
                            join(annotation_root, reference_json_name),
                            image_root)
    return reference_dataset_name
