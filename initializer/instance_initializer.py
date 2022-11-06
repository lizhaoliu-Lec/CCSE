from detectron2.data import MetadataCatalog

from dataset.coco_format_instance import register_coco_format_instance, register_coco_format_reference_instance, \
    register_coco_format_instance_with_reference
from initializer.base_initializer import BaseInitializer


class InstanceInitializer(BaseInitializer):
    """
    Instance initializer that
    1) sets instance dataset
    2) sets datasets' meta data
    """
    MORE_REQUIRED_FIELDS = [
        'DATA_ROOT', 'DATASET_NAME',
        'ANNOTATION_DIR', 'TRAIN_IMAGE_DIR', 'VAL_IMAGE_DIR',
        'TRAIN_JSON_NAME', 'VAL_JSON_NAME',
        'NUM_VIS', 'TASKS',
    ]
    MORE_REQUIRED_FIELDS_WITH_DEFAULT = {
        'DATASET_NAME': 'chinese_stroke_2021',
        'ANNOTATION_DIR': 'annotations',
        'TRAIN_IMAGE_DIR': 'train2021',
        'VAL_IMAGE_DIR': 'val2021',
        'TRAIN_JSON_NAME': 'instances_train2021.json',
        'VAL_JSON_NAME': 'instances_val2021.json',
        'NUM_VIS': 10,
        'TASKS': ["bbox", "segm"]
    }

    def __init__(self, config_filepath):
        super().__init__(config_filepath=config_filepath)
        self.train_set_name = None
        self.val_set_name = None
        self.dataset_metadata = None
        self.launch_calling()

    def launch_calling(self):
        try:
            # (1) init dataset
            config = self.config
            train_set_name, val_set_name = register_coco_format_instance(data_root=config.DATA_ROOT,
                                                                         dataset_name=config.DATASET_NAME,
                                                                         annotation_dir=config.ANNOTATION_DIR,
                                                                         train_image_dir=config.TRAIN_IMAGE_DIR,
                                                                         val_image_dir=config.VAL_IMAGE_DIR,
                                                                         train_json_name=config.TRAIN_JSON_NAME,
                                                                         val_json_name=config.VAL_JSON_NAME)

            self.train_set_name = train_set_name
            self.val_set_name = val_set_name

            self.dataset_metadata = MetadataCatalog.get(train_set_name)
        except:
            # wrap it, otherwise calling it multiple times will cause dataset already registered exception
            pass


class InferenceInstanceInitializer(InstanceInitializer):
    MORE_REQUIRED_FIELDS = InstanceInitializer.MORE_REQUIRED_FIELDS + ['IMAGE_PATHS', 'VIS_DATASET_RESULT',
                                                                       'NUM_VIS_PROPOSAL']
    InstanceInitializer.MORE_REQUIRED_FIELDS_WITH_DEFAULT.update({
        'IMAGE_PATHS': None,
        'VIS_DATASET_RESULT': False,
        'NUM_VIS_PROPOSAL': 30,
    })


class ReferenceInstanceInitializer(InstanceInitializer):
    MORE_REQUIRED_FIELDS = InstanceInitializer.MORE_REQUIRED_FIELDS + [
        'REFERENCE_DATASET_NAME', 'REFERENCE_DATA_ROOT',
        'REFERENCE_ANNOTATION_DIR', 'REFERENCE_IMAGE_DIR',
        'REFERENCE_JSON_NAME',
    ]
    InstanceInitializer.MORE_REQUIRED_FIELDS_WITH_DEFAULT.update({
        'REFERENCE_ANNOTATION_DIR': 'annotations',
    })

    def launch_calling(self):
        try:
            # (1) init dataset
            config = self.config
            train_set_name, val_set_name = register_coco_format_instance_with_reference(
                data_root=config.DATA_ROOT,
                dataset_name=config.DATASET_NAME,
                annotation_dir=config.ANNOTATION_DIR,
                train_image_dir=config.TRAIN_IMAGE_DIR,
                val_image_dir=config.VAL_IMAGE_DIR,
                train_json_name=config.TRAIN_JSON_NAME,
                val_json_name=config.VAL_JSON_NAME)

            self.train_set_name = train_set_name
            self.val_set_name = val_set_name

            self.dataset_metadata = MetadataCatalog.get(train_set_name)

            # (2) init reference dataset
            reference_dataset_name = register_coco_format_reference_instance(
                reference_data_root=config.REFERENCE_DATA_ROOT,
                reference_dataset_name=config.REFERENCE_DATASET_NAME,
                reference_annotation_dir=config.REFERENCE_ANNOTATION_DIR,
                reference_image_dir=config.REFERENCE_IMAGE_DIR,
                reference_json_name=config.REFERENCE_JSON_NAME,
            )
            self.reference_dataset_name = reference_dataset_name

        except:
            # wrap it, otherwise calling it multiple times will cause dataset already registered exception
            pass
