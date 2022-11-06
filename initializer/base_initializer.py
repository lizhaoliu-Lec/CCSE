import copy
import torch

from detectron2.config import CfgNode, get_cfg

from common.config import read_config_from_file, save_config
from common.logging import get_logger
from common.utils import get_output_dir, setup_gpu, setup_seed


def merge_dict(dict1, dict2):
    return {**dict1, **dict2}


class BaseInitializer:
    """
    Base initializer that
    1) sets up and check config
    2) sets up output dir
    4) sets up logger
    3) save config
    5) sets up gpu visibility
    6) sets up seed
    7) others things should be done in the inherited initializer
    """
    REQUIRED_FIELDS = ['OUTPUT_DIR', 'GPU_IDS', 'OUTPUT_ID', 'SAVE_LOG', 'SEED', 'VERSION', 'DEBUG']
    MORE_REQUIRED_FIELDS = []
    # dict (key, value)
    REQUIRED_FIELDS_WITH_DEFAULT = {
        'OUTPUT_DIR': './output',
        'SEED': 1234,
        'SAVE_LOG': True,
        'VERSION': 2,
        'DEBUG': False,
    }
    MORE_REQUIRED_FIELDS_WITH_DEFAULT = {}

    def __init__(self, config_filepath):
        self.config_filepath = config_filepath
        # do something before check and setup
        self.do_something_before_check_field()

        base_config = self.get_default_config()

        # setup config
        base_config.merge_from_file(config_filepath)
        self.config = base_config

        # check config
        self.check_required_field(self.config)

        # setup output dir
        self.config.OUTPUT_DIR = get_output_dir(self.config.OUTPUT_DIR, self.config.OUTPUT_ID)

        # set up logger
        config = self.config
        logger = get_logger(output_dir=config.OUTPUT_DIR,
                            output_id=config.OUTPUT_ID,
                            save_log=config.SAVE_LOG)
        logger.info("Using config: \n{}".format(config))
        # save config
        config_save_path = save_config(self.config)
        logger.info("*** [Path to delete is here!!!] Saving config file in {} ***".format(config_save_path))

        # set gpu visibility
        # if not torch.cuda.is_available():
        #     self.logger.info('CUDA is not available. Using CPU, this will be slow')
        # else:
        self.config.GPU_IDS = setup_gpu(gpu_ids=config.GPU_IDS)
        logger.info("Using {} gpu(s)".format(torch.cuda.device_count()))

        # set seed
        setup_seed(config.SEED)
        if config.SEED >= 0:
            logger.info("Fixing random seed to {}".format(config.SEED))

    @classmethod
    def get_default_config(cls):
        detectron2_config = get_cfg()
        detectron2_config = cls.prepare_default_field(detectron2_config)
        return detectron2_config

    @classmethod
    def check_required_field(cls, config):

        TOTAL_FIELDS_WITH_DEFAULT_DICT = copy.deepcopy(cls.REQUIRED_FIELDS_WITH_DEFAULT)
        TOTAL_FIELDS_WITH_DEFAULT_DICT.update(cls.MORE_REQUIRED_FIELDS_WITH_DEFAULT)

        # check required field
        for field in cls.REQUIRED_FIELDS + cls.MORE_REQUIRED_FIELDS:
            if config.get(field) is None and field not in TOTAL_FIELDS_WITH_DEFAULT_DICT:
                # sometime some field's default is None
                raise ValueError('`{}` is not provided in config file'.format(field))

    @classmethod
    def prepare_default_field(cls, config):
        # prepare extra default field and value other than the fields in detectron2
        TOTAL_FIELDS_WITH_DEFAULT_DICT = merge_dict(cls.REQUIRED_FIELDS_WITH_DEFAULT,
                                                    cls.MORE_REQUIRED_FIELDS_WITH_DEFAULT)
        for field in cls.REQUIRED_FIELDS + cls.MORE_REQUIRED_FIELDS:
            if field not in config:
                value = None
                if field in TOTAL_FIELDS_WITH_DEFAULT_DICT:
                    value = TOTAL_FIELDS_WITH_DEFAULT_DICT[field]
                config.setdefault(field, value)
        return config

    @classmethod
    def dynamic_modify_field_before_parsing(cls, _SET_DEFAULT_FIELDS_WITH_VALUES, _ADD_DEFAULT_FIELDS_WITH_VALUES):
        # dynamic modify config fields before passing

        # set value for already exist fields
        for k, v in _SET_DEFAULT_FIELDS_WITH_VALUES.items():
            if k not in cls.MORE_REQUIRED_FIELDS or k not in cls.MORE_REQUIRED_FIELDS_WITH_DEFAULT:
                raise KeyError
            cls.MORE_REQUIRED_FIELDS_WITH_DEFAULT[k] = v

        # add value for not exist fields
        for k, v in _ADD_DEFAULT_FIELDS_WITH_VALUES.items():
            if k in cls.MORE_REQUIRED_FIELDS or k in cls.MORE_REQUIRED_FIELDS_WITH_DEFAULT:
                raise ValueError
            cls.MORE_REQUIRED_FIELDS.append(k)
            cls.MORE_REQUIRED_FIELDS_WITH_DEFAULT[k] = v

    def do_something_before_check_field(self):
        pass
