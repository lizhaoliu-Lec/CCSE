import yaml
from detectron2.config import get_cfg

__all__ = ['get_empty_cfg', 'save_config', 'read_config_from_file']

from common.utils import join


def get_empty_cfg():
    cfg = get_cfg()
    cfg.clear()
    return cfg


def save_config(config, filename='config.yaml'):
    config_yaml_string = config.dump()
    config_save_path = join(config.OUTPUT_DIR, filename)
    with open(config_save_path, 'w') as f:
        f.write(config_yaml_string)
    return config_save_path


def read_config_from_file(filepath):
    cfg = get_empty_cfg()
    # logger.debug("===> before loaded config: \n{}".format(cfg))
    # logger.info("Loading config from file: {}".format(filepath))
    with open(filepath, 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config_yaml_string = yaml.dump(config_yaml)
    # must return a value, the load_cfg function is not a inplace operation
    cfg = cfg.load_cfg(config_yaml_string)
    # logger.debug("===> loaded config: \n{}".format(cfg))
    return cfg


if __name__ == '__main__':
    # get_output_dir("./output/chinese_stroke_50000")
    read_config_from_file('./output/debug/20210608.161148/config.yaml')
