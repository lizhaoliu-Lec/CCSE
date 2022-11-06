from detectron2.utils.logger import setup_logger
from common.utils import Singleton, join

__all__ = ['set_logging', 'get_logger']


@Singleton
def set_logging(output_dir=None, output_id=None, save_log=False):
    # output_dir='./output', output_id='debug'
    if output_dir is not None and output_id is not None and save_log:
        log_filepath = join(output_dir, 'log.txt')
        logger = setup_logger(output=log_filepath)
        logger.info("Saving logs to {}".format(log_filepath))
    else:
        logger = setup_logger()
    return logger


@Singleton
def get_logger(output_dir=None, output_id=None, save_log=False):
    return set_logging(output_dir, output_id, save_log)
    # return logging.getLogger('detectron2')
