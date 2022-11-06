# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from common.utils import mkdirs_if_not_exist


class Tensorboard(SummaryWriter):
    def __init__(self, logdir, **kwargs):
        mkdirs_if_not_exist(logdir)
        super().__init__(logdir, **kwargs)

    def __del__(self):
        self.close()


if __name__ == '__main__':
    def run_tensorboard():
        tensorboard = Tensorboard(logdir='./tmp')
        for i in range(10000):
            if i == 100:
                raise ValueError


    run_tensorboard()
