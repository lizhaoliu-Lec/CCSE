import os
import time
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import socket
from shlex import quote
import sys
import zipfile
from glob import glob
import shutil

plt.rcParams['font.sans-serif'] = ['SimHei']  # to show chinese
plt.rcParams['axes.unicode_minus'] = False  # to show plus minors

__all__ = ['join', 'check_if_file_exists', 'check_if_has_required_args',
           'mkdirs_if_not_exist', 'Singleton',
           'plt_show', 'join', 'get_output_dir', 'setup_gpu', 'setup_seed', 'basename']

join = os.path.join

basename = os.path.basename


def check_if_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File `%s` not found" % file_path)


def check_if_has_required_args(_dict: dict, keys: list):
    for k in keys:
        if k not in _dict:
            raise ValueError('Required arguments `%s` not found.' % k)


def mkdirs_if_not_exist(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=False)


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton


def plt_show(img, save_filename=None):
    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    if save_filename is None:
        plt.show()
    else:
        plt.savefig(save_filename, dpi=300)
    plt.close()


def get_output_dir(output_dir, output_id):
    time_string = time.strftime("%Y%m%d.%H%M%S", time.localtime())
    output_dir = join(output_dir, output_id, time_string)
    mkdirs_if_not_exist(output_dir)
    return output_dir


def setup_gpu(gpu_ids: list):
    if isinstance(gpu_ids, (int, str)):
        gpu_ids = [gpu_ids]
    if not isinstance(gpu_ids, list):
        raise ValueError('Unrecognized gpu_ids: {}'.format(gpu_ids))

    gpu_ids = [str(g) for g in gpu_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids)

    return gpu_ids


def setup_seed(seed):
    if seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

def save_code(name, file_list=None,ignore_dir=['']):
    with zipfile.ZipFile(name, mode='w',
                         compression=zipfile.ZIP_DEFLATED) as zf:
        if file_list is None:
            file_list = []

        first_list = []
        for first_contents in ['*']:
            first_list.extend(glob(first_contents, recursive=True))

        for dir in ignore_dir:
            if dir in first_list:
                first_list.remove(dir)
        patterns = [x + '/**' for x in first_list]
        for pattern in patterns:
            file_list.extend(glob(pattern, recursive=True))

        file_list = [x[:-1] if x[-1] == "/" else x for x in file_list]
        for filename in file_list:
            zf.write(filename)

def save_config(config, path, type):
    if config is not None:
        F = open(path+'/config_of_{}.txt'.format(type), 'a')
        F.write(str(config))
        F.close()

def experiment_saver(run_type, args, output_dir, overwrite = False):

    TENSORBOARD_DIR = os.path.join(output_dir,'tb')
    CHECKPOINT_FOLDER = os.path.join(output_dir,'ckpt')
    sh_n_code = os.path.join(output_dir,'sh_n_code')
    configs = os.path.join(output_dir,'configs')
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    os.makedirs(sh_n_code, exist_ok=True)
    os.makedirs(configs, exist_ok=True)
    dirs = [TENSORBOARD_DIR, CHECKPOINT_FOLDER, sh_n_code, configs]

    if run_type == 'train':
        if any([os.path.exists(d) for d in dirs]):
            for d in dirs:
                if os.path.exists(d):
                    print('{} exists'.format(d))
            # if overwrite or input('Output directory already exists! Overwrite the folder? (y/n)') == 'y':
            #     for d in dirs:
            #         if os.path.exists(d):
            #             shutil.rmtree(d)
            if overwrite:
                for d in dirs:
                    if os.path.exists(d):
                        shutil.rmtree(d)

    """保存运行时的指令为'run_{服务器名字}.sh'，您下次可以在任何地方使用 sh run_{服务器名字}.sh 运行当前实验"""
    with open(sh_n_code +'/run_{}_{}.sh'.format(run_type,socket.gethostname()), 'w') as f:
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('unzip -d {}/code {}\n'.format(sh_n_code, os.path.join(sh_n_code, 'code.zip')))
        f.write('cp -r -f {} {}\n'.format(os.path.join(sh_n_code, 'code', '*'), quote(os.getcwd())))
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')

    """保存本次运行的所有代码"""
    # save_code(os.path.join(sh_n_code, 'code.zip'),
    #           ignore_dir=['coco', 'binary_offline_reference_handwritten_stroke_2021', \
    #                       'chinese_stroke_2021', 'result'])

    """保存config"""
    save_config(args, configs, run_type)

    return TENSORBOARD_DIR, CHECKPOINT_FOLDER