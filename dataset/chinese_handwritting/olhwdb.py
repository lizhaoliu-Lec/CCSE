"""
Online handwriting database 1.0, 1.1 and 1.2

By combining 1.0 and 1.1, There are totally 2,693,183 samples for training and 224,590 samples for testing.
The training and test data were produced by different writers.
The number of character class is 3,755 (level-1 set ofGB2312-80)

Description reference:
http://www.nlpr.ia.ac.cn/databases/handwriting/Online_database.html

Download reference:
http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
Pot1.0 Train (273MB) : http://www.nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.0Train.zip
Pot1.0 Test (68MB) : http://www.nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.0Test.zip
Pot1.1 Train (189MB) : http://www.nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.1Train.zip
Pot1.1 Test (47MB) : http://www.nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.1Test.zip
Pot1.2 Train (196MB) : http://www.nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.2Train.zip
Pot1.2 Test (50MB) : http://www.nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.2Test.zip

Paper reference:
ICDAR 2013 Robust Reading Competition http://refbase.cvc.uab.es/files/KSU2013.pdf
Drawing and Recognizing Chinese Characters with Recurrent Neural Network https://arxiv.org/pdf/1606.06539

Code reference:
https://github.com/YifeiY/hanzi_recognition
"""
import os
import pickle
import struct
from time import time
from math import cosh
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import Dataset

from common.utils import join, check_if_file_exists

import matplotlib

from dataset.chinese_handwritting.utils import process_tag_code_for_olhwdb, OnlineSample, get_rand_index

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']


class OLHWDB(Dataset):
    _repr_indent = 4

    """
    Online handwriting database 1.0, 1.1 and 1.2
    """

    def __init__(self,
                 root,
                 split='train',
                 versions=('1.0', '1.1', '1.2'),
                 transforms=None):
        super().__init__()
        assert split in ['train', 'test']
        for v in versions:
            assert v in ['1.0', '1.1', '1.2']
        self.root = root
        self.split = split
        self.versions = versions
        self.transforms = transforms
        self.cache_file = join(root, split.capitalize() + '.' + '-'.join(versions) + '.cache')

        try:
            check_if_file_exists(self.cache_file)
            print("Loading stroke file for {} set from {}...".format(self.split, self.cache_file))
            with open(self.cache_file, mode='rb') as f:
                characters = pickle.load(f)
        except:
            pot_list = []
            for version in versions:
                version_root = join(root, 'Pot' + version + split.capitalize())
                version_pot_list = os.listdir(version_root)
                pot_list.extend([join(version_root, _) for _ in version_pot_list])

            self.pot_list = pot_list

            characters = []
            print("Decoding stroke file for {} set...".format(self.split))
            for _ in tqdm(self.pot_list):
                # check if file exists
                # check_if_file_exists(_)
                characters.extend(self.decode_pot_file(_))
            print("Dumping stroke file for {} set to {}...".format(self.split, self.cache_file))
            with open(self.cache_file, 'wb') as f:
                pickle.dump(characters, f)

        self.characters = characters

    def __getitem__(self, item):
        ...

    def __len__(self):
        return len(self.characters)

    @staticmethod
    def decode_pot_file(filename):
        """
        read file, create internal representation of binary file data in ints
        """
        characters = []
        # print("Decoding {} stroke files...".format(filename))
        # print("Data will be normalized, Redundant points in stroke will be removed")
        start = time()

        with open(filename, "rb") as f:
            while True:
                sample_size = f.read(2)
                if sample_size == b'':
                    break

                # "啊"=0x0000b0a1 Stored as 0xa1b00000, Only two bytes (GB2132 or GBK) are meaningful

                # dword_code = f.read(2)
                # # if dword_code[0] != 0:
                # #     dword_code = bytes((dword_code[1], dword_code[0]))
                # dword_code = bytes((dword_code[1], dword_code[0]))
                # tag_code = struct.unpack(">H", dword_code)[0]
                # f.read(2)
                # tag = struct.pack('>H', tag_code).decode("gbk")[0]
                # tag_code = hex(tag_code)
                # stroke_number = struct.unpack("<H", f.read(2))[0]

                tag, tag_code = process_tag_code_for_olhwdb(f.read(2))
                f.read(2)
                stroke_number = struct.unpack("<H", f.read(2))[0]

                strokes_samples = []
                stroke_samples = []
                next_byte = b'\x00'
                while next_byte != (b'\xff\xff', b'\xff\xff'):
                    next_byte = (f.read(2), f.read(2))
                    if next_byte == (b'\xff\xff', b'\x00\x00'):
                        strokes_samples.append(stroke_samples)
                        stroke_samples = []
                    else:
                        stroke_samples.append(
                            (struct.unpack("<H", next_byte[0])[0], struct.unpack("<H", next_byte[1])[0]))

                sample = OnlineSample(tag_code=tag_code, tag=tag, data=strokes_samples)
                sample.shrinkPixels()
                sample.normalize(128)
                sample.removeRedundantPoints()
                characters.append(sample)

                print("===> ", sample)

        print("Online file decoded in {:.3f} seconds to get {} character.\n".format((time() - start), len(characters)))
        return characters

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    @staticmethod
    def _format_transform_repr(transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


if __name__ == '__main__':
    ROOT = '/home/liulizhao/datasets/OLHWD'
    NUM_SHOW = 10
    import random


    def run_OLHWDB():
        d = OLHWDB(root=ROOT, split='test')
        print(d)
        characters = d.characters
        rand_index = get_rand_index(num_show=NUM_SHOW, length=len(d.characters))
        for _ in rand_index[:NUM_SHOW]:
            characters[_].show()


    def run_decode():
        d = OLHWDB.decode_pot_file('E:/Datasets/OLHWD/Pot1.0Test/121.pot')
        for i in range(-5, -1):
            d[i].show()


    run_decode()
