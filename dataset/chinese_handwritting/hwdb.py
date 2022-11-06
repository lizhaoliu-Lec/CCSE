"""
Handwriting database 1.0, 1.1 and 1.2

By combining 1.0 and 1.1, There are totally 2,693,183 samples for training and 224,590 samples for testing.
The training and test data were produced by different writers.
The number of character class is 3,755 (level-1 set ofGB2312-80)

Description reference:
http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html

Download reference:
http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
Gnt1.0 Train
part1 (962MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0TrainPart1.zip
part2 (973MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0TrainPart2.zip
part3 (983MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0TrainPart3.zip
Gnt1.0 Test (722MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.0Test.zip

Gnt1.1 Train
part1 (897MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart1.zip
part2 (943MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1TrainPart2.zip
Gnt1.1 Test (468MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.1Test.zip

Gnt1.2 Train
part1 (897MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.2TrainPart1.zip
part2 (943MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.2TrainPart2.zip
Gnt1.2 Test (468MB) : http://www.nlpr.ia.ac.cn/databases/Download/Offline/CharData/Gnt1.2Test.zip

HWDB1.0 includes 3,866 Chinese characters and 171 alphanumeric and symbols. Among the 3,866 Chinese characters, 3,
740 characters are in the GB2312-80 level-1 set (which contains 3,755 characters in total). 　
HWDB1.1 includes 3,755 GB2312-80 level-1 Chinese characters and 171 alphanumeric and symbols.
HWDB1.2 includes 3,319 Chinese characters and 171 alphanumeric and symbols. The set of Chinese characters
in HWDB1.2 (3,319 classes) is a disjoint set of HWDB1.0. 　
** HWDB1.0 and HWDB1.2 together include 7185 Chinese characters (7,185=3,866+3,319),which include all of 6763
Chinese characters in GB2312. **

Paper reference:
ICDAR 2013 Robust Reading Competition http://refbase.cvc.uab.es/files/KSU2013.pdf
Drawing and Recognizing Chinese Characters with Recurrent Neural Network https://arxiv.org/pdf/1606.06539

Code reference:
https://github.com/YifeiY/hanzi_recognition
"""
import codecs
import os
import pickle
import struct
from time import time
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset

from common.utils import join, check_if_file_exists
from dataset.chinese_handwritting.utils import process_tag_code_for_hwdb, OfflineSample, get_rand_index


class HWDB(Dataset):
    """
    Handwriting database 1.0, 1.1 and 1.2
    """
    _repr_indent = 4

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
                characters.extend(self.decode_gnt_file(_))
            print("Dumping stroke file for {} set to {}...".format(self.split, self.cache_file))
            with open(self.cache_file, 'wb') as f:
                pickle.dump(characters, f)

        self.characters = characters

    def __getitem__(self, item):
        ...

    def __len__(self):
        return len(self.characters)

    @staticmethod
    def pre_process_into_image_folder(root, versions, split, chinese_only=True, rebuild=False, image_save_dir=None):
        print("===> Start pre preprocessing... with (root, versions, split, chinese_only) "
              "= ({}, {}, {}, {})".format(root,
                                          versions,
                                          split,
                                          chinese_only))

        def is_chinese_character(_char):
            if '\u4e00' <= _char <= '\u9fa5':
                return True
            return False

        # new place for images
        if image_save_dir is None:
            image_save_dir = join(root, '-'.join(versions), 'image', split)

        # if rebuild and os.path.exists(image_save_dir):
        #     shutil.rmtree(image_save_dir)

        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        print("===> image_save_dir: ", image_save_dir)

        version_split_to_dir = {
            ('1.0', 'train'): ['Gnt1.0TrainPart1', 'Gnt1.0TrainPart2', 'Gnt1.0TrainPart3'],
            ('1.0', 'test'): ['Gnt1.0Test'],
            ('1.1', 'train'): ['Gnt1.1TrainPart1', 'Gnt1.1TrainPart2'],
            ('1.1', 'test'): ['Gnt1.1Test'],
            ('1.2', 'train'): ['Gnt1.2TrainPart1', 'Gnt1.2TrainPart2'],
            ('1.2', 'test'): ['Gnt1.2Test'],
        }

        # aggregate all pot file under the gnt root
        gnt_dirs = [
            join(root, version_split_dir) for v in versions for version_split_dir in version_split_to_dir[(v, split)]
        ]
        for _ in gnt_dirs:
            check_if_file_exists(_)

        gnt_path_list = []
        for gnt_dir in gnt_dirs:
            gnt_name_list = os.listdir(gnt_dir)
            _gnt_path_list = [join(gnt_dir, _) for _ in gnt_name_list]
            gnt_path_list.extend(_gnt_path_list)

        image_id = 0
        for _ in tqdm(gnt_path_list):
            # print("===> Current/Total: {}/{}, Percentage: {}%".format(image_id, len(gnt_path_list),
            #                                                           round((100 * image_id / len(gnt_path_list)))))
            check_if_file_exists(_)
            # get many images from one gnt file

            samples = HWDB.decode_gnt_file(_)

            for sample in samples:
                # check if is chinese character
                if chinese_only and not is_chinese_character(sample.tag):
                    continue

                # check if class dir exist
                class_dir = join(image_save_dir, sample.tag_code)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                # print("===> sample.tag: ", sample.tag)
                img = sample.data
                img = np.stack([img, img, img], axis=2)

                # 利用PIL保存
                try:
                    img = Image.fromarray(img)
                    file_name = '{}_{}_{}.jpg'.format('ChineseHandWrittenDatabase', split, str(image_id).zfill(12))
                    img.save(os.path.join(class_dir, file_name))
                    image_id += 1
                except:
                    print("===> An error occurs...")

        print("===> Done pre preprocessing...")

    @staticmethod
    def decode_gnt_file(filename):
        """
        :param filename: a writer's encoded ground truth file.
        :return:    samples list, each sample with format (charname, img)
        """
        samples = []
        start = time()
        with codecs.open(filename, mode='rb') as fin:
            while True:
                left_cache = fin.read(4)
                if len(left_cache) < 4:
                    break
                sample_size = struct.unpack("I", left_cache)[0]
                tag, tag_code = process_tag_code_for_hwdb(fin.read(2))
                width = struct.unpack("H", fin.read(2))[0]
                height = struct.unpack("H", fin.read(2))[0]
                img = np.zeros(shape=[height, width], dtype=np.uint8)
                for r in range(height):
                    for c in range(width):
                        img[r, c] = struct.unpack("B", fin.read(1))[0]
                if width * height + 10 != sample_size:
                    break
                sample = OfflineSample(tag_code=tag_code, tag=tag, data=img)
                samples.append(sample)

        # print("Offline file  decoded in {:.3f} seconds to get {} character.\n".format((time() - start), len(samples)))

        return samples

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
    ROOT = '/home/liulizhao/datasets/HWDB'
    NUM_SHOW = 10


    def run_OLHWDB():
        d = HWDB(root=ROOT, split='test')
        print(d)
        characters = d.characters
        rand_index = get_rand_index(num_show=NUM_SHOW, length=len(d.characters))
        for _ in rand_index[:NUM_SHOW]:
            characters[_].show()


    def run_decode():
        d = HWDB.decode_gnt_file('E:/Datasets/HWD/Gnt1.0Test/121-t.gnt')
        for i in range(-5, -1):
            d[i].show()


    def run_preprocess():
        # root = 'E:/Datasets/HWD'
        root = '/mnt/cephfs/dataset/HWDB'
        # versions = ['1.0', '1.1', '1.2']
        versions = ['1.0']
        split = 'train'
        HWDB.pre_process_into_image_folder(root=root, versions=versions, split=split, chinese_only=True, rebuild=True)


    # run_decode()
    run_preprocess()
