import os
import json

import struct
from codecs import decode
import numpy as np
import tqdm
import cv2


def load_gnt_file(filename):
    """Parse GNT a file

    Load characters and images from a given GNT file.
    The format of gnt file can be seen at nlpr.ia.ac.an/databases/Download/GntRead.cpp.pdf

    :param filename: The file path to load.

    :return: (image: Pillow.Image.Image, character) tuples
    """
    with open(filename, "rb") as f:
        while True:
            packed_length = f.read(4)
            if packed_length == b'':
                break
            raw_label = struct.unpack(">cc", f.read(2))
            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]
            photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))
            label = decode(raw_label[0] + raw_label[1], encoding="gb18030")
            image = np.array(photo_bytes).reshape(height, width)

            yield image, label


def read_gnt_image(path, img_dir, count):
    """Save image and label

    Write the image and label read from GNT to disk
    Every GNT file represent a handwriting style including about 3000~4000 images
    All images and labels from the same GNT file are write in the same dir,
    A json file does a map from image name to label

    :param path: the path of GNT file
    :param img_dir: the dir the images from GNT file are write to
    """
    d = {}
    data = load_gnt_file(path)
    sub_img_dir = os.path.join(img_dir, path.split('/')[-1][:-4])
    if not os.path.exists(sub_img_dir):
        os.mkdir(sub_img_dir)
    while True:
        try:
            image, label = next(data)
            cv2.imwrite(os.path.join(sub_img_dir, str(count) + '.jpg'), image)
            d[str(label)[0]] = str(count)
            count += 1
        except StopIteration:
            with open(os.path.join(sub_img_dir, 'dict.json'), 'w') as f:
                json.dump(d, f)
            break
        except:
            print("===> error occur: ")
        # print(count)


def run():
    gnt_dir = '/home/liulizhao/datasets/HWDB/gnt'
    img_dir = '/home/liulizhao/datasets/HWDB/img'

    total_gnt = len(os.listdir(gnt_dir))
    dir_cnt = 0
    for sub_gnt_dir_name in os.listdir(gnt_dir):
        dir_cnt += 1
        if dir_cnt > 4:
            continue
        print("===> Processing Gnt: {} {}/{}".format(sub_gnt_dir_name, dir_cnt, total_gnt))
        sub_gnt_dir = os.path.join(gnt_dir, sub_gnt_dir_name)
        sub_img_dir = os.path.join(img_dir, sub_gnt_dir_name)

        if not os.path.exists(sub_img_dir):
            os.mkdir(sub_img_dir)
        for gnt in tqdm.tqdm(os.listdir(sub_gnt_dir)):
            count = 0
            sub_gnt_path = os.path.join(sub_gnt_dir, gnt)
            read_gnt_image(sub_gnt_path, sub_img_dir, count)
            count += 1


if __name__ == '__main__':
    run()
