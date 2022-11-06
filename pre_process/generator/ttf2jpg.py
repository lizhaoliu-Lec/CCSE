from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm
import numpy as np

from common.utils import mkdirs_if_not_exist, check_if_file_exists, plt_show


def try_font(font_file='huaWenKaiTi.ttf'):
    font = ImageFont.truetype(font_file, 64)  # the size of character is also 64*64
    print("===> font: ", font)


def get_font_size_information(font_file='huaWenKaiTi.ttf', size=64):
    max_w, max_h, max_ox, max_oy = -1, -1, -1, -1
    for c in tqdm(range(0x3402, 0x9fd1)):  # use the encoding of chinese characters
        text = chr(c)

        font = ImageFont.truetype(font_file, size)  # the size of character is 64*64

        text_width, text_height = font.getsize(text)
        offsetx, offsety = font.getoffset(text)

        if text_width > max_w:
            max_w = text_width
        if text_height > max_h:
            max_h = text_height
        if offsetx > max_ox:
            max_ox = offsetx
        if offsety > max_oy:
            max_oy = offsety

        # print("===> {}: w:{}, h:{}, x:{}, y:{}".format(text, text_width, text_height, offsetx, offsety))

    print(
        "font, size, max_w, max_h, max_ox, max_oy = ({}, {}, {}, {}, {}, {})".format(font_file, size, max_w, max_h, max_ox,
                                                                                 max_oy))


def generateImageFromTTF(save_dir='../../resources/kaiTi', font_file='huaWenKaiTi.ttf', img_size=80, font_size=64):
    mkdirs_if_not_exist(save_dir)
    check_if_file_exists(font_file)

    for c in tqdm(range(0x3402, 0x9fd1)):  # use the encoding of chinese characters

        text = chr(c)
        # for sinsum.ttf, set the img size to 64
        # for kaiti, set the img size to 80
        # the size of generated image is img_size*img_size, and background is 255 white
        img = Image.new('L', (img_size, img_size), 255)
        dr = ImageDraw.Draw(img)

        font = ImageFont.truetype(font_file, font_size)  # the size of character is 64*64
        # write the character to image from the top left (0, 0) and the color of character is black
        dr.text((0, 0), text, font=font, fill='#000000')

        if np.sum(255 - np.array(img)) == 0:
            continue

        img.save('{}/{}.jpg'.format(save_dir, chr(c)))


if __name__ == '__main__':
    # generateImageFromTTF()
    # generateImageFromTTF(save_dir='../../resources/songTi', font_file='simsun.ttf')
    # generateImageFromTTF(save_dir='../../resources/kaiTiV2', font_file='huawenkaitiV2.ttf')
    generateImageFromTTF(save_dir='../../resources/kaiTiV3', font_file='huawenkaitiV2.ttf', font_size=50, img_size=64)
    # get_font_size_information(font_file='huaWenKaiTi.ttf', size=50)
    # get_font_size_information(font_file='huawenkaitiV2.ttf', size=64)
    # get_font_size_information(font_file='simsun.ttf', size=64)
    # try_font()
