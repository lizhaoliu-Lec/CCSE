import numpy as np
import cv2 as cv
import os, sys
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--font', default='2', type=str)
parser.add_argument('--input_addr', required=True, type=str)
parser.add_argument('--run_id', required=True, type=str)
args = parser.parse_args()
font = args.font
run_id = args.run_id
input_addr = args.input_addr
output_addr = os.path.join('results/{}/{}'.format(font, run_id))
print("Using input_addr: {}".format(input_addr))
print("Using output_addr: {}".format(output_addr))
assert os.path.exists(input_addr), 'input_addr: {} not exists'.format(input_addr)
if not os.path.exists(output_addr):
    os.makedirs(output_addr)

cnt = 0

font_list = os.listdir(input_addr)
print(font, len(font_list))
for font_id in tqdm(font_list):
    stroke_list = os.listdir(os.path.join(input_addr, font_id))
    L = len(stroke_list)
    IMG = np.ones((320, 320, 3))
    for i in range(L):
        p = os.path.join(input_addr, '{}/{}_{}.png'.format(font_id, font_id, i + 1))
        # print("===> p: {}".format(p))
        img = cv.imread(p) / 255
        IMG *= img
    cv.imwrite('{}/{}.png'.format(output_addr, font_id), IMG * 255)
    cnt += 1
    if cnt % 1000 == 0:
        print("Cnt/Total: {}/{}".format(cnt, len(font_list)))
