import struct
import random
from math import cosh

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def process_tag_code_for_olhwdb(byte_string):
    tag_code = bytes((byte_string[1], byte_string[0]))
    tag_code = struct.unpack('>H', tag_code)[0]
    tag = struct.pack('>H', tag_code).decode('gbk')[0]
    tag_code = hex(tag_code)
    return tag, tag_code


def process_tag_code_for_hwdb(byte_string):
    tag_code = bytes((byte_string[0], byte_string[1]))
    tag_code = struct.unpack('>H', tag_code)[0]
    tag = byte_string.decode('gbk')[0]
    tag_code = hex(tag_code)
    return tag, tag_code


def get_rand_index(num_show, length):
    assert num_show <= length
    index = list(range(length))
    random.shuffle(index)
    return index


class Sample:
    """
    tag_code: hex of tag
    tag: character
    data: TO INHERIT
    """

    def __init__(self, tag_code, tag, data):
        self.tag_code = tag_code
        self.tag = tag
        self.data = data

    def __repr__(self):
        return 'Sample(tag_code={}, tag={})'.format(self.tag_code, self.tag)

    def show(self):
        raise NotImplementedError


class OnlineSample(Sample):
    """
    data: list[list], strokes that has multiple point position in one stroke
    """

    def show(self):
        """plots the character using matplotlib"""
        for i, stroke in enumerate(self.data):
            plt.plot([p[0] for p in stroke], [p[1] for p in stroke], label='{} draw'.format(i), linewidth=1)
        plt.title('stroke_number: {}'.format(len(self.data)))
        plt.legend()
        # plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

    def shrinkPixels(self):
        """normalize the pixel values to a minimum of 0,
        eg. (1234,2345) -> (34,45) so that the character has minimum coordinates of (0,_),(_,0)"""
        minx = self.data[0][0][0]
        maxy = 0
        for stroke in self.data:
            for v in stroke:
                minx = min(minx, v[0])
                maxy = max(maxy, v[1])

        for strokes in self.data:
            for s in range(len(strokes)):
                strokes[s] = (strokes[s][0] - minx, maxy - strokes[s][1])

    def normalize(self, upper_bound):
        bounds = [self.data[0][0][0], self.data[0][0][1]]
        for stroke in self.data:
            for v in stroke:
                bounds = [max(bounds[0], v[0]), max(bounds[1], v[1])]
        bound = max(bounds)
        for stroke in self.data:
            for i in range(len(stroke)):
                stroke[i] = (stroke[i][0] / bound * upper_bound, stroke[i][1] / bound * upper_bound)

    def removeRedundantPoints(self):
        new_stroke_data = []
        for stroke in self.data:
            new_stroke = [stroke[0]]
            # add the stroke if it only contains one point
            if len(stroke) == 1:
                new_stroke_data.append(new_stroke)
                continue
            if (new_stroke[-1][1] - stroke[1][1]) == 0:
                last_cos = 100
            else:
                last_cos = - cosh((new_stroke[-1][0] - stroke[1][0]) / (new_stroke[-1][1] - stroke[1][1]))

            for i in range(1, len(stroke) - 1):
                # save the stroke if it represents a large euclidean change
                dx = new_stroke[-1][0] - stroke[i][0]
                dy = new_stroke[-1][1] - stroke[i][1]
                if dy == 0:
                    this_cos = 100
                else:
                    this_cos = cosh(dx / dy)
                if dx ** 2 + dy ** 2 > 100 or abs(this_cos - last_cos) > 0.2:
                    new_stroke.append(stroke[i])
                last_cos = this_cos
            new_stroke.append(stroke[-1])
            for i in range(len(stroke)):
                new_v = (int(stroke[i][0]), int(stroke[i][1]))
                stroke[i] = new_v
            new_stroke_data.append(new_stroke)
        self.data = new_stroke_data

    def __repr__(self):
        return 'OnlineSample(tag_code={}, tag={}, num_stroke={})'.format(self.tag_code, self.tag, len(self.data))


class OfflineSample(Sample):
    def show(self):
        plt.figure(figsize=(3, 3))
        plt.imshow(self.data, cmap='binary')
        plt.axis('off')
        plt.tight_layout()
        plt.title(u"{}".format(self.tag))
        plt.show()
        plt.close()

    def __repr__(self):
        return 'OfflineSample(tag_code={}, tag={}, height={}, weight={})'.format(self.tag_code, self.tag,
                                                                                 self.data.shape[0],
                                                                                 self.data.shape[1])
