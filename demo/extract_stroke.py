import cv2
import numpy as np
import torch
from torchvision.ops import nms

import os
import requests
import math
from pathlib import Path
from argparse import ArgumentParser
from functools import reduce
import operator

import random as rng
rng.seed(12345)



def insert_suffix(filename, suffix):
    return str(Path(filename).with_suffix(suffix + Path(filename).suffix))


def nms_on_contour_rect(rects):
    x1_y1_x2_y2s = []
    scores = []
    labels = []
    for rect in rects:
        x1_y1_x2_y2s.append([rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]])
        scores.append(1)
        labels.append(1)
    boxes = torch.Tensor(x1_y1_x2_y2s)
    scores = torch.Tensor(scores)
    labels = torch.Tensor(labels)
    IOU_THRESHOLD = 0.3  # may need to carefully tune

    keep = nms(boxes=boxes, scores=scores, iou_threshold=IOU_THRESHOLD)
    kept_boxes = boxes[keep]
    kept_scores = scores[keep]
    kept_labels = labels[keep]
    # print(keep)

    # return boxes, kept_boxes, scores, kept_scores, labels, kept_labels, keep
    return keep

def extract_square_box(src_gray, threshold, save_middle_path=""):
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    # cv2.imshow("edge", canny_output)
    if len(save_middle_path) > 1:
        canny_output_path = insert_suffix(save_middle_path, ".canny")
        cv2.imwrite(canny_output_path, canny_output)
    

    contours, _ = cv2.findContours(
        canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)            

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    areas = [None] * len(contours)
    centers = [None] *len(contours)
    radius = [None] * len(contours)
    is_kept = [False] * len(contours)
    largest_poly = None
    largest_contour_area = -100000
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        areas[i] = cv2.contourArea(c)
        bbox_area = boundRect[i][2] * boundRect[i][3]
        if bbox_area * 0.9 <= areas[i] <= bbox_area * 1.1 and boundRect[i][2] * 0.8 <= boundRect[i][3] <= boundRect[i][2] * 1.2 and boundRect[i][2] >= 30:
            is_kept[i] = True
            if bbox_area > largest_contour_area and len(contours_poly[i]) == 4:
                largest_contour_area = bbox_area
                largest_poly = contours_poly[i]
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    contours = [item for idx, item in enumerate(contours) if is_kept[idx]]
    contours_poly = [item for idx, item in enumerate(contours_poly) if is_kept[idx]]
    boundRect = [item for idx, item in enumerate(boundRect) if is_kept[idx]]
    areas = [item for idx, item in enumerate(areas) if is_kept[idx]]
    centers = [item for idx, item in enumerate(centers) if is_kept[idx]]
    radius = [item for idx, item in enumerate(radius) if is_kept[idx]]

    is_kept = nms_on_contour_rect(boundRect).tolist()

    contours = [contours[idx] for idx in is_kept]
    contours_poly = [contours_poly[idx] for idx in is_kept]
    boundRect = [boundRect[idx] for idx in is_kept]
    areas = [areas[idx] for idx in is_kept]
    centers = [centers[idx] for idx in is_kept]
    radius = [radius[idx] for idx in is_kept]

    if len(save_middle_path) > 1:
        contour_output = insert_suffix(save_middle_path, ".contour")
        drawing = np.zeros(
            (canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            if not is_kept[i]:
                continue
            color = (rng.randint(0, 256), rng.randint(
                0, 256), rng.randint(0, 256))
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            # cv2.circle(drawing, (int(centers[i][0]), int(
            #     centers[i][1])), int(radius[i]), color, 2)

        # cv2.imshow('Contours', drawing)
        cv2.imwrite(contour_output, drawing)

    return contours, contours_poly, boundRect, areas, centers, radius, largest_poly


def filter_by_similar_area(contours, contours_poly, boundRect, areas, centers, radius, largest_poly):
    sorted_area = sorted(areas)
    lst_len = len(areas)
    mid_area_index = (lst_len - 1) // 2
    if (lst_len % 2):
        mid_area =  sorted_area[mid_area_index]
    else:
        mid_area = (sorted_area[mid_area_index] + sorted_area[mid_area_index + 1]) / 2
    filter_list = [False] * lst_len
    avg = mid_area
    kept_count = 0
    for idx, area in enumerate(areas):
        if avg * 0.7 <= area <= avg * 1.3:
            filter_list[idx] = True
            avg = (avg * kept_count + area) / (kept_count + 1)
            kept_count += 1
    return filter_list



def preprocess_user_image(image_in_path, image_out_path=None):
    if image_out_path is None:
        image_out_path = insert_suffix(image_in_path, ".prep")
    middle_result_path = insert_suffix(image_out_path, ".mid")
    
    src = cv2.imread(image_in_path)

    # Convert image to gray and blur it
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))

    thresh = 20
    contours, contours_poly, boundRect, areas, centers, radius, largest_poly = extract_square_box(src_gray, thresh, save_middle_path=middle_result_path)

    try:
        print('try to fix perspective issue.')
        coords = [x[0] for x in list(largest_poly)]
        print('base points: {0}'.format(str(coords)))
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        all_degrees = [math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1])) for coord in coords]
        base_degree = min(all_degrees)
        points1 = np.float32(sorted(coords, key=lambda coord: (base_degree - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))
        print('sorted base points: {0}'.format(str(points1)))
        # fix to smaller im
        x_left_max, x_right_min = max([points1[0][0], points1[1][0]]), min([points1[2][0], points1[3][0]])
        y_up_max, y_down_min = max([points1[0][1], points1[3][1]]), min([points1[1][1], points1[2][1]])
        points2 = np.float32([[x_left_max, y_up_max], [x_left_max, y_down_min], [x_right_min, y_down_min], [x_right_min, y_up_max]])
        # fix to larger im
        
        affine_M = cv2.getPerspectiveTransform(points1, points2)
        src = cv2.warpPerspective(src, affine_M, (src.shape[1], src.shape[0]))
        cv2.imwrite(insert_suffix(middle_result_path, ".persp"), src)
        print('fix perspective done.')

        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.blur(src_gray, (3, 3))

        contours, contours_poly, boundRect, areas, centers, radius, largest_poly = extract_square_box(src_gray, thresh, save_middle_path=middle_result_path)
    except Exception as e: 
        print(e)

    kept_list = filter_by_similar_area(contours, contours_poly, boundRect, areas, centers, radius, largest_poly)
    print(len(areas))
    print(areas)
    print(len(kept_list))
    print(kept_list)

    out_paths = []
    for idx, keep in enumerate(kept_list):
        if keep:
            row_top = boundRect[idx][1]
            row_down = row_top + boundRect[idx][3]
            col_left = boundRect[idx][0]
            col_right = col_left + boundRect[idx][2]
            out_img = src[row_top + 3: row_down - 3, col_left + 3: col_right - 3, :]
            out_img_path = insert_suffix(image_out_path, '.{0}'.format(idx))
            cv2.imwrite(out_img_path, out_img)
            out_paths.append(out_img_path)

    return out_paths

def call_nn_stroke_extraction_server(char_img_paths):
    for idx, img_path in enumerate(char_img_paths):
        print(img_path)
        resp = requests.post("http://localhost:5000/check",
                             data={
                                 "image_path": os.path.abspath(img_path),
                                 "output_path": os.path.abspath(insert_suffix(img_path, '.nnmask'))
                             }
                             )
        print(resp.status_code)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("in_file")
    arg_parser.add_argument("out_file")
    args = arg_parser.parse_args()

    char_img_paths = preprocess_user_image(args.in_file, args.out_file)
    call_nn_stroke_extraction_server(char_img_paths)

if __name__ == "__main__":
    main()