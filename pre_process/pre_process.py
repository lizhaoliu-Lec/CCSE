from PIL import Image
import numpy as np
import cv2


def read_to_gray_scale(input_image_path):
    # img = Image.open(input_image_path)
    # img = img.convert("L")
    #
    # # img = skimage.color.gray2rgb(np.array(img))
    # img = np.array(img)
    # img = np.stack([img, img, img], axis=2)

    # img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]

    img = cv2.imread(input_image_path, 0)

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.boxFilter(img, -1, (3, 3), normalize=True)
    img = np.array(img)
    img = np.stack([img, img, img], axis=2)

    return img
