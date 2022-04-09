from cv2 import cv2
import numpy as np
import scipy.ndimage
from numpy import ndarray
import os


def preprocess(img: ndarray):
    # clean border
    img[0:, 0:1] = 255
    img[0:, -1:] = 255
    img[0:1, 0:] = 255
    img[-1:, 0:] = 255
    # clean background
    img[img > 180] = 255
    img[0:15, 0:15] = 255

    height, width = img.shape[:2]

    start_col = next(filter(lambda x: min(x[1]) < 255, list(enumerate(img.T))))[0]
    start_row = next(filter(lambda x: x[1] < 255, list(enumerate(img.T[start_col]))))[0]
    end_col = next(filter(lambda x: min(x[1]) < 255, reversed(list(enumerate(img.T)))))[0]
    end_row = next(filter(lambda x: x[1] < 255, list(enumerate(img.T[end_col]))))[0]

    img[:, :75] = 255

    # TODO: optimizacija ovih parametara
    # img = ~img
    # img = cv2.erode(img, np.ones((1, 1), np.uint8), iterations=1)
    # img = ~img
    # img = scipy.ndimage.median_filter(img, (4, 1))
    # img = cv2.erode(img, np.ones((1, 1), np.uint8), iterations=1)
    # img = scipy.ndimage.median_filter(img, (1, 1))

    # img = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

    return img


def main():
    directory_src = "./dataset"

    for filename in os.listdir(directory_src):
        print(filename)

        img: ndarray = preprocess(cv2.imread("./dataset/" + filename, 0))

        cv2.imwrite("./dataset_cleared/" + filename, img)

if __name__ == '__main__':
    main()
