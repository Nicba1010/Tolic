import matplotlib.pyplot as plt
from cv2 import cv2

# load
import numpy as np
from numpy import ndarray

img: ndarray = cv2.imread('./dataset/c7nwp.png', 0)
# clean border
img = img[2:-2, 2:-2]
# clean background
img[img > 180] = 255
img[0:15, 0:15] = 255

height, width = img.shape[:2]

start_col = next(filter(lambda x: min(x[1]) < 255, list(enumerate(img.T))))[0]
start_row_one = next(filter(lambda x: x[1] < 255, list(enumerate(img.T[start_col]))))[0]
start_row_two = next(filter(lambda x: x[1] < 255, reversed(list(enumerate(img.T[start_col])))))[0]
end_col = next(filter(lambda x: min(x[1]) < 255, reversed(list(enumerate(img.T)))))[0]
end_row = next(filter(lambda x: x[1] < 255, list(enumerate(img.T[end_col]))))[0]

print(f"Start Small: {start_col}, {start_row_one}")
print(f"Start Big: {start_col}, {start_row_two}")
print(f"End: {end_col}, {end_row}")

img[:, :69] = 255

cv2.imshow("Kaptcha", cv2.resize(img, (width * 4, height * 4)))
cv2.waitKey(0)
cv2.destroyAllWindows()
