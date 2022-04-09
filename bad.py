import os

from cv2 import cv2
import matplotlib.pyplot as plt
from shutil import move

img_folder: str = "./bad_predictions_processing"
out_img_folder: str = "./dataset_bad_corrected"
os.makedirs(out_img_folder, exist_ok=True)

for _, _, files in os.walk(img_folder):
    for f in files:
        path = os.path.join(img_folder, f)
        img = cv2.imread(path)
        plt.imshow(img)
        plt.show()
        solution = input(f"Solve ({f[:-4]}): ")

        if solution == "s" or solution.__len__() != 5 or solution == f[:-4]:
            continue
        else:
            move(path, os.path.join(out_img_folder, f"{solution}.png"))
