from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen

while True:
    req = urlopen('https://oss.uredjenazemlja.hr/servlets/kaptcha.jpg')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'
    plt.imshow(img)
    plt.show()
    solution = input("Solve: ")
    cv2.imwrite(f"dataset/{solution}.png", img)
