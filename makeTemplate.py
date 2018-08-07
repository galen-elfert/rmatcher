"""
Load a labelled resistor image and generate a template for matching.
"""

import sys
from PointCollector import PointCollector
import matplotlib.pyplot as plt
import pylab
import numpy as np
import cv2 as cv

def main(argv):
    if len(sys.argv) > 1:
        inputs = argv[1:]
        print(inputs)
    else:
        print("Usage: python trainer.py INPUT_IMAGE [INPUT_IMAGE ...]")
        sys.exit()
    fig, ax = plt.subplots()
    template_size = 100
    shape = (template_size, template_size)
    img_sum = np.zeros(shape, dtype=np.float32)
    count = 0
    for img_file in inputs:
        pc = PointCollector(img_file, mode="grayscale")
        templates = pc.getTemplates(template_size)
        for template in templates:
            templatef = template.astype(np.float32)
            img_sum = img_sum + templatef
            count += 1
    img_avg = img_sum / count
    img_out = img_avg.astype(np.uint8)

    # Post processing
    hue_min = int(25/2)
    hue_max = int(35/2)
    # hue_min = 10
    # hue_max = 20
    
    # img_hsv = cv.cvtColor(img_out, cv.COLOR_BGR2HSV)
    # thresh_low = np.array([hue_min, 0, 0])
    # thresh_high = np.array([hue_max, 255, 255])
    # mask = cv.inRange(img_hsv, thresh_low, thresh_high)
    # img_out = cv.bitwise_and(img_out
    # img_bw = cv.cvtColor(img_out, cv.COLOR_BGR2GRAY)
    # img_blur = cv.GaussianBlur(img_bw, (5,5), 3)
    # ret, img_out = cv.threshold(img_bw, 70, 255, cv.THRESH_TOZERO)
    ax.imshow(img_out, 'gray')
    pylab.show()

if __name__ == "__main__":
    main(sys.argv)
