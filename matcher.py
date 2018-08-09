"""
Apply rotational template matching
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import numpy as np
import cv2 as cv
import sys

# Globals
num_angles = 8
template_size = 80
templates = []
matches = []
locations = []
locations_real = []
img_input = None

fig, axes = plt.subplots(1,2)
current_image = 0

class location:
    def __init__(self, value, location):
        self.val = value
        self.loc = location

    def __eq__(self, other):
        return self.val == other.val

    def __ne__(self, other):
        return self.val != other.val

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __gt__(self, other):
        return self.val > other.val

    def __ge__(self, other):
        return self.val >= other.val

# Draw the match image for the current template next to the input image 
# with the best match location plotted on both
def drawMatch(index):
    global fig, ax, img_input
    axes[0].clear()
    axes[0].imshow(img_input)
    x = locations_real[index][0]
    y = locations_real[index][1]
    axes[0].plot(x, y, 'r+')
    axes[1].clear()
    axes[1].imshow(matches[index])
    x = locations[index][0] 
    y = locations[index][1]
    axes[1].plot(x, y, 'r+')
    fig.canvas.draw()

# Callback for match viewing
def onKey(event):
    global fig, ax, current_image
    if event.key == 'j':
        if current_image < num_angles-1:
            current_image += 1
    if event.key == 'k':
        if current_image > 0:
            current_image -= 1
    drawMatch(current_image)

def main(argv):
    global img_input, locations, locations_real, fig, axes
    if len(argv) < 2:
        print("Usage: python matcher.py INPUT_IMAGE TEMPLATE_IMAGE")
        sys.exit()

    print(argv)

    # Load template and input images
    img_input = cv.imread(argv[0])
    img_template = cv.imread(argv[1])

    # Generate rotated templates
    tin_shape = img_template.shape
    tin_width = tin_shape[0]
    pad = tin_width - template_size
    if pad < 0:
        raise Exception("template size must be less than input template size")

    for i in range(num_angles):
        angle = (i / num_angles) * 360
        mat_rot = cv.getRotationMatrix2D((tin_width/2, tin_width/2), angle, 1)
        img_rot = cv.warpAffine(img_template, mat_rot, (tin_width, tin_width))
        templates.append(img_rot)

    # Generate template match images
    count = 0
    maxmaxval = 0
    maxmaxloc = 0
    maxmaxindex = 0
    for template in templates:
        match = cv.matchTemplate(img_input, template, cv.TM_CCOEFF)
        matches.append(match)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
        locations.append(max_loc)
        real_loc = (max_loc[0]+(template_size/1.3), max_loc[1]+(template_size/1.3))
        locations_real.append(real_loc)
        print("max_val: ", max_val)
        print("max_loc: ", max_loc)
        if max_val > maxmaxval:
            maxmaxval = max_val
            maxmaxloc = max_loc
            maxmaxindex = count
        count += 1

    print("maxmaxindex: ", maxmaxindex)
    current_image = maxmaxindex
    drawMatch(current_image)
    fig.canvas.mpl_connect('key_press_event', onKey)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
