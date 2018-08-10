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
import math

# Globals
class location:
    def __init__(self, value, location, angle):
        self.value = value
        self.location = location
        self.angle = angle

    def __str__(self):
        return "{0:<15} val: {1:<10}, angle: {2}".format(str(self.location), self.value, self.angle)

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

def main(argv):
    num_angles = 16
    template_size = 80

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

    templates = []
    for i in range(num_angles):
        angle = (i / num_angles) * 360
        mat_rot = cv.getRotationMatrix2D((tin_width/2, tin_width/2), angle, 1)
        img_rot = cv.warpAffine(img_template, mat_rot, (tin_width, tin_width))
        templates.append(img_rot)

    # Generate template match images
    count = 0
    match_sum = None
    match_array = None 
    for template in templates:
        match = cv.matchTemplate(img_input, template, cv.TM_CCOEFF)
        if count == 0:
            match_sum = np.copy(match)
            match_array = np.copy(match)
        else:
            match_sum += match
            match_array = np.dstack((match_array, match))
        lmax = np.unravel_index(match.argmax(), match.shape) 
        print("layer: ", count, "max: ", np.amax(match), "loc: ", lmax)
        count += 1
    match_mean = match_sum / float(count)
    match_min = np.amin(match_array)
    match_max = np.amax(match_array)
    match_norm = (match_array - match_min) / (match_max - match_min)

    # Find best locations
    num_locations = 8
    blank_radius = 32
    match_copy = np.copy(match_norm)
    shape = match_copy.shape
    xmax = shape[0]
    ymax = shape[1]
    print("shape ", match_copy.shape)
    locations = []
    for i in range(num_locations):
        value = np.amax(match_copy)
        loc = np.unravel_index(match_copy.argmax(), match_copy.shape)
        print("loc: ", str(loc))
        angle = (loc[2] / num_angles) * math.pi * 2
        x = loc[0]
        y = loc[1]
        match_copy[max(0, x-blank_radius):min(xmax, x+blank_radius), max(0, y-blank_radius):min(ymax, y+blank_radius), :] = 0
        locations.append(location(value, (x,y), angle))
        print(str(locations[-1]))
        
    # Display matches
    plt.imshow(cv.cvtColor(img_input, cv.COLOR_BGR2GRAY), 'gray')
    for l in locations:
        x = l.location[1] + template_size/1.3
        y = l.location[0] + template_size/1.3
        x1 = x + math.cos(l.angle) * 32
        x2 = x - math.cos(l.angle) * 32
        y1 = y - math.sin(l.angle) * 32
        y2 = y + math.sin(l.angle) * 32
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)

    # plt.plot(x, y, 'ws', markersize=20, markerfacecolor='none')

    # Display match output arrays
    # fig, axes = plt.subplots(2,4)
    # count = 0
    # for i in range(2):
        # for j in range(4):
            # axes[i][j].imshow(match_copy[:,:,(j+i*4) * int(num_angles/8)])
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
