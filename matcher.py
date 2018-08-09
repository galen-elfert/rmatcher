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
from pprint import pprint

class location:
    def __init__(self, array, depth=0, offset=np.array([0,0,0])):
        # Pad array to make all dims even
        padding = [(0, i % 2) for i in array.shape]
        self.array = np.pad(array, padding, 'edge')
        self.value = np.amax(array)
        self.offset = offset
        # location of max point relative to original image
        self.location = np.array(offset) + np.unravel_index(array.argmax(), array.shape)
        self.depth = depth

    def expand(self):
        """ Generate children """
        shape = self.array.shape
        children = []
        for i in range(2):
            xdim = int(shape[0] / 2)
            xoff = i * xdim
            for j in range(2):
                ydim = int(shape[1] / 2)
                yoff = j * ydim
                for k in range(2):
                    zdim = int(shape[2] / 2)
                    zoff = k * zdim
                    depth = self.depth + 1
                    array = self.array[xoff:xoff+xdim, yoff:yoff+ydim, zoff:zoff+zdim]
                    offset = self.offset + np.array([xoff, yoff, zoff])
                    children.append(location(array, depth=depth, offset=offset))
        return children

    def __str__(self):
        return "{0:<15}: {1}, ({2}), offset: {3}".format(str(tuple(self.location)), self.value, self.depth, str(self.offset))

    # Comparison functions, for sorting
    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

def main(argv):
    num_angles = 8
    template_size = 80

    if len(argv) < 2:
        print("Usage: python matcher.py INPUT_IMAGE TEMPLATE_IMAGE")
        sys.exit()

    print(argv)

    # Load template and input images
    img_input = cv.imread(argv[0], cv.IMREAD_GRAYSCALE)
    img_template = cv.imread(argv[1], cv.IMREAD_GRAYSCALE)

    # Generate rotated templates
    templates = []
    tin_shape = img_template.shape
    tin_width = tin_shape[0]
    if tin_width - template_size < 0:
        raise Exception("template size must be less than input template size")

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
            match_sum = match
            match_array = match
        else:
            match_sum += match
            match_array = np.dstack((match_array, match))
        count += 1
    match_mean = match_sum / float(count)
    match_min = np.amin(match_mean)
    match_max = np.amax(match_mean)
    match_norm = (match_mean - match_min) / (match_max - match_min)

    # Search for maxima
    max_depth = 4
    max_locations = 8
    locations = [location(match_array)]
    pprint([str(l) for l in locations])
    print("---")
    while min([l.depth for l in locations]) < max_depth:
        children = []
        for l in locations:
            children += l.expand()
        children.sort(reverse=True)
        locations = children[:max_locations]
        pprint([str(l) for l in locations])
        print("---")

    # plt.imshow(match_norm, cmap=plt.get_cmap('inferno'))
    x = []
    y = []
    for l in locations:
        x.append(l.location[0])
        y.append(l.location[1])
    plt.imshow(img_input, 'gray')
    plt.plot(y, x, 'ws', markersize=20, markerfacecolor='none')
    plt.plot([0], [0], 'rx', markersize=20)
    plt.plot(img_input.shape[1], img_input.shape[0], 'gx', markersize=20)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
