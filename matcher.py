#!python

"""
Apply rotational template matching
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2 as cv
import sys
import math
import time
import os.path as op

# Globals
class location:
    def __init__(self, value, location, angle):
        self.value = value
        self.location = location
        self.angle = angle

    def __str__(self):
        return "{0:<11} value: {1:<6.4} angle: {2:.2}".format(str(self.location), self.value, self.angle)

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


def match(img_input, img_template, num_angles=16, template_size=None):
    # Generate rotated templates
    tin_shape = img_template.shape
    tin_width = tin_shape[0]
    mask = np.zeros(tin_shape, dtype='uint8')
    if len(tin_shape) == 3:
        mask[int(tin_width/2)-15:int(tin_width/2)+15,0:tin_width,:] = 255
    else:
        mask[int(tin_width/2)-15:int(tin_width/2)+15, 0:tin_width] = 255
    if template_size == None:
        template_size = int(tin_width / math.sqrt(2))
    pad = int(tin_width/2 - template_size/2)
    if pad < 0:
        raise Exception("template size must be less than input template size")

    templates = []
    for i in range(num_angles):
        angle = (i / num_angles) * 360
        mat_rot = cv.getRotationMatrix2D((tin_width/2, tin_width/2), angle, 1)
        img_rot = cv.warpAffine(img_template, mat_rot, (tin_width, tin_width))
        img_trim = img_rot[pad:tin_width-pad, pad:tin_width-pad]
        templates.append(img_trim)
    count = 0

    # Match templates and stack results
    match_array = None 
    start = time.time()
    for template in templates:
        match = cv.matchTemplate(img_input, template, cv.TM_CCORR_NORMED)

        if count == 0:
            match_array = np.copy(match)
        else:
            match_array = np.dstack((match_array, match))
        lmax = np.unravel_index(match.argmax(), match.shape) 
        count += 1

    end = time.time()
    print("Elapsed: ", end - start)
    return match_array


def findLocations(match_array, num_locations=1, blank_radius=32):
    # Normalize match array
    match_copy  = np.copy(match_array)
    match_min   = np.amin(match_copy)
    match_max   = np.amax(match_copy)
    match_copy  = (match_copy - match_min) / (match_max - match_min)
    shape       = match_copy.shape
    xmax        = shape[0]
    ymax        = shape[1]
    num_angles  = shape[2]
    locations = []
    for i in range(num_locations):
        value = np.amax(match_copy)
        loc = np.unravel_index(match_copy.argmax(), match_copy.shape)
        angle = (loc[2] / num_angles) * math.pi * 2
        x = loc[0]
        y = loc[1]
        # blank out area around match
        match_copy[max(0, x-blank_radius):min(xmax, x+blank_radius), max(0, y-blank_radius):min(ymax, y+blank_radius), :] = 0
        locations.append(location(value, (x,y), angle))
        print(str(locations[-1]))
    return locations


def hueFilter(img_input):
    """ Takes a color image, returns a grayscale image with areas matching the resistor hue boosted """
    img_hsv = cv.cvtColor(img_input, cv.COLOR_BGR2HSV)
    hue_low = 15
    hue_high = 20
    thresh_low = np.array([hue_low, 20, 150])
    thresh_high = np.array([hue_high, 255, 255])
    mask = cv.inRange(img_hsv, thresh_low, thresh_high)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    mask = cv.erode(mask, kernel)
    mask = cv.dilate(mask, kernel)
    mask = (cv.GaussianBlur(mask, (3,3), 3).astype('float32') / 255) * 2.0 + 0.5
    img_gray = cv.cvtColor(img_input, cv.COLOR_BGR2GRAY)
    img_proc = np.minimum(img_gray * mask, np.ones(mask.shape)*255)
    return img_proc.astype('uint8')

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('template', metavar='TEMPLATE_IMAGE', type=str, help="Template image file")
    parser.add_argument('input', metavar='INPUT_IMAGE', type=str, help="Input images to search")
    parser.add_argument('--tsize', type=int, default=None, help="Trim template to this width")
    parser.add_argument('--num_angles', '-n', type=int, default=16, help='Number of angles to rotate template through')
    parser.add_argument('--num_locations', '-l', type=int, default=5, help='Number of locations to display')
    parser.add_argument('--mode', type=str, default='gray', choices=('color', 'gray'), help='Image mode')
    parser.add_argument('--scale', type=float, default=1.0, help='Prescale the image')

    args = parser.parse_args(argv)
    if args.scale <= 0 or args.scale > 1:
        print("Scale must be between 0 and 1")
        sys.exit()

    read_mode = cv.IMREAD_GRAYSCALE
    if args.mode == 'color':
        read_mode = cv.IMREAD_COLOR

    img_template = cv.imread(args.template, read_mode)

    tsize = args.tsize
    if tsize == None:
        tsize = int(img_template.shape[0] / math.sqrt(2))

    # Load template and input images
    img_input = cv.imread(args.input, read_mode)
    img_display = cv.imread(args.input, cv.IMREAD_COLOR)

    # Template matching
    img_scaled = cv.resize(img_input, None, fx=args.scale, fy=args.scale, interpolation=cv.INTER_CUBIC)
    template_scaled = cv.resize(img_template, None, fx=args.scale, fy=args.scale, interpolation=cv.INTER_CUBIC)
    match_array = match(img_scaled, template_scaled, template_size=int(tsize * args.scale), num_angles=args.num_angles)
    match_array = cv.resize(match_array, None, fx=(1/args.scale), fy=(1/args.scale), interpolation=cv.INTER_CUBIC)

    # Find peaks
    locations = findLocations(match_array, num_locations=6, blank_radius=int(tsize/2))
    locations.sort(reverse=True)

    # Display matches
    plt.imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))
    rank = 1
    for l in locations:
        x = l.location[1] + tsize/2
        y = l.location[0] + tsize/2
        x1 = x - math.cos(l.angle) * 32
        x2 = x + math.cos(l.angle) * 32
        y1 = y + math.sin(l.angle) * 32
        y2 = y - math.sin(l.angle) * 32
        plt.plot([x1, x2], [y1, y2], color=(0.2,1,0.5), linewidth=4)
        plt.annotate(str(rank), (x,y), xytext=(0,30), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=1.0),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        rank += 1

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
