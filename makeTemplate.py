"""
Load a labelled resistor image and generate a template for matching.
"""

import sys
from PointCollector import PointCollector
import matplotlib.pyplot as plt
import pylab
import numpy as np
import cv2 as cv
import argparse

def main(argv):
    parser = argparse.ArgumentParser(description="Generate a template image")
    parser.add_argument("input", metavar="FILE", type=str, nargs='+', help="input png files")
    parser.add_argument("--output", metavar="FILE", type=str, help="output png template file")
    parser.add_argument("--size", type=int, default=128, help="width of square template image to produce")
    parser.add_argument("--mode", type=str, default="gray", choices=("gray", "rgb"), help="template image mode")
    # parser.add_argument("--vmode", type=str, default="gray", choices=("gray", "rgb"), help="variance image mode")
    args = parser.parse_args(argv)
    print(args)

    fig, ax = plt.subplots()
    if args.mode == "rgb":
        template_shape = (args.size, args.size, 3)
    else:
        template_shape = (args.size, args.size)
    img_sum = np.zeros(template_shape, dtype=np.float32)
    count = 0

    # Calculate mean of template images
    for img_file in args.input:
        pc = PointCollector(img_file, mode=args.mode)
        templates = pc.getTemplates(args.size)
        for template in templates:
            templatef = template.astype(np.float32)
            img_sum = img_sum + templatef
            count += 1
    img_avg = img_sum / count

    # Calculate variance of template images
    img_sqrdiff = np.zeros(template_shape, dtype=np.float32)
    for img_file in args.input:
        pc = PointCollector(img_file, mode=args.mode)
        templates = pc.getTemplates(args.size)
        for template in templates:
            templatef = template.astype(np.float32)
            img_sqrdiff = img_sqrdiff + (templatef - img_avg) ** 2

    img_var = img_sqrdiff / count

    # Invert and normalize variance
    shape = img_var.shape
    img_var_norm = np.zeros(shape)
    if len(shape) == 3:
        for i in range(3):
            var_min = np.amin(img_var[:,:,i])
            var_max = np.amax(img_var[:,:,i])
            img_var_norm[:,:,i] = (1.0 - ((img_var[:,:,i] - var_min) / (var_max - var_min)))
    else:
        var_min = np.amin(img_var)
        var_max = np.amax(img_var)
        img_var_norm = (1.0 - ((img_var - var_min) / (var_max - var_min)))

    # Scale the template image by the inverted variance
    img_out = (img_avg * img_var_norm).astype(np.uint8)
    # img_out = img_avg.astype(np.uint8)

    # Set the top and bottom to black and output the template image
    clip_top = int(args.size/2 - 15)
    clip_bottom = int(args.size - clip_top)
    if args.mode == 'rgb':
        img_out[0:clip_top,:,:] = 0
        img_out[clip_bottom:,:,:] = 0
        ax.imshow(img_out)
        img_out = cv.cvtColor(img_out, cv.COLOR_RGB2BGR)
    else:
        img_out[0:clip_top,:] = 0
        img_out[clip_bottom:,:] = 0
        ax.imshow(img_out, 'gray')
    if args.output:
        print("writing to ", args.output)
        cv.imwrite(args.output, img_out)
    pylab.show()

if __name__ == "__main__":
    main(sys.argv[1:])
