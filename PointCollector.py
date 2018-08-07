#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI for generating training data from resistor image. 

Created on Sat Jul 21 13:13:57 2018

@author: Galen Elfert
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pylab
import csv
import sys
import os.path as op
import math
import time

SQRT2 = math.sqrt(2)

# Global point collector and pyplot
pc = None
fig = None
ax = None

# Class for storing a 2D point
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __str__(self):
        return "x:{}, y:{}".format(self.x, self.y)

def getMidpoint(pa, pb):
    return Point((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)

def getAngle(pa, pb):
    angle = math.atan((pb.y-pa.y)/(pb.x-pa.x))
    if pa.x > pb.x:
        angle += math.pi
    return angle

def getDistance(pa, pb):
    return math.sqrt((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2)

def rotate(Point, center, angle):
    distance = getDistance(center, Point)
    current_angle = getAngle(center, Point)
    new_angle = current_angle + angle
    return Point(center.x + distance * math.cos(new_angle), 
                 center.y + distance * math.sin(new_angle))

def drawPair(pa, pb, fig, ax):
    midpoint = getMidpoint(pa, pb)
    ax.plot([pa.x, midpoint.x], [pa.y, midpoint.y], color='cyan', lw=2)
    ax.plot([midpoint.x, pb.x], [midpoint.y, pb.y], color='yellow', lw=2)
    fig.canvas.draw()


# Point collector object
class PointCollector:
    def __init__(self, img_file, mode="rgb", fig=None, ax=None, remove_radius=10):
        self.fig = fig
        self.ax = ax
        if fig and ax:
            self.visible = True
        else:
            self.visible = False
        self.img_file = img_file
        if mode == "color":
            self.img = cv.imread(img_file, cv.IMREAD_COLOR)
        elif mode == "rgb":
            img_bgr = cv.imread(img_file, cv.IMREAD_COLOR)
            self.img = img_bgr[...,::-1]
        elif mode == "grayscale":
            self.img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
        else:
            raise ValueError("invalid mode: ", mode)
        self.mode = mode
        # Get name of matching CSV file
        img_file_base, img_file_ext = op.splitext(img_file)
        self.csv_file = img_file_base + ".csv"
        self.points = []
        self.p1 = None
        self.remove_radius = remove_radius
        self.state = 0
        self.load()

    def addPoint(self, event):
        """ event: mouse event from pyplot """
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        # print("xdata: {}, ydata: {}".format(x, y))
        if self.state == 0:
            self.p1 = Point(x, y)
            self.state = 1
        elif self.state == 1:
            self.points.append((self.p1, Point(x, y)))
            self.state = 0
        else:
            raise Exception("PointCollector got into invalid state")
        if len(self.points) % 5 == 0:
            # Save after every 5 points
            self.save()
            
    def draw(self):
        if self.visible:
            self.ax.lines.clear()
            for p in self.points:
                drawPair(p[0], p[1], self.fig, self.ax)
        
    def removePoint(self, event):
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        mp = Point(x,y)
        self.points = [p for p in self.points if getDistance(p[0], mp) > self.remove_radius and 
                                                 getDistance(p[1], mp) > self.remove_radius]
        self.draw()

    def save(self):
        print("Saving points to " + self.csv_file)
        with open(self.csv_file, 'w') as fh:
            writer = csv.writer(fh)
            writer.writerow(["ax", "ay", "bx", "by"])
            for p in self.points:
                writer.writerow([p[0].x, p[0].y, p[1].x, p[1].y])
    
    def load(self):
        try:
            with open(self.csv_file, 'r') as fh:
                print("Reading csv")
                reader = csv.reader(fh)
                # Skip header
                next(reader)
                i = 0
                for line in reader:
                    line = [int(s) for s in line]
                    self.points.append((Point(line[0], line[1]), Point(line[2], line[3])))
                    i += 1
                print("Read {} lines".format(i))
        except FileNotFoundError:
            pass
        self.draw()

    def getTemplates(self, size):
        size = int(size/2)
        for (pa, pb) in self.points:
            angle = getAngle(pa, pb)
            diag = size * SQRT2
            edge_dist1 = math.ceil(abs((diag / math.sin(0.25 * math.pi)) * (math.sin((0.25 * math.pi) + angle))))
            edge_dist2 = math.ceil(abs((diag / math.sin(0.25 * math.pi)) * (math.cos((0.25 * math.pi) + angle))))
            edge_dist = int(math.ceil(max(edge_dist1, edge_dist2)))
            center = getMidpoint(pa, pb)
            center.x = int(center.x)
            center.y = int(center.y)
            shape = self.img.shape
            height = shape[0]
            width = shape[1]
            if min(center.x, center.y, width - center.x, height - center.y) < edge_dist:
                # Skip this one, too close to edge
                continue
            img_clip = self.img[center.y - edge_dist:center.y + edge_dist, center.x - edge_dist:center.x + edge_dist]
            # Get rotation matrix
            angled = ((angle/math.pi) * 180)
            mat_rot = cv.getRotationMatrix2D((edge_dist, edge_dist), angled, 1)
            img_rot = cv.warpAffine(img_clip, mat_rot, (edge_dist*2, edge_dist*2))
            img_trim = img_rot[edge_dist - size:edge_dist + size, edge_dist - size:edge_dist + size]
            yield img_trim
        return None

def main(argv):
    print("running main")
    global pc, fig, ax

    # Load image
    if len(sys.argv) > 1:
        img_file = argv[1]
    else:
        print("Usage: python trainer.py INPUT_IMAGE")
        sys.exit()

    templates = pc.getTemplates(32)
    fig2, ax2 = plt.subplots(4,4)
    img_sum = np.ndarray((64,64), dtype=float)
    count = 0
    print(next(templates).shape)
    for i in range(4):
        for j in range(4):
            img_temp = cv.cvtColor(next(templates), cv.COLOR_BGR2GRAY)
            ax2[i,j].imshow(img_temp)
            img_sum = img_sum + img_temp
            count += 1
    print(img_sum)

    img_avg = img_sum / count
    img_out = img_avg.astype('uint8')
    print(img_out)
    fig3,ax3 = plt.subplots()
    ax3.imshow(img_avg)

    pylab.show()


if __name__ == "__main__":
    main(sys.argv)
