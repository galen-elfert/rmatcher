# rmatcher
OpenCV for locating resistors in a pile. 

Requires opencv, numpy, and matplotlib python packages. 

To recreate the conda environment:

`conda env create -f environment.yml`

## labeller.py
A tool for marking locations of resistors in an image, storing them in a csv. 

`python labeller.py INPUT_IMAGE`

## makeTemplate.py
Loads one or more labelled images and extracts and sums the marked resistors to generate a template image for matching

`python makeTemplate.py [-h] [--output FILE] [--size SIZE] [--mode {gray,rgb}] INPUT_IMAGE1 [INPUT_IMAGE2 ...]`

## matcher.py
Rotates a template image through a number of angles and runs template matching on all of them to determine location and angle of resistors. Displays a visualization of the best match for each angle. 

`python matcher.py INPUT_IMAGE TEMPLATE_IMAGE`

## TODO
- Improve matcher with quadtree search algorithm to find n best matches everywhere. 
- Generate mask from differences in successive images, to speed up repeat matching. 
- Port matching program to C/++. 
- Read resistor values. 
