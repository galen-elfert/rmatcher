# rmatcher
OpenCV for locating resistors in a pile. 

Requires opencv, numpy, and matplotlib python packages. 

The simplest way to install all dependencies is to use the Conda package manager (https://conda.io/docs/user-guide/install/download.html). 

To recreate the conda environment:

`conda env create -f environment.yml`

To start the environment:

`source activate opencv`

## labeller.py
A tool for marking locations of resistors in an image, storing them in a csv. 

`python labeller.py INPUT_IMAGE`

## makeTemplate.py
Loads one or more labelled images and extracts and sums the marked resistors to generate a template image for matching

`python makeTemplate.py [-h] [--output FILE] [--size SIZE] [--mode {gray,rgb}] INPUT_IMAGE1 [INPUT_IMAGE2 ...]`

## matcher.py
Rotates a template image through a number of angles and runs template matching on all of them to determine location and angle of resistors. Displays a visualization of the best match for each angle. 

```
usage: matcher.py [-h] [--tsize TSIZE] [--num_angles NUM_ANGLES]
                  [--num_locations NUM_LOCATIONS] [--mode {color,gray}]
                  [--scale SCALE]
                  TEMPLATE_IMAGE INPUT_IMAGE
```
Example:
`python matcher.py output/template-color.png input/input04.png`

## rmatcher
There is a C++ port of rmatcher. It has fewer options and runs only the optimized version of the algorithm. To build, simply run make. Usage is the same as rmatcher.py, but without the extra flags:

`rmatcher TEMPLATE_IMAGE INPUT_IMAGE`

The program will display the input image in a window with the best match highlighted, and print the location and angle to the stdout. 

## TODO
- Port matching program to C/++. 
- Read resistor values. 
