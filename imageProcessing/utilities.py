#!/usr/bin/env python

# utilities.py - Utility functions for image processing based on pixel arrays

import sys

# r,g,b expected to be between 0 and 255 respectively.
# greyvalue will be an int between 0 and 255 as well.
def rgbToGreyscale(r, g, b):
    greyvalue = int(round(0.299 * r + 0.587 * g + 0.114 * b))
    return greyvalue

def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = []
    for row in range(image_height):
        new_row = []
        for col in range(image_width):
            new_row.append(initValue)
        new_array.append(new_row)

    return new_array

def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min_value = sys.maxsize
    max_value = -min_value

    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] < min_value:
                min_value = pixel_array[y][x]
            if pixel_array[y][x] > max_value:
                max_value = pixel_array[y][x]

    return(min_value, max_value)
