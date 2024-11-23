#!/usr/bin/env python

# pixelops.py - Image processing based on pixel arrays involving single pixel operations

import imageProcessing.utilities as IPUtils

def scaleAndQuantize(pixel_array, image_width, image_height, min_value, max_value):

    output_pixel_array = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)

    scale_factor = 255.0 / (max_value - min_value)

    if max_value > min_value:
        for y in range(image_height):
            for x in range(image_width):
                value = int(round((pixel_array[y][x] - min_value) * scale_factor))
                if value < 0:
                    output_pixel_array[y][x] = 0
                elif value > 255:
                    output_pixel_array[y][x] = 255
                else:
                    output_pixel_array[y][x] = value

    return output_pixel_array

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):

    (min_value, max_value) = IPUtils.computeMinAndMaxValues(pixel_array, image_width, image_height)

    print("before scaling, min value = {}, max value = {}".format(min_value, max_value))

    return scaleAndQuantize(pixel_array, image_width, image_height, min_value, max_value)

def scaleTo0And1(pixel_array, image_width, image_height):

    (min_value, max_value) = IPUtils.computeMinAndMaxValues(pixel_array, image_width, image_height)

    print("before scaling, min value = {}, max value = {}".format(min_value, max_value))

    output_pixel_array = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)

    scale_factor = 1.0 / (max_value - min_value)

    if max_value > min_value:
        for y in range(image_height):
            for x in range(image_width):
                output_pixel_array[y][x] = (pixel_array[y][x] - min_value) * scale_factor

    return output_pixel_array
