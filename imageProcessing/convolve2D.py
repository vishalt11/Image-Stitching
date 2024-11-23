#!/usr/bin/env python

# convolve2D.py - Convolution operations on pixel arrays in 2D

import imageProcessing.utilities as IPUtils

def computeSeparableConvolution2DOddNTapBorderZero(pixel_array, image_width, image_height, kernelAlongX, kernelAlongY = []):

    if len(kernelAlongY) == 0:
        kernelAlongY = kernelAlongX

    intermediate = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    final = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)

    # two pass algorithm for separable convolutions

    kernel_offset = len(kernelAlongX) // 2
    #print("ntap kernel offset", kernel_offset)

    for y in range(image_height):
        for x in range(image_width):
            if x >= kernel_offset and x < image_width - kernel_offset:
                convolution = 0.0
                for xx in range(-kernel_offset, kernel_offset+1):
                    convolution = convolution + kernelAlongX[kernel_offset+xx] * pixel_array[y][x+xx]
                intermediate[y][x] = convolution

    kernel_offset = len(kernelAlongY) // 2

    for y in range(image_height):
        for x in range(image_width):
            if y >= kernel_offset and y < image_height - kernel_offset:
                convolution = 0.0
                for yy in range(-kernel_offset, kernel_offset+1):
                    convolution = convolution + kernelAlongY[kernel_offset+yy] * intermediate[y+yy][x]
                final[y][x] = convolution

    return final
    