#!/usr/bin/env python

# smoothing.py - Smoothing filter operations on pixel arrays in 2D

import imageProcessing.utilities as IPUtils
import imageProcessing.convolve2D as IPConv2D

def computeGaussianAveraging3x3(pixel_array, image_width, image_height):

    # sigma is 3 pixels
    smoothing_3tap = [0.27901, 0.44198, 0.27901]

    averaged = IPConv2D.computeSeparableConvolution2DOddNTapBorderZero(pixel_array, image_width, image_height, smoothing_3tap)

    return averaged
