
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch

from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth
import numpy as np
from scipy import ndimage
import math
import random
import cv2

# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
def prepareRGBImageFromIndividualArrays(r_pixel_array,g_pixel_array,b_pixel_array,image_width,image_height):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = []
            triple.append(r_pixel_array[y][x])
            triple.append(g_pixel_array[y][x])
            triple.append(b_pixel_array[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):

    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage

#harris corner detection
def harrisCorner(image):
    # get the image derivatives in x and y directions
    I_x, I_y = np.gradient(image)

    I_x_sqr = I_x**2
    I_y_sqr = I_y**2
    I_x_y = I_x*I_y
    #apply gaussian filter on the M components
    I_x_sqr = ndimage.gaussian_filter(I_x_sqr, sigma=2)
    I_y_sqr = ndimage.gaussian_filter(I_y_sqr, sigma=2)
    I_x_y = ndimage.gaussian_filter(I_x_y, sigma=2)

    height, width = I_x.shape
    win_size = 3
    skip = int(win_size/2)
    alpha = 0.04
    list_corners = []
    R_values = np.zeros((height, width))
    # calculate the M matrix and corner responses and put them in R_values
    for i in range(win_size, height-win_size):
        for j in range(win_size, width-win_size):
            window_I_x_sqr = I_x_sqr[i-skip:i+skip+1,
                             j-skip:j+skip+1]
            window_I_y_sqr = I_y_sqr[i-skip:i+skip+1,
                             j-skip:j+skip+1]
            window_I_x_y = I_x_y[i-skip:i+skip+1,
                           j-skip:j+skip+1]
            sum_I_x_x = window_I_x_sqr.sum()
            sum_I_y_y = window_I_y_sqr.sum()
            sum_I_x_y = window_I_x_y.sum()
            M = np.asarray([[sum_I_x_x,sum_I_x_y],[sum_I_x_y, sum_I_y_y]])
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            R = det_M - alpha*(trace_M**2)
            R_values[i,j] = R
    # make the negative values as 0 (edges)
    R_values[R_values < 0] = 0

    win_size = 3
    R_final = np.zeros((height, width))
    # non-max suppression
    for i in range(win_size, height-win_size):
        for j in range(win_size, width-win_size):
            # check for a given window of size n, the max value in it
            max_value = np.amax(R_values[i-int(win_size/2):i+int(win_size/2)+1,
                                j-int(win_size/2):j+int(win_size/2)+1])
            # for the respective windows check if the given r value is greater or equal to max value in window,
            # then append it, if yes.
            if R_values[i,j] >= max_value and R_values[i,j] !=0:
                R_final[i,j] = R_values[i,j]
    # making it into a 1d array
    R_final_f = R_final.flatten()
    # take the top 1000 corners
    R_final_sorted = R_final_f.argsort()[-1000:]
    # get x and y coordinates
    x_id, y_id = np.unravel_index(R_final_sorted, R_final.shape)

    for i in range(0, len(x_id)):
        list_corners.append([x_id[i], y_id[i]])
    # return the corner coordinates for the respective image
    return list_corners

def nccElements(corners, image, win_size):
    skip = int(win_size/2)
    numer = []
    denom = []

    for i in range(0, len(corners)):
        window = image[int(corners[i][0])-skip:int(corners[i][0])+skip + 1,
                   int(corners[i][1])-skip:int(corners[i][1])+skip + 1]
        mean_window = np.mean(window)
        diff_sqr_sum = 0
        window = window - mean_window
        window_flat = window.flatten()
        numer.append(window_flat)
        window = window**2
        diff_sqr_sum = np.sum(window)
        denom.append(math.sqrt(diff_sqr_sum))

    return numer, denom

def featureMatching(corners_L, corners_R, image_L, image_R):
    #window size of 15x15
    win_size = 15
    skip = int(win_size/2)
    height, width = image_L.shape
    # normalizing the values in the two images
    # image_L *= int(255/image_L.max())
    # image_R *= int(255/image_R.max())

    corners_L_pruned = []
    corners_R_pruned = []
    # remove the corners close to the boundary (15x15 window crosses the image boundary for the particular corner)
    for i in range(0, len(corners_L)):
        if (int(corners_L[i][0]) - skip) >= 0 and (int(corners_L[i][0])+skip+1) <= height\
                and (int(corners_L[i][1]) - skip) >= 0 and (int(corners_L[i][1])+skip+1) <= width:
            corners_L_pruned.append(corners_L[i])

    for i in range(0, len(corners_R)):
        if (int(corners_R[i][0]) - skip) >= 0 and (int(corners_R[i][0])+skip+1) <= height\
                and (int(corners_R[i][1]) - skip) >= 0 and (int(corners_R[i][1])+skip+1) <= width:
            corners_R_pruned.append(corners_R[i])
    # store the NCC elements for final computation
    numer_L, denom_L = nccElements(corners_L_pruned, image_L, win_size)
    numer_R, denom_R = nccElements(corners_R_pruned, image_R, win_size)

    best_match_R = []

    for i in range(0, len(corners_L_pruned)):
        NCC_best = 0
        NCC_2nd_best = 0
        temp_R_coords = 0
        for j in range(0, len(corners_R_pruned)):
            numer_sum = 0
            # summing up the product of the numerator terms
            for k in range(0, len(numer_L[i])):
                numer_sum += numer_L[i][k]*numer_R[j][k]
            #finding the final denominator
            denom_prod = denom_L[i]*denom_R[j]
            NCC = numer_sum/denom_prod
            if NCC > NCC_best:
                NCC_best = NCC
                temp_R_coords = corners_R_pruned[j]
            elif NCC > NCC_2nd_best:
                NCC_2nd_best = NCC
        # choose the pairs with relatively lower similarity measure score (to choose similar points)
        if NCC_2nd_best/NCC_best < 0.8:
            best_match_R.append([corners_L_pruned[i],temp_R_coords])

    return best_match_R

def DLT(src, dst):

    src = np.asarray(src)
    dst = np.asarray(dst)
    N = len(src)
    A = np.zeros((2*N,9))

    # compute the 2 rows of each match pair of the A matrix
    for i in range(0, N):
        A[2*i,:] = [src[i][0],src[i][1],1, 0,0,0, -src[i][0]*dst[i][0],-dst[i][0]*src[i][1],-dst[i][0]]
        A[2*i+1,:] = [0,0,0, src[i][0],src[i][1],1, -src[i][0]*dst[i][1],-dst[i][1]*src[i][1],-dst[i][1]]
    # perform svd
    u, s, v_t = np.linalg.svd(A)

    #np.set_printoptions(suppress=True, precision=4)
    # last row is our flattened H matrix
    h = v_t[8]
    h[:] = [x/h[-1] for x in h]
    # reshape into 3x3 matrix
    H = [[h[0],h[1],h[2]], [h[3],h[4],h[5]], [h[6],h[7],h[8]]]

    return H

def inlier(point_pair, H):

    mappingThreshold = 1
    src = [point_pair[0][0], point_pair[0][1], 1]
    dst = [point_pair[1][0], point_pair[1][1], 1]
    H = np.asarray(H)
    src = np.asarray(src)
    # calculate the estimated destination points by multiplying source points with the transformation matrix
    new_dst = np.dot(H,src)
    new_dst[:] = [x/new_dst[-1] for x in new_dst]
    # calculate distance
    dist = np.linalg.norm(dst - new_dst)

    if dist < mappingThreshold:
        return 1, new_dst, dst
    else:
        return 0, new_dst, dst

def ransac(match_pairs, runs):

    random.seed()
    largest_inlier_count = 0
    best_H = 0
    largest_inlier_set = 0
    corresponding_dst_set = 0
    for i in range(0, runs):
        inlier_count = 0
        indices = list(range(len(match_pairs)))
        # gives 4 distinct indices from the matching pair list
        four_matches = random.sample(indices, 4)
        src = []
        dst = []
        inlier_set = []
        temp_dst_set = []
        for id in range(0, len(four_matches)):
            src.append([match_pairs[four_matches[id]][0][0], match_pairs[four_matches[id]][0][1]])
            dst.append([match_pairs[four_matches[id]][1][0], match_pairs[four_matches[id]][1][1]])

        H = DLT(src,dst)
        # apply H on all the match pairs source points (left image)
        for j in range(0, len(match_pairs)):
            s_cond, transformed_point, dst_point = inlier(match_pairs[j], H)
            # if it falls within the threshold append to the necessary lists
            if s_cond:
                inlier_count+=1
                inlier_set.append([transformed_point[0], transformed_point[1]])
                temp_dst_set.append([dst_point[0], dst_point[1]])
        # finding the best H and largest inlier list
        if inlier_count > largest_inlier_count:
            largest_inlier_count = inlier_count
            best_H = H
            largest_inlier_set = inlier_set
            corresponding_dst_set = temp_dst_set

    print("best homography:", best_H)

    final_H = DLT(largest_inlier_set, corresponding_dst_set)
    print("\n")
    print("final homography:", final_H)

    return final_H, best_H


def warp(src, dst, H):

    height = src.shape[0]
    width = src.shape[1]
    H = np.asarray(H)
    stitched_img = np.zeros((height,2*width))

    for x in range(0,height):
        for y in range(0, 2*width):
            src_pt = np.asarray([x, y, 1])
            new_dst = np.dot(H, src_pt)

            x_m = new_dst[0]/new_dst[2]
            y_m = new_dst[1]/new_dst[2]
            # print("x_m:",x_m, " ,y_m": y_m)
            # print("x:",x, "y",y)
            # print("\n")
            left_inside_cond = x in range(0,height) and y in range(0,width)
            left_outside_cond = x not in range(0, height) or y not in range(0,width)
            right_inside_cond = (0 <= x_m and x_m < height) and (0 <= y_m and y_m < width)
            right_outside_cond = (x_m < 0 or x_m >= height) or (y_m < 0 or y_m >= width)

            if right_outside_cond and left_inside_cond:
                stitched_img[x,y] = src[x,y]

            elif right_inside_cond and left_inside_cond:
                x1 = np.floor(x_m).astype(int)
                y1 = np.floor(y_m).astype(int)
                x2 = x1 + 1
                y2 = y1 + 1
                a = x_m - x1
                b = y_m - y1
                interpolated_color = (1-a)*(1-b)*dst[x1,y1] + a*(1-b)*dst[x2,y1] + a*b*dst[x2,y2] + (1-a)*b*dst[x1,y2]
                left_color = src[x,y]
                stitched_img[x,y] = (left_color+interpolated_color)/2

            elif right_outside_cond and left_outside_cond:
                stitched_img[x, y] = 0

    return stitched_img

def main():
    # comment and uncomment the image set you want to perform harris corner detection
    # for new images, just change the paths of the input image
    # image set 1

    # filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    # filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    # image set 2
    # filename_left_image = "./images/panoramaStitching/bryce_left_02.png"
    # filename_right_image = "./images/panoramaStitching/bryce_right_02.png"

    # image set 3
    filename_left_image = "./images/panoramaStitching/bryce_left_01.png"
    filename_right_image = "./images/panoramaStitching/bryce_right_01.png"

    (image_width, image_height, px_array_left_original)  = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_left_image)
    (image_width, image_height, px_array_right_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_right_image)

    image_L = np.asarray(px_array_left_original)
    image_R = np.asarray(px_array_right_original)

    print(image_L[749,999])


    # Phase 1
    # Harris corners
    list_corner_L = harrisCorner(image_L)
    list_corner_R = harrisCorner(image_R)
    '''
    # Saving the the corner coordinates into a file, so you can avoid calling harriscorner function
    # every time you want to change the featureMatching code, threshold.
    # Comment out the above harrisCorner call if you will be running featureMatching
    # multiple times on the same image set after an initial run
    np.savetxt('left.out', list_corner_L)
    np.savetxt('right.out', list_corner_R)
    list_corner_L = np.loadtxt('left.out')
    list_corner_R = np.loadtxt('right.out')
    '''
    # Phase 2
    # find the matching pairs across the left and right image
    match_pairs = featureMatching(list_corner_L,list_corner_R, image_L, image_R)
    print("Number of matching pairs:", len(match_pairs))
    np.save("match_pairs.npy", match_pairs)


    # PHASE 3
    # loading the match pair list from a file to reduce re-runs
    # uncomment the above section to recompute match pairs for other image pairs
    # Note: source -> left image, destination -> right image
    # match_pairs1 = np.load("match_pairs.npy")
    # H, best_H = ransac(match_pairs1, 10000)
    #
    # #H = [[1.34,-0.02,-277.56],[0.12,1.18,-60.5],[0.12,0.000001,1]]
    # #PHASE 4
    # stitched_img = warp(image_L, image_R, H)
    # # cv2.imshow('Stitched Image', stitched_img)
    # # cv2.waitKey()
    # cv2.imwrite("output.png", stitched_img)


    px_array_left = IPSmooth.computeGaussianAveraging3x3(px_array_left_original, image_width, image_height)
    px_array_right = IPSmooth.computeGaussianAveraging3x3(px_array_right_original, image_width, image_height)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)

    ######################################
    # visualizations
    ######################################
    # Phase 1 - harris corners
    # set the titles and display the image
    fig1, axs1 = pyplot.subplots(1, 2)

    axs1[0].set_title('Harris response left overlaid on orig image')
    axs1[1].set_title('Harris response right overlaid on orig image')
    axs1[0].imshow(px_array_left, cmap='gray')
    axs1[1].imshow(px_array_right, cmap='gray')


    # plot the corners as red point in the image

    for i in range(0, len(list_corner_L)):
        circle = Circle((int(list_corner_L[i][1]), int(list_corner_L[i][0])), 0.5, color='r')
        axs1[0].add_patch(circle)


    for i in range(0, len(list_corner_R)):
        circle = Circle((int(list_corner_R[i][1]), int(list_corner_R[i][0])), 0.5, color='r')
        axs1[1].add_patch(circle)

    pyplot.show()
    
    # Phase 2 - feature matching
    # a combined image including a red matching line as a connection patch artist (from matplotlib\)

    matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)

    pyplot.imshow(matchingImage, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matching image")

    print(len(match_pairs))
    # draw lines between each pair points
    for i in range(0, len(match_pairs)):
        pointA = (int(match_pairs[i][0][1]), int(match_pairs[i][0][0]))
        pointB = (int(match_pairs[i][1][1])+image_width, int(match_pairs[i][1][0]))
        connection = ConnectionPatch(pointA, pointB, "data", edgecolor='r', linewidth=1)
        ax.add_artist(connection)

    pyplot.show()
    


if __name__ == "__main__":
    main()

