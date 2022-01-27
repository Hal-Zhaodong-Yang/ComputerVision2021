#!/usr/bin/python3

import numpy as np
from torch import nn
import torch
from typing import Tuple
import copy
import pdb
import time
import matplotlib.pyplot as plt


"""
Authors: Vijay Upadhya, John Lambert, Cusuh Ham, Patsorn Sangkloy, Samarth
Brahmbhatt, Frank Dellaert, James Hays, January 2021.

Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells. 
"""

SOBEL_X_KERNEL = torch.tensor(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],dtype=torch.float32)
SOBEL_Y_KERNEL = torch.tensor(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ],dtype=torch.float32)


#TODO 1
def compute_image_gradients(image_bw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use convolution with Sobel filters to compute the image gradient at each pixel.

    Args:
        image_bw: A torch tensor of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image w.r.t. y-direction
    """

    # Create convolutional layer
    conv2d = nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        bias=False,
        padding=(1,1),
        padding_mode='zeros'
    )

    # Torch parameter representing (2, 1, 3, 3) conv filters

    # There should be two sets of filters: each should have size (1 x 3 x 3)
    # for 1 channel, 3 pixels in height, 3 pixels in width. When combined along
    # the batch dimension, this conv layer should have size (2 x 1 x 3 x 3), with
    # the Sobel_x filter first, and the Sobel_y filter second.
    
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    conv2d.weight = nn.parameter.Parameter(torch.stack([torch.unsqueeze(SOBEL_X_KERNEL,dim=0), torch.unsqueeze(SOBEL_Y_KERNEL, dim = 0)], dim = 0))
    sobel = conv2d(image_bw.unsqueeze(0).unsqueeze(0))
    Ix = sobel[0,0,].detach()
    Iy = sobel[0,1,].detach()

    return (Ix, Iy)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


#TODO 2.1
def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel
    Args:
        ksize: dimension of square kernel 
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    x = torch.arange(int(- (ksize / 2)), int(ksize / 2 + 1),dtype = torch.float32)
    gaussian_1d = torch.exp(-1 * (x * x) / (2 * sigma * sigma))
    gaussian_2d = torch.ger(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d / torch.sum(gaussian_2d)
    

    return gaussian_2d
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

#TODO 2.2
def second_moments(
    image_bw: torch.tensor,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """ Compute second moments from image.

    Compute image gradients Ix and Iy at each pixel, then mixed derivatives,
    then compute the second moments (sx2, sxsy, sy2) at each pixel, using
    convolution with a Gaussian filter.
    
    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
    
    Returns:
        sx2: array of shape (M,N) containing the second moment in the x direction
        sy2: array of shape (M,N) containing the second moment in the y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    Ix, Iy = compute_image_gradients(image_bw)
    kernel = get_gaussian_kernel_2D_pytorch(ksize, sigma)

    conv2d = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=ksize,
        bias=False,
        padding=(int(ksize / 2),int(ksize / 2)),
        padding_mode='zeros'
    )

    conv2d.weight = nn.parameter.Parameter(kernel.unsqueeze(0).unsqueeze(0))
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    second_moment = conv2d(torch.stack([torch.unsqueeze(Ixx, dim = 0), torch.unsqueeze(Iyy, dim = 0), torch.unsqueeze(Ixy, dim = 0)], dim = 0))
    sx2 = second_moment[0, 0, ].detach()
    sy2 = second_moment[1, 0, ].detach()
    sxsy = second_moment[2, 0, ].detach()

    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy


#TODO 3
def compute_harris_response_map(
    image_bw: torch.tensor,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
):
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Recall that R = det(M) - alpha * (trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
    http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.
    You may find the Pytorch function nn.Conv2d() helpful here.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score
    
    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    sxx, syy, sxsy = second_moments(image_bw, ksize, sigma)
    R = sxx * syy - sxsy * sxsy - alpha * (sxx + syy) * (sxx + syy)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

#TODO 4.1
def maxpool_numpy(R: torch.tensor, ksize: int) -> torch.tensor:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Note: the implementation is identical to my_conv2d_numpy(), except we
    replace the dot product with a max() operator. You can implement with torch or numpy functions but do not use
    torch's exact maxpool 2d function here.
    
    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d score/response map
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    maxpooled_R = torch.empty(R.size(), dtype = torch.float32)
    pad_layer = nn.ReflectionPad2d(int(ksize / 2))
    padded_R = pad_layer(R.unsqueeze(0).unsqueeze(0))
    padded_R = padded_R.squeeze(0).squeeze(0)
    for i in range(maxpooled_R.shape[0]):
        for j in range(maxpooled_R.shape[1]):
            maxpooled_R[i, j] = torch.max(padded_R[i:(i + ksize), j:(j + ksize)])
    
    return maxpooled_R
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################    

#TODO 4.2
def nms_maxpool_pytorch(R: torch.tensor, k: int, ksize: int  = 7) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """ Get top k interest points that are local maxima over (ksize,ksize) neighborhood.
    
    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d

    Threshold globally everything below the median to zero, and then
    MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
    with the maximum nearby value. Binarize the image according to
    locations that are equal to their maximum. Multiply this binary
    image, multiplied with the cornerness response values. We'll be testing
    only 1 image at a time.

    Args:
        R: score response map of shape (M,N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator
    
    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    th_layer = nn.Threshold(torch.median(R), 0)
    thresholded = th_layer(R)
    maxpool_layer = nn.MaxPool2d(ksize, padding = int(ksize / 2), stride = 1)
    maxpool_R = maxpool_layer(thresholded.unsqueeze(0).unsqueeze(0))
    maxpool_R = maxpool_R.squeeze(0).squeeze(0)

    pos = torch.ones(R.size())
    neg = torch.zeros(R.size())
    binary_img = torch.where(maxpool_R == R, pos, neg)
    masked_img = R * binary_img
    k_max = torch.topk(masked_img.view(1, -1).squeeze(), k)
    x = k_max.indices % R.shape[1]
    y = (k_max.indices - x) / R.shape[1]
    c = k_max.values
    
    return x, y, c
    



    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################    


#TODO 5.1    
def remove_border_vals(
    img: torch.tensor,
    x: torch.tensor,
    y: torch.tensor,
    c: torch.tensor
) -> Tuple[torch.tensor,torch.tensor,torch.tensor]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,)
        y: array of shape (k,)
        c: array of shape (k,)

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    xr = []
    yr = []
    cr = []    
    for i in range(x.shape[0]):
        if x[i] - 8 < 0 or x[i] + 10 > img.shape[1]:
            continue
        if y[i] - 8 < 0 or y[i] + 10 > img.shape[0]:
            continue
        xr.append(x[i])
        yr.append(y[i])
        cr.append(c[i])
    xr = torch.Tensor(xr)
    yr = torch.Tensor(yr)
    cr = torch.Tensor(cr)

    cr = nn.functional.normalize(cr, p = 1, dim = 0)



    return xr, yr, cr

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################     


#TODO 5.2
def get_harris_interest_points(image_bw: torch.tensor, k: int = 2500) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Implement the Harris Corner detector. You will find
        compute_harris_response_map(), nms_maxpool_pytorch(), and remove_border_vals() useful.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        confidences: array of dim (p,) containing the strength of each interest point
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    R = compute_harris_response_map(image_bw)
    x, y, c = nms_maxpool_pytorch(R, k)
    xr, yr, cr = remove_border_vals(image_bw, x, y, c)

    return xr, yr, cr
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################       

#TODO 6
def get_magnitudes_and_orientations(Ix: torch.tensor, Iy: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location. 
    
    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location. Square root of (Ix ^ 2  + Iy ^ 2)
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from 
            -PI to PI. (you may find torch.atan2 helpful here)
    """
    magnitudes = []#placeholder
    orientations = []#placeholder

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    magnitudes = torch.sqrt(Ix * Ix + Iy * Iy)
    orientations = torch.atan2(Iy, Ix)


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return magnitudes, orientations


#TODO 7
def get_gradient_histogram_vec_from_patch(window_magnitudes: torch.tensor, window_orientations: torch.tensor) -> torch.tensor:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms
    
    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 - 1 to the left of the "center", and feature_width/2 to
        the right. The same applies to above and below, respectively.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram 
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be added
        to the feature vector left to right then row by row (reading order).  

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. You may find numpy's np.histogram() function to be useful here.

    Args:
        window_magnitudes: (16,16) tensor representing gradient magnitudes of the patch
        window_orientations: (16,16) tensor representing gradient orientations of the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    wgh = []
    cell_size = int(window_magnitudes.shape[0] / 4)
    for i in range(4):
        for j in range(4):
            cell_mag = window_magnitudes[cell_size * i : (cell_size * i + cell_size), cell_size * j : (cell_size * j + cell_size)]
            cell_ori = window_orientations[cell_size * i : (cell_size * i + cell_size), cell_size * j : (cell_size * j + cell_size)]
            hist = np.histogram(cell_ori, bins = 8, range = (-np.pi, np.pi), weights = cell_mag)
            wgh.append(hist[0])
    
    wgh = np.array(wgh).reshape((128, 1))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return torch.from_numpy(wgh)


#TODO 8
def get_feat_vec(
    x: float,
    y: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> torch.tensor:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e. square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)
    
    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions.
    The autograder will only check for each gradient contributing to a single bin.
    
    Args:
        x: a float, the x-coordinate of the interest point
        y: A float, the y-coordinate of the interest point
        magnitudes: A torch tensor of shape (m,n), representing image gradients
            at each pixel location
        orientations: A torch tensor of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A torch tensor of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    window_mag = magnitudes[int(y - feature_width / 2 + 1) : int(y + feature_width / 2 + 1), int(x - feature_width / 2 + 1) : int(x + feature_width / 2 + 1)]
    window_ori = orientations[int(y - feature_width / 2 + 1) : int(y + feature_width / 2 + 1), int(x - feature_width / 2 + 1) : int(x + feature_width / 2 + 1)]
    wgh = get_gradient_histogram_vec_from_patch(window_mag, window_ori)
    wgh = wgh / torch.sqrt(torch.sum(wgh * wgh))
    fv = torch.sqrt(wgh)


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


#TODO 9
def get_SIFT_descriptors(
    image_bw: torch.tensor,
    X: torch.tensor,
    Y: torch.tensor,
    feature_width: int = 16
) -> torch.tensor:
    """
    This function returns the 128-d SIFT features computed at each of the input points
    Implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A torch tensor of shape (m,n), the image
        X: A torch tensor of shape (k,), the x-coordinates of interest points
        Y: A torch tensor of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fvs: A torch tensor of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    Ix, Iy = compute_image_gradients(image_bw)
    mag, ori = get_magnitudes_and_orientations(Ix, Iy)
    fvs = torch.empty((X.shape[0], 128))
    for i in range(X.shape[0]):
        fv = get_feat_vec(X[i], Y[i], mag, ori, feature_width)
        fvs[i,] = fv.transpose(0, 1)
        


    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fvs

#TODO 10
def compute_feature_distances(
    features1: torch.tensor,
    features2: torch.tensor
) -> torch.tensor:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow.
    One for-loop is the maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second set
            features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances 
            (in feature space) from each feature in features1 to each feature 
            in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    dists = torch.empty((features1.shape[0], features2.shape[0]))
    features1 = features1
    features2 = features2
    for i in range(features1.shape[0]):
        diff = features2 - features1[i]
        dists[i] = torch.sqrt(torch.sum(diff * diff, axis = 1))


    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


#TODO 11
def match_features_ratio_test(
    features1: torch.tensor,
    features2: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in
    section 7.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A torch tensor of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A torch tensor of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A torch tensor of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is an
            index in features2
        confidences: A torch tensor of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    nndr_thres = 0.75
    dists = compute_feature_distances(features1, features2)
    top2 = torch.topk(dists, 2, dim = 1, largest = False)
    nndrs = top2.values[:, 0] / top2.values[:, 1]
    
    matches = []
    confidences = []
    for i in range(dists.shape[0]):
        if nndrs[i] < nndr_thres:
            matches.append([i, top2.indices[i, 0]])
            confidences.append([nndrs[i]])
    
    matches = torch.tensor(matches)
    confidences = torch.tensor(confidences)

    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
