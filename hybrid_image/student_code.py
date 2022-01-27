# PyTorch tutorial on constructing neural networks:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import os
from typing import List, Tuple, Union
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms


# TODO - 1
def create_1d_gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    # Create a 1D Gaussian kernel using the specified standard deviation.
    # Note: ensure that the value of the kernel sums to 1.
    #
    # Args:
    #     standard_deviation (float): standard deviation of the gaussian
    # Returns:
    #     torch.FloatTensor: required kernel as a row vector

    kernel = torch.FloatTensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    k = 4 * standard_deviation + 1
    #print("sd:{}".format(standard_deviation))
    #print("the kernel k:{}".format(k))
    xrange = int(np.ceil(np.array([k])))
    mu = int(k / 2)
    #print("the kernel mu:{}".format(mu))

    kernel_list = []

    for x in range(0, xrange):
        kernel_list.append(-(x - mu) ** 2 / (2 * (standard_deviation ** 2)))
        #if x == 0:
            #print("operation:{}".format(   -196 / torch.tensor(98)   ))

    kernel = torch.FloatTensor(kernel_list)
    kernel = torch.exp(kernel)
    kernel = kernel / torch.sum(kernel)


    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return kernel


# TODO - 2
def my_1d_filter(signal: torch.FloatTensor,
                 kernel: torch.FloatTensor) -> torch.FloatTensor:
    # Filter the signal by the kernel.
    #
    # output = signal * kernel where * denotes the cross-correlation function.
    # Cross correlation is similar to the convolution operation with difference
    # being that in cross-correlation we do not flip the sign of the kernel.
    #
    # Reference:
    # - https://mathworld.wolfram.com/Cross-Correlation.html
    # - https://mathworld.wolfram.com/Convolution.html
    #
    # Note:
    # 1. The shape of the output should be the same as signal.
    # 2. You may use zero padding as required. Please do not use any other
    #    padding scheme for this function.
    # 3. Take special care that your function performs the cross-correlation
    #    operation as defined even on inputs which are asymmetric.
    #
    # Args:
    #     signal (torch.FloatTensor): input signal. Shape=(N,)
    #     kernel (torch.FloatTensor): kernel to filter with. Shape=(K,)
    # Returns:
    #     torch.FloatTensor: filtered signal. Shape=(N,)

    filtered_signal = torch.FloatTensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    filtered_list = []
    signal_length = signal.size()[0]
    kernel_length = kernel.size()[0]
    #print('the signal length is {}'.format(signal_length))
    #print('the kernel length is {}'.format(kernel_length))
    padding1 = nn.ConstantPad1d((kernel_length - 1) // 2, 0)
    padding2 = nn.ConstantPad1d((1, 0), 0)
    signal = padding1(signal)
    if (kernel_length - 1) % 2 != 0:
        signal = padding2(signal)
    for i in range(signal_length):
        filtered_list.append((signal[i:i + kernel_length] * kernel).sum())

    filtered_signal = torch.FloatTensor(filtered_list)
    #print(signal)
    #print('the output length is {}'.format(filtered_signal.size()[0]))
    #print(filtered_signal)


    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return filtered_signal


# TODO - 3
def create_2d_gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    # Create a 2D Gaussian kernel using the specified standard deviation in
    # each dimension, and no cross-correlation between dimensions,
    #
    # i.e.
    # sigma_matrix = [standard_deviation^2    0
    #                 0                       standard_deviation^2]
    #
    # The kernel should have:
    # - shape (k, k) where k = standard_deviation * 4 + 1
    # - mean = floor(k / 2)
    # - values that sum to 1
    #
    # Args:
    #     standard_deviation (float): the standard deviation along a dimension
    # Returns:
    #     torch.FloatTensor: 2D Gaussian kernel
    #
    # HINT:
    # - The 2D Gaussian kernel here can be calculated as the outer product of two
    #   vectors drawn from 1D Gaussian distributions.

    kernel_2d = torch.Tensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    kernel_1d = create_1d_gaussian_kernel(standard_deviation)
    kernel_2d = torch.ger(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / torch.sum(kernel_2d)

    #print("manual 2d kernel result:")
    #print(kernel_2d[12:16, 12:16])

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return kernel_2d


# TODO - 4
def my_imfilter(image, image_filter, image_name="Image"):
    # Apply a filter to an image. Return the filtered image.
    #
    # Args:
    #     image: Torch tensor of shape (m, n, c)
    #     filter: Torch tensor of shape (k, j)
    # Returns:
    #     filtered_image: Torch tensor of shape (m, n, c)
    #
    # HINTS:
    # - You may not use any libraries that do the work for you. Using torch to work
    #  with matrices is fine and encouraged. Using OpenCV or similar to do the
    #  filtering for you is not allowed.
    # - I encourage you to try implementing this naively first, just be aware that
    #  it may take a long time to run. You will need to get a function
    #  that takes a reasonable amount of time to run so that the TAs can verify
    #  your code works.
    # - Useful functions: torch.nn.functional.pad

    filtered_image = torch.Tensor()

    assert image_filter.shape[0] % 2 == 1
    assert image_filter.shape[1] % 2 == 1

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    k0 = image_filter.shape[0]
    k1 = image_filter.shape[1]
    pad2d = (0, 0, k1 // 2, k1 // 2, k0 // 2, k0 // 2)
    image_padded = F.pad(image, pad2d, "constant", 0)
    filtered_image = torch.zeros([image.shape[0], image.shape[1], image.shape[2]])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                filtered_image[i, j, k] = (image_padded[i : (i + k0), j : (j + k1), k] * image_filter).sum()

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    #filtered_image = F.conv2d(x, kernel, padding = kernel.shape[2] // 2, groups = self.n_channels)

    return filtered_image


# TODO - 5
def create_hybrid_image(image1, image2, filter):
    # Take two images and a low-pass filter and create a hybrid image. Return
    # the low frequency content of image1, the high frequency content of image2,
    # and the hybrid image.
    #
    # Args:
    #     image1: Torch tensor of dim (m, n, c)
    #     image2: Torch tensor of dim (m, n, c)
    #     filter: Torch tensor of dim (x, y)
    # Returns:
    #     low_freq_image: Torch tensor of shape (m, n, c)
    #     high_freq_image: Torch tensor of shape (m, n, c)
    #     hybrid_image: Torch tensor of shape (m, n, c)
    #
    # HINTS:
    # - You will use your my_imfilter function in this function.
    # - You can get just the high frequency content of an image by removing its low
    #   frequency content. Think about how to do this in mathematical terms.
    # - Don't forget to make sure the pixel values of the hybrid image are between
    #   0 and 1. This is known as 'clipping' ('clamping' in torch).
    # - If you want to use images with different dimensions, you should resize them
    #   in the notebook code.

    hybrid_image = torch.Tensor()
    low_freq_image = torch.Tensor()
    high_freq_image = torch.Tensor()

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    image1_lowfreq = my_imfilter(image1, filter)
    image2_lowfreq = my_imfilter(image2, filter)
    low_freq_image = image1_lowfreq
    high_freq_image = image2 - image2_lowfreq
    #print("high:")
    #print(high_freq_image[:20, :20, 0])
    #print("low:")
    #print(low_freq_image[:20, :20, 0])
    hybrid_image = low_freq_image + high_freq_image
    hybrid_image = torch.clamp(hybrid_image, min = 0, max = 1)

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return low_freq_image, high_freq_image, hybrid_image


# TODO - 6.1
def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    # Create a dataset of paired images from a directory.
    #
    # The dataset should be partitioned into two sets: one contains images that
    # will have the low pass filter applied, and the other contains images that
    # will have the high pass filter applied.
    #
    # Args:
    #     path: string specifying the directory containing images
    # Returns:
    #     images_a: list of strings specifying the paths to the images in set A,
    #         in lexicographically-sorted order
    #     images_b: list of strings specifying the paths to the images in set B,
    #         in lexicographically-sorted order

    images_a = []
    images_b = []

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    for filename in os.listdir(path):
        if filename[1] == 'a':
            images_a.append(path + '//' + filename)
        if filename[1] == 'b':
            images_b.append(path + '//' + filename)
    images_a = sorted(images_a)
    images_b = sorted(images_b)

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    #print("images_a:")
    #print(images_a)
    #print("images_b")
    #print(images_b)
    return images_a, images_b


# TODO - 6.2
def get_cutoff_standardddeviations(path: str) -> List[int]:
    # Get the cutoff standard deviations corresponding to each pair of images
    # from the cutoff_standarddeviations.txt file
    #
    # Args:
    #     path: string specifying the path to the .txt file with cutoff standard
    #         deviation values
    # Returns:
    #     List[int]. The array should have the same
    #         length as the number of image pairs in the dataset

    cutoffs = []

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    f = open(path,"r")
    for everyline in f:
        cutoffs.append(everyline)
    f.close()
    cutoffs = [int(x) for x in cutoffs]

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    #print("cutoffs:")
    #print(cutoffs)
    return cutoffs

# TODO - 6.3


class HybridImageDataset(data.Dataset):
    # Hybrid images dataset
    def __init__(self, image_dir: str, cf_file: str) -> None:
        # HybridImageDataset class constructor.
        #
        # You must replace self.transform with the appropriate transform from
        # torchvision.transforms that converts a PIL image to a torch Tensor. You can
        # specify additional transforms (e.g. image resizing) if you want to, but
        # it's not necessary for the images we provide you since each pair has the
        # same dimensions.
        #
        # Args:
        #     image_dir: string specifying the directory containing images
        #     cf_file: string specifying the path to the .txt file with cutoff
        #         standard deviation values

        images_a, images_b = make_dataset(image_dir)

        cutoffs = get_cutoff_standardddeviations(cf_file)

        self.transform = transforms.Compose([transforms.ToTensor()])

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        
        self.length = len(images_a)
        self.imagelist_a = images_a
        self.imagelist_b = images_b
        self.cutofflist = cutoffs

        if self.transform is None:
            raise NotImplementedError

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

    def __len__(self) -> int:
        # Return the number of pairs of images in dataset

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        return self.length

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Return the pair of images and corresponding cutoff standard deviation
        # value at index `idx`.
        #
        # Since self.images_a and self.images_b contain paths to the images, you
        # should read the images here and normalize the pixels to be between 0 and 1.
        # Make sure you transpose the dimensions so that image_a and image_b are of
        # shape (c, m, n) instead of the typical (m, n, c), and convert them to
        # torch Tensors.
        #
        # If you want to use a pair of images that have different dimensions from
        # one another, you should resize them to match in this function using
        # torchvision.transforms.
        #
        # Args:
        #     idx: int specifying the index at which data should be retrieved
        # Returns:
        #     image_a: Tensor of shape (c, m, n)
        #     image_b: Tensor of shape (c, m, n)
        #     cutoff: int specifying the cutoff standard deviation corresponding to
        #         (image_a, image_b) pair
        #
        # HINTS:
        # - You should use the PIL library to read images
        # - You will use self.transform to convert the PIL image to a torch Tensor

        image_a = torch.Tensor()
        image_b = torch.Tensor()
        cutoff = 0

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        
        
        pil_img_a = PIL.Image.open(self.imagelist_a[idx])
        pil_img_b = PIL.Image.open(self.imagelist_b[idx])

        image_a = self.transform(pil_img_a)
        #print("image_a")
        #print(image_a)
        #print("image_b path")
        #print(self.imagelist_b[idx])
        image_b = self.transform(pil_img_b)
        #print("image_b")
        #print(pil_img_b)
        #print(image_b)

        cutoff = self.cutofflist[idx]


        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return image_a, image_b, cutoff


# TODO - 7
class HybridImageModel(nn.Module):
    def __init__(self):
        # Initializes an instance of the HybridImageModel class.
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_standarddeviation: int) -> torch.Tensor:
        # Returns a Gaussian kernel using the specified cutoff standard deviation.
        #
        # PyTorch requires the kernel to be of a particular shape in order to apply
        # it to an image. Specifically, the kernel needs to be of shape (c, 1, k, k)
        # where c is the # channels in the image.
        #
        # Start by getting a 2D Gaussian kernel using your implementation from earlier,
        # which will be of shape (k, k). Then, let's say you have an RGB image, you
        # will need to turn this into a Tensor of shape (3, 1, k, k) by stacking the
        # Gaussian kernel 3 times.
        #
        # Args:
        #     cutoff_standarddeviation: int specifying the cutoff standard deviation
        # Returns:
        #     kernel: Tensor of shape (c, 1, k, k) where c is # channels
        #
        # HINTS:
        # - Since the # channels may differ across each image in the dataset, make
        #   sure you don't hardcode the dimensions you reshape the kernel to. There
        #   is a variable defined in this class to give you channel information.
        # - You can use torch.reshape() to change the dimensions of the tensor.
        # - You can use torch's repeat() to repeat a tensor along specified axes.

        kernel = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        #print("cutoff_standarddeviation:")
        #print(cutoff_standarddeviation)
        kernel2d = create_2d_gaussian_kernel(cutoff_standarddeviation)
        #print("kernel2d_crop:")
        #print(kernel2d[12:16, 12:16])
        #print("get kernel sd:{}".format(cutoff_standarddeviation))
        kernel = kernel2d.unsqueeze(0)
        kernel = kernel.unsqueeze(0)

        kernel = kernel.repeat(self.n_channels, 1, 1, 1)

        #print(kernel.size())


        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return kernel

    def low_pass(self, x, kernel):
        # Apply low pass filter to the input image.
        #
        # Args:
        #     x: Tensor of shape (b, c, m, n) where b is batch size
        #     kernel: low pass filter to be applied to the image
        # Returns:
        #     filtered_image: Tensor of shape (b, c, m, n)
        #
        # HINT:
        # - You should use the 2d convolution operator from torch.nn.functional.
        # - Make sure to pad the image appropriately (it's a parameter to the
        #   convolution function you should use here!).
        # - Pass self.n_channels as the value to the "groups" parameter of the
        #   convolution function. This represents the # of channels that the filter
        #   will be applied to.

        filtered_image = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        
        #print("x:")
        #print(x)
        #print(x.size())
        filtered_image = F.conv2d(x, kernel, padding = kernel.shape[2] // 2, groups = self.n_channels)

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return filtered_image

    def forward(self, image1, image2, cutoff_standarddeviation):
        # Take two images and creates a hybrid image. Returns the low frequency
        # content of image1, the high frequency content of image 2, and the hybrid
        # image.
        #
        # Args:
        #     image1: Tensor of shape (b, m, n, c)
        #     image2: Tensor of shape (b, m, n, c)
        #     cutoff_standarddeviation: Tensor of shape (b)
        # Returns:
        #     low_frequencies: Tensor of shape (b, m, n, c)
        #     high_frequencies: Tensor of shape (b, m, n, c)
        #     hybrid_image: Tensor of shape (b, m, n, c)
        #
        # HINTS:
        # - You will use the get_kernel() function and your low_pass() function in
        #   this function.
        # - Don't forget to make sure to clip the pixel values >=0 and <=1. You can
        #   use torch.clamp().
        # - If you want to use images with different dimensions, you should resize
        #   them in the HybridImageDataset class using torchvision.transforms.

        self.n_channels = image1.shape[1]

        low_frequencies = torch.Tensor()
        high_frequencies = torch.Tensor()
        hybrid_image = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        kernel = self.get_kernel(float(cutoff_standarddeviation))

        kernel = self.get_kernel(7)
        #print(kernel[0, :, 12:16, 12:16])
        low_frequencies = self.low_pass(image1, kernel)
        low_freq_image2 = self.low_pass(image2, kernel)
        high_frequencies = image2 - low_freq_image2
        hybrid_image = low_frequencies + high_frequencies
        hybrid_image = torch.clamp(hybrid_image, min = 0, max = 1)


        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return low_frequencies, high_frequencies, hybrid_image


# TODO - 8
def my_median_filter(image: torch.FloatTensor, filter_size: Union[tuple, int]) -> torch.FloatTensor:
    """
    Apply a median filter to an image. Return the filtered image.
    Args
    - image: Torch tensor of shape (m, n, 1) or Torch tensor of shape (m, n).
    - filter: Torch tensor of shape (k, j). If an integer is passed then all dimensions
              are considered of the same size. Input will always have odd size.
    Returns
    - filtered_image: Torch tensor of shape (m, n, 1)
    HINTS:
    - You may not use any libraries that do the work for you. Using torch to work
     with matrices is fine and encouraged. Using OpenCV/scipy or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take a long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Useful functions: torch.median and torch.nn.functional.pad
    """
    if len(image.size()) == 3:
        assert image.size()[2] == 1

    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert filter_size[0] % 2 == 1
    assert filter_size[1] % 2 == 1

    filtered_image = torch.Tensor()

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    k0 = filter_size[0]
    k1 = filter_size[1]

    if len(image.size()) == 2:
        image = torch.unsqueeze(image, 2)
    pad2d = (0, 0, k1 // 2, k1 // 2, k0 // 2, k0 // 2)
    image_padded = F.pad(image, pad2d, "constant", 0)
    filtered_image = torch.zeros([image.shape[0], image.shape[1], image.shape[2]])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                neighbor = image_padded[i : (i + k0), j : (j + k1), k]
                neighbor = torch.reshape(neighbor, (-1,))
                filtered_image[i, j, k] = float(torch.median(neighbor))

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return filtered_image


#############################################################################
# Extra credit opportunity (for UNDERGRAD) below
#
# Note: This part is REQUIRED for GRAD students
#############################################################################

# Matrix multiplication helper
def complex_multiply_real(m1, m2):
    # Take the one complex tensor matrix and a real matrix, and do matrix multiplication
    # Args:
    #     m1: the Tensor matrix (m,n,2) which represents complex number;
    #         E.g., the real part is t1[:,:,0], the imaginary part is t1[:,:,1]
    #     m2: the real matrix (m,n)
    # Returns
    #     U: matrix multiplication result in the same form as input m1

    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2
    imag2 = torch.zeros(real2.shape)
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


# Matrix multiplication helper
def complex_multiply_complex(m1, m2):
    # Take the two complex tensor matrix and do matrix multiplication
    # Args:
    #     t1, t2: the Tensor matrix (m,n,2) which represents complex number;
    #             E.g., the real part is t1[:,:,0], the imaginary part is t1[:,:,1]
    # Returns
    #     U: matrix multiplication result in the same form as input

    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2[:, :, 0]
    imag2 = m2[:, :, 1]
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


# TODO - 9
def dft_matrix(N):
    # Take the square matrix dimension as input, generate the DFT matrix correspondingly
    #
    # Args:
    #     N: the DFT matrix dimension
    # Returns:
    #     U: the generated DFT matrix (torch.Tensor) of size (N,N,2);
    #         the real part is represented by U[:,:,0],
    #         and the complex part is represented by U[:,:,1]

    U = torch.Tensor()

    torch.pi = torch.acos(torch.zeros(1)).item() * \
        2  # which is 3.1415927410125732

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    U = torch.zeros(N, N, 2)
    for i in range(N):
        for j in range(N):
            U[i, j, 0] = -2 * torch.pi * i * j / N
            U[i, j, 1] = -2 * torch.pi * i * j / N
    
    U[:, :, 0] = torch.cos(U[:, :, 0])
    U[:, :, 1] = torch.sin(U[:, :, 1])
    U = U / N

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return U


# TODO - 10
def my_dft(img):
    # Take a square image as input, perform Discrete Fourier Transform for the image matrix
    # This function is expected to behave similar as torch.rfft(x,2,onesided=False) except a scale parameter
    #
    # Args:
    #     img: a 2D grayscale image (torch.Tensor) whose width equals height, size: (N,N)
    # Returns:
    #     dft: the DFT results of img; the size should be (N,N,2),
    #          where the real part is dft[:,:,0], while the imag part is dft[:,:,1]
    #
    # HINT:
    # - We provide two function to do complex matrix multiplication:
    #   complex_multiply_real and complex_multiply_complex

    dft = torch.Tensor()

    assert img.shape[0] == img.shape[1], "Input image should be a square matrix"

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################


    U = dft_matrix(img.shape[0])
    #U = U * img.shape[0]
    dft = complex_multiply_real(U, img)
    dft = complex_multiply_complex(dft, U)


    

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return dft


# TODO - 11
def dft_filter(img):
    # Take a square image as input, performs a low-pass filter and return the filtered image
    #
    # Args
    # - img: a 2D grayscale image whose width equals height, size: (N,N)
    # Returns
    # - img_back: the filtered image whose size is also (N,N)
    #
    # HINTS:
    # - You will need your implemented DFT filter for this function
    # - We don't care how much frequency you want to retain, if only it returns reasonable results
    # - Since you already implemented DFT part, you're allowed to use the torch.ifft in this part for convenience, though not necessary

    img_back = torch.Tensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    dft_img = my_dft(img)

    N = dft_img.shape[0]

    freq_rate = 0.98

    for i in range(N):
        for j in range(N):
            for k in range(2):
                if (i > freq_rate * N) or (j > freq_rate * N):
                    dft_img[i, j, k] = 0
    
    U = dft_matrix(img.shape[0])
    re_dft = complex_multiply_complex(U, dft_img)
    re_dft = complex_multiply_complex(re_dft, U)
    img_back = re_dft[:, :, 0]
    #print(img_back.size())
    img_back = torch.flip(img_back, [0, 1])

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return img_back
