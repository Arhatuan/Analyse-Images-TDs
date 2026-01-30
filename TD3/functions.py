import cv2 as cv
import numpy as np
from numpy import ndarray
import math

def read_image(filePath: str, dataType: int = cv.IMREAD_COLOR) -> ndarray:
    """Get an image in the 'ndarray' format

    Args:
        filePath (str): the path to the image
        dataType (int, optional): the data type to convert to. Defaults to cv.CV_8UC3.

    Returns:
        ndarray: the image in the 'ndarray' format, and in the data type given
    """
    img = cv.imread(filePath, dataType)
    img = np.array(img)
    return img

def show_image(image: ndarray, title: str ="image without title"):
    """Show 1 image

    Args:
        image (ndarray): the image to show
        title (str, optional): a title to the showcased window. Defaults to "image without title".
    """
    cv.imshow(title, image)
    cv.waitKey(0)

def show_two_images(img1: ndarray, img2: ndarray, title: str ="image without title"):
    """Show 2 images side by side

    Args:
        img1 (ndarray): the first image
        img2 (ndarray): the second image
        title (str, optional): a title to the showcased window. Defaults to "image without title".
    """
    concatenatedImgs = np.concatenate((img1, img2), axis=1)
    cv.imshow(title, concatenatedImgs)
    cv.waitKey(0)


#######################
### Exercice 1 (and 2)
#######################
def mean_filter(arr: ndarray, k: int = 3) -> ndarray:
    """Apply a mean filter, ok size k, on a given image

    Args:
        arr (ndarray): the image on which to apply the mean filter
        k (int, optional): size of the mean filter. Defaults to 3.

    Returns:
        ndarray: the image after applying the mean filter
    """
    assert k > 0 and k % 2 == 1, "k parameter must be positive and odd"

    mean_kernel = np.ones((k,k), dtype=arr.dtype) * (1/(k*k))
    convoluted_arr = convolution(arr, mean_kernel)

    return convoluted_arr


def convolution_per_pixel(img: ndarray, i: int, j: int, filter: ndarray) -> float:
    """Apply the convolution on a given pixel of the image, and the given filter

    Args:
        img (ndarray): the image 
        i (int): the i index of the pixel
        j (int): the j index of the pixel
        filter (ndarray): the filter to convolve with

    Returns:
        int: the value of the pixel after convolution with the filter
    """
    filterSize = filter.shape[0]
    halfFilterSize = int(filterSize/2)
    height = img.shape[0]
    width = img.shape[1]
    convolutionSum = 0

    for a in range(-halfFilterSize, halfFilterSize+1):
        for b in range(-halfFilterSize, halfFilterSize+1):
            if (0 <= i+a and i+a < width and 0 <= j+b and j+b < height):
                pixelValue = img[i+a, j+b]
            else:
                pixelValue = 0 # zero-padding

            filterWeight = filter[-a+halfFilterSize, -b+halfFilterSize]
            convolutionSum += pixelValue * filterWeight
    
    return convolutionSum

def convolution(arr: ndarray, kernel: ndarray) -> ndarray:
    """Convolution of an image with a filter (the kernel)

    Args:
        arr (ndarray): the image
        kernel (ndarray): the filter

    Returns:
        ndarray: the result of the convolution between the image and the filter
    """
    new_arr = np.zeros(arr.shape, arr.dtype)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = int(convolution_per_pixel(arr, i, j, kernel))

            # if the value goes outside the type possibles values
            #   (for example outside of [0;255])
            if value < np.iinfo(arr.dtype).min:
                value = np.iinfo(arr.dtype).min
            elif value > np.iinfo(arr.dtype).max:
                value = np.iinfo(arr.dtype).max
            
            new_arr[i,j] = value

    return new_arr




#######################
### Exercice 3
#######################
def edgeSobel(arr: ndarray) -> ndarray:
    """Show Sobel edges of an image

    Args:
        arr (ndarray): the image

    Returns:
        ndarray: the sobel edges of the image
    """
    sobelKernel_dx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobelKernel_dy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    new_arr = np.zeros(arr.shape, arr.dtype)

    dx_conv = convolution(arr, sobelKernel_dx)
    dy_conv = convolution(arr, sobelKernel_dy)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i,j] = np.linalg.norm([dx_conv[i,j], dy_conv[i,j]], 2)
    return new_arr


#######################
### Exercice 4
#######################
def bilateralFilter(arr: ndarray, filterSize: int) -> ndarray:
    """Apply the bilateral filter on an image

    Args:
        arr (ndarray): the image
        filterSize (int): the filter size

    Returns:
        ndarray: the image after the bilateral filter applied
    """
    new_arr = np.zeros(arr.shape, arr.dtype)
    sigma_s = 150
    sigma_r = sigma_s

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i,j] = apply_bilateral_filter_on_pixel(arr, i, j, filterSize, sigma_s, sigma_r)
    
    return new_arr

def gaussian(x: float, sigma: float) -> float:
    """The gaussian function of a variable x, based on the given sigma

    Args:
        x (float): the variable
        sigma (float): the sigma of the gaussian function

    Returns:
        float: the value of the gaussian function
    """
    normalization_constant = 1.0 / (2*math.pi* (sigma**2))
    exponential = math.exp( -(x**2) / (2*(sigma**2)) )
    return normalization_constant * exponential

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute the distance between 2 points"""
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2 )


def apply_bilateral_filter_on_pixel(arr: ndarray, x: int, y: int, filterSize: int, sigma_s: float, sigma_r: float) -> int:
    """Apply the bilateral filter on a given pixel of an image

    Args:
        arr (ndarray): the image
        x (int): the x coordinate
        y (int): the y coordinate
        filterSize (int): the size of the kernel
        sigma_s (float): sigma for the space weight
        sigma_r (float): sigma for the range weight

    Returns:
        int: _description_
    """
    halfKernelDiameter = filterSize // 2
    value = 0
    Wp = 0

    for a in range(-halfKernelDiameter, halfKernelDiameter + 1):
        for b in range(-halfKernelDiameter, halfKernelDiameter + 1):
            
            if (0 <= x+a and x+a < arr.shape[0] and 0 <= y+b and y+b < arr.shape[1]):
                Gs = gaussian( distance(x+a, y+b, x, y), sigma_s) # space weight
                Gr = gaussian( arr[x+a, y+b] - arr[x,y], sigma_r) # range weight
                weight = Gs*Gr
                value += arr[x+a, y+b] * weight
                Wp += weight

    value = value / Wp
    return int(value)
    


#######################
### Exercice 5
#######################
def median(arr: ndarray, filterSize: int = 3) -> ndarray:
    """Apply the median filter on an image

    Args:
        arr (ndarray): the image
        filterSize (int, optional): the median filter size. Defaults to 3.

    Returns:
        ndarray: the image after being median filtered
    """
    new_arr = np.zeros(arr.shape, arr.dtype)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i,j] = median_per_pixel(arr, i, j, filterSize)

    return new_arr

def median_per_pixel(arr: ndarray, i: int, j: int, filterSize: int) -> int:
    """Apply the median filter on a given pixel

    Args:
        img (ndarray): the image
        i (int): the i index of the pixel
        j (int): the j index of the pixel
        filterSize (int): the size of the median filter

    Returns:
        int: the value of the pixel after being median filtered
    """
    assert filterSize >= 1 and filterSize % 2 == 1, "The median filter size must be strictly positive and odd."
    halfFilterSize = filterSize // 2
    width = arr.shape[0] # hauteur de l'image
    height = arr.shape[1] # largeur de l'image

    values_list = []

    # Obtain the list of values around the pixel
    for a in range(-halfFilterSize, halfFilterSize+1):
        for b in range(-halfFilterSize, halfFilterSize+1):
            if (0 <= i+a and i+a < width and 0 <= j+b and j+b < height):
                values_list.append(arr[i+a,j+b])
            else:
                values_list.append(0)
    
    # median value = middle value of sorted array (size is odd)
    sorted_values = sorted(values_list)
    median_value = sorted_values[len(sorted_values) // 2]

    return median_value





