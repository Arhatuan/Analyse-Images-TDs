import numpy as np
from numpy import ndarray
import cv2 as cv


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
### EXPAND functions
#######################
def interpolate_nearest(coordX: float, coordY: float, arr: ndarray) -> float:
    """Interpolate a point value to the nearest pixel value

    Args:
        coordX (float): x coordinate of a point
        coordY (float): y coordinate of a point
        arr (ndarray): the image

    Returns:
        float: value of the nearest pixel of the point
    """
    newX = int(coordX + 0.5)
    newY = int(coordY + 0.5)
    
    if newX >= arr.shape[0]: newX = arr.shape[0] - 1
    if newY >= arr.shape[1]: newY = arr.shape[1] - 1
    
    return arr[newX, newY]

def expand(arr: ndarray, ratio: float = 2) -> ndarray:
    """Expand an image by the given ratio

    Args:
        arr (ndarray): the image to expand
        ratio (float, optional): the ratio for the expansion. Defaults to 2.

    Returns:
        ndarray: the expanded image
    """

    new_arr = np.zeros((int(arr.shape[0]*ratio), int(arr.shape[1]*ratio)), arr.dtype)

    for i in range(new_arr.shape[0]):
        for j in range(new_arr.shape[1]):
            new_arr[i,j] = interpolate_nearest(i/ratio, j/ratio, arr)

    return new_arr

#######################
### Exercice 1
#######################

def dilate(arr: ndarray, SE: ndarray) -> ndarray:
    """Apply a structurant element on an image, and manage dilation

    Args:
        arr (ndarray): the image to change
        SE (ndarray): the structurant element

    Returns:
        ndarray: the image after applying the structurant element
    """
    assert SE.shape[0] == SE.shape[1] and SE.shape[0] % 2 == 1, "The structurant element must be of symetrical and odd sizes"
    new_arr = np.zeros(arr.shape, arr.dtype)
    
    # 1) Gets the relative positions of the structurant element that are positives
    list_relativePositions_SE = []
    half_SE_size = SE.shape[0] // 2
    for a in range(-half_SE_size, half_SE_size + 1):
        for b in range(-half_SE_size, half_SE_size + 1):
            if (SE[half_SE_size + a, half_SE_size + b] > 0):
                list_relativePositions_SE.append( (a, b) )

    # 2) For dilation : maximum of values corresponding to the structurant element around a pixel
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            relative_values = get_values_2D_structurant_element_around_pixel( (i,j), arr, list_relativePositions_SE)
            new_arr[i,j] = max(relative_values)

    return new_arr


def get_values_2D_structurant_element_around_pixel(pixel: tuple[int, int], arr: ndarray, 
                                                        list_SE_positive_positions: list[tuple[int, int]]) -> list[float]:
    """Gets the values around a pixel of the image, corresponding to the structurant element given

    Args:
        pixel (tuple[int, int]): the pixel of coordinates (x,y)
        arr (ndarray): the image
        list_SE_positive_positions (list[tuple[int, int]]): list of relative positions in the structurant element that are positives 

    Returns:
        list[float]: the values associated to the pixels corresponding to the structurant element around the original pixel
    """
    list_values = []
    for (relativePositionX, relativePositionY) in list_SE_positive_positions:
        newX = pixel[0] + relativePositionX
        newY = pixel[1] + relativePositionY
        if (0 <= newX < arr.shape[0] and 0 <= newY < arr.shape[1]):
            list_values.append( arr[newX, newY] )
        else:
            list_values.append(0) # zero-padding
    return list_values


#######################
### Exercice 2
#######################
def erode(arr: ndarray, SE: ndarray) -> ndarray:
    """Apply erosion on an image, based on a structurant element

    Args:
        arr (ndarray): the image to erode
        SE (ndarray): the structurant element

    Returns:
        ndarray: the image after erosion
    """
    return np.invert(dilate(np.invert(arr), SE))

#######################
### Exercice 3
#######################
def close(arr: ndarray, SE: ndarray) -> ndarray:
    dilated_img = dilate(arr, SE)
    return erode(dilated_img, SE)

#######################
### Exercice 4
#######################
def open(arr: ndarray, SE: ndarray) -> ndarray:
    eroded_img = erode(arr, SE)
    return dilate(eroded_img, SE)


#######################
### Exercice 5
#######################
def morphologicalGradient(arr: ndarray, SE: ndarray) -> ndarray:
    internalGradient = arr - erode(arr, SE)
    externalGradient = dilate(arr, SE) - arr
    return (internalGradient + externalGradient)





