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
### Exercice 1
#######################
def transpose(arr: ndarray, transposeX: int, transposeY: int) -> ndarray:
    """Apply a transposition of the image

    Args:
        arr (ndarray): the image to transpose
        transposeX (int): transposition in the X direction
        transposeY (int): transposition in the Y direction

    Returns:
        ndarray: the transposed image
    """
    new_arr = np.ones(arr.shape, arr.dtype)*255

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):

            if (i - transposeX >= 0 and i - transposeX < arr.shape[0] 
                and j - transposeY >= 0 and j - transposeY < arr.shape[1]):
                new_arr[i,j] = arr[i-transposeX, j-transposeY]

    return new_arr


#######################
### Exercice 2
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
### Exercice 3
#######################
def interpolate_bilinear(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, arr: ndarray) -> float:
    if (x2 >= arr.shape[0]):
        x2 = arr.shape[0] - 1
    if (y2 >= arr.shape[1]):
        y2 = arr.shape[1] - 1

    alpha = (x3-x1)/(x2-x1) if (x2-x1) != 0 else 0
    if alpha < 0: alpha = 0
    elif alpha > 1: alpha = 1
    beta = (y3-y1)/(y2-y1) if (y2-y1) != 0 else 0
    if beta < 0: beta = 0
    elif beta > 1 : beta = 1
    
    finalValue = ((1-alpha) * (1-beta) * arr[x1,y1] 
                  + alpha * (1-beta) * arr[x2,y1]
                  + (1-alpha) * beta * arr[x1,y2]
                  + alpha * beta * arr[x2,y2] )
    return finalValue

def expand_2(arr: ndarray, ratio: float = 2) -> ndarray:
    new_arr = np.zeros((int(arr.shape[0]*ratio), int(arr.shape[1]*ratio)), arr.dtype)

    for i in range(new_arr.shape[0]):
        for j in range(new_arr.shape[1]):
            new_arr[i,j] = interpolate_bilinear(int(i/ratio), int(j/ratio), 
                                                math.ceil(i/ratio), math.ceil(j/ratio),
                                                i, j,
                                                arr)

    return new_arr


#######################
### Exercice 4
#######################
def rotate(arr: ndarray, angle: float = 45) -> ndarray:
    """Apply a rotation on the image

    Args:
        arr (ndarray): the image to rotate
        angle (float, optional): the angle for the rotation. Defaults to 45.

    Returns:
        ndarray: the rotated image
    """
    # 0) Angles must be converted from degree to radian
    cosAngle = math.cos(angle * math.pi / 180)
    sinAngle = math.sin(angle * math.pi / 180)

    # 1) Create a new image with added width and height
    max_size = max(arr.shape[0], arr.shape[1])
    new_arr = np.zeros((arr.shape[0] + 2*max_size, arr.shape[1] + 2*max_size), arr.dtype)

    # 2) Compute the new pixel values
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            (translatedX, translatedY) = change_coordinates_with_image_center_as_origin(i, j, arr)

            newX = int(cosAngle * translatedX - sinAngle * translatedY)
            newY = int(sinAngle * translatedX + cosAngle * translatedY)

            (finalX, finalY) = change_coordinates_with_image_center_as_origin(newX + arr.shape[0], newY + arr.shape[1], arr, reverse=True)
            
            new_arr[finalX, finalY] = arr[i,j]
    
    new_arr = np.trim_zeros(new_arr)

    return new_arr

def change_coordinates_with_image_center_as_origin(x: int, y: int, arr: ndarray, reverse: bool = False) -> tuple[int, int]:
    """Compute the pixel coordinates when the image has its center as a (0,0) origin

    Args:
        x (int): the x coordinate of the pixel
        y (int): the y coordinate of the pixel
        arr (ndarray): the image
        reverse (bool optional): if False, convert to 'image center as origin',
            If True, convert back to (0,0) as the corner of the image. Default to False.

    Returns:
        tuple[float,float]: new X and Y coordinates in the new plan
    """
    middleX = arr.shape[0] // 2
    middleY = arr.shape[1] // 2

    if not reverse:
        newX = x - middleX
        newY = y - middleY
    else:
        newX = x + middleX
        newY = y + middleY
    
    return (newX, newY)

    
