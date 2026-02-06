import cv2 as cv
import functions_td6 as fcts
import argparse
import numpy as np


# Exercice 1 : Dilatation
def exo1(k: int, img_nb: int, path_SE: str):
    match img_nb:
        case 1: path_img = "test.png"
        case 2: path_img = "cameraman.jpg"
    img = fcts.read_image(path_img, cv.CV_8UC1)
    SE = fcts.read_image(path_SE, cv.CV_8UC1)

    match img_nb:
        case 1: dilated_img = np.invert( fcts.dilate(np.invert(img), np.invert(SE)) )
        case 2: dilated_img = fcts.dilate(img, np.invert(SE))
    fcts.show_two_images(fcts.expand(img, k), fcts.expand( dilated_img, k), f"Dilated image with structurant element : {path_SE}")


# Exercice 2 : Erosion
def exo2(k: int, img_nb: int, path_SE: str):
    match img_nb:
        case 1: path_img = "test.png"
        case 2: path_img = "cameraman.jpg"
    img = fcts.read_image(path_img, cv.CV_8UC1)
    SE = fcts.read_image(path_SE, cv.CV_8UC1)

    match img_nb:
        case 1: eroded_img = np.invert( fcts.erode( np.invert(img), np.invert(SE) ) )
        case 2: eroded_img = fcts.erode(img, np.invert(SE))    
    fcts.show_two_images(fcts.expand(img, k), fcts.expand( eroded_img, k), f"Eroded image with structurant element : {path_SE}")


# Exercice 3 : Fermeture
def exo3(k: int, img_nb: int, path_SE: str):
    match img_nb:
        case 1: path_img = "test.png"
        case 2: path_img = "cameraman.jpg"
    img = fcts.read_image(path_img, cv.CV_8UC1)
    SE = fcts.read_image(path_SE, cv.CV_8UC1)

    match img_nb:
        case 1: closed_img = np.invert( fcts.close( np.invert(img), np.invert(SE) ) )
        case 2: closed_img = fcts.close(img, np.invert(SE))
    fcts.show_two_images(fcts.expand(img, k), fcts.expand( closed_img, k), f"Close operation with structurant element : {path_SE}")


# Exercice 4 : Ouverture
def exo4(k: int, img_nb: int, path_SE: str):
    match img_nb:
        case 1: path_img = "test.png"
        case 2: path_img = "cameraman.jpg"
    img = fcts.read_image(path_img, cv.CV_8UC1)
    SE = fcts.read_image(path_SE, cv.CV_8UC1)

    match img_nb:
        case 1: opened_img = np.invert( fcts.open( np.invert(img), np.invert(SE)) )
        case 2: opened_img = fcts.open(img, np.invert(SE))
    fcts.show_two_images(fcts.expand(img, k), fcts.expand( opened_img, k), f"Open operation with structurant element : {path_SE}")


# Exercice 5 : Gradient morphologique
def exo5(k: int, img_nb: int, path_SE: str):
    match img_nb:
        case 1: path_img = "test.png"
        case 2: path_img = "cameraman.jpg"
    img = fcts.read_image(path_img, cv.CV_8UC1)
    SE = fcts.read_image(path_SE, cv.CV_8UC1)

    match img_nb:
        case 1: gradient_morph = np.invert( fcts.morphologicalGradient( np.invert(img), np.invert(SE)) )
        case 2: gradient_morph = fcts.morphologicalGradient(img, np.invert(SE))
    fcts.show_two_images(fcts.expand(img, k), fcts.expand( gradient_morph, k), f"Morphological gradient with structurant element : {path_SE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e",
                      type=int,
                      help="the exercice's numero",
                      metavar="EXERCICE_NUMERO")
    parser.add_argument("-k",
                        type=float,
                        default=2,
                        help="k parameter (defaults to 2)")
    parser.add_argument("-t",
                        type=int,
                        default=1,
                        help="Type of Structurant Element used")
    parser.add_argument("-i",
                        type=int,
                        default=1,
                        help="Image to use as an exemple")
    args = parser.parse_args()

    match args.t:
        case 1: path_SE = "SE_crux.png"
        case 2: path_SE = "SE_horizontal_line.png"
        case 3: path_SE = "SE_vertical_line.png"
        case _: 
            print("Error, parameter couldn't be used to choose a Structurant Element.")
            exit()

    match args.e:
        case 1: exo1(args.k, args.i, path_SE)
        case 2: exo2(args.k, args.i, path_SE)
        case 3: exo3(args.k, args.i, path_SE)
        case 4: exo4(args.k, args.i, path_SE)
        case 5: exo5(args.k, args.i, path_SE)
        case _:
            print(f"There is no exercice number '{args.e}' that was implemented for the TD.")
            print("Please give a valid exercice numero, with the option -e (for example -e2) ")
            exit()