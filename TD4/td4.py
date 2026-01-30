import cv2 as cv
import functions_td4 as fcts
import argparse

# Exercice 1 : Transposée d'une image
def exo1():
    img = fcts.read_image("patatoide.png", cv.IMREAD_COLOR)
    transposed_img = fcts.transpose(img, -40, 40)
    fcts.show_two_images(img, transposed_img, "Transposed image")


# Exercice 2 : Agrandissement d'image et interpolation par plus proche voisin
def exo2(k: int):
    img = fcts.read_image("92_mini.jpg", cv.CV_8UC1)
    bigger_img = fcts.expand(img, k)
    fcts.show_image(img, "Original image")
    fcts.show_image(bigger_img, f"Bigger image (ratio = {k})")

# Exercice 3 : Agrandissement d'image et interpolation bilinéaire
def exo3(k: int):
    img = fcts.read_image("92_mini.jpg", cv.CV_8UC1)
    bigger_img = fcts.expand_2(img, k)
    fcts.show_image(img, "Original image")
    fcts.show_image(bigger_img, f"Bigger image (ratio = {k})")

# Exercice 4 : Rotation
def exo4(degree: float = 45):
    img = fcts.read_image("92_mini.jpg", cv.CV_8UC1)
    rotated_img = fcts.rotate(img, degree)
    fcts.show_image(rotated_img, f"Rotated image with angle of degree {degree}")
    pass

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
    parser.add_argument("-d",
                        type=float,
                        default=45,
                        help="degree of rotation for exercice 4 (defaults to 45)")
    args = parser.parse_args()

    match args.e:
        case 1: exo1()
        case 2: exo2(args.k)
        case 3: exo3(args.k)
        case 4: exo4(args.d)
        case _:
            print(f"There is no exercice number '{args.e}' that was implemented for the TD2.")
            print("Please give a valid exercice numero, with the option -e (for example -e2) ")
            exit()