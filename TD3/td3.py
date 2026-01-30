import cv2 as cv
import functions
import argparse
import random


# Exercice 1 : Filtre moyenneur
def exo1(k: int = 3):
    img = functions.read_image("cameraman.jpg", cv.CV_8UC1)
    mean_filtered_img = functions.mean_filter(img, k)
    functions.show_two_images(img, mean_filtered_img, f"Mean filter with k = {k}")

# Exercice 2 : Produit de convolution
def exo2():
    print("Nothing to test for the exercice 2 (convolution product).")

# Exercice 3 : Détecteur de contours de Sobel
def exo3():
    img = functions.read_image("cameraman.jpg", cv.CV_8UC1)
    sobel_edges_img = functions.edgeSobel(img)
    functions.show_two_images(img, sobel_edges_img, "Sobel edges")

# Exercice 4 : Filtre bilatéral
def exo4(k: int = 3):
    img = functions.read_image("cameraman.jpg", cv.CV_8UC1)
    bilateral_filtered_img = functions.bilateralFilter(img, k)
    functions.show_two_images(img, bilateral_filtered_img, f"Bilateral filter of size k = {k}")

# Exercice 5 : Filtre médian
def exo5(k: int = 3):
    img = functions.read_image("92_mini.jpg", cv.CV_8UC1)
    # random pixels white and black, with probability p=1/10
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_value = random.random()*10
            if random_value < 1:
                img[i,j] = 0 if random_value < 0.5 else 255

    median_filtered_img = functions.median(img, k)
    functions.show_two_images(img, median_filtered_img, f"Median filter of size k = {k}")
                
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e",
                      type=int,
                      help="the exercice's numero",
                      metavar="EXERCICE_NUMERO")
    parser.add_argument("-k",
                        type=int,
                        default=3,
                        help="k parameter used in exercice 1 (defaults to 3)")
    args = parser.parse_args()

    match args.e:
        case 1: exo1(args.k)
        case 2: exo2()
        case 3: exo3()
        case 4: exo4(args.k)
        case 5: exo5(args.k)
        case _:
            print(f"There is no exercice number '{args.e}' that was implemented for the TD2.")
            print("Please give a valid exercice numero, with the option -e (for example -e2) ")
            exit()