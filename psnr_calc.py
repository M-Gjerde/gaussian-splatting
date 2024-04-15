from math import log10, sqrt
import cv2
import numpy as np


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    original = cv2.imread("/home/magnus/Downloads/raw_data_v1_part1/0000/images/IMG_20220818_173906.jpg")

    gs = cv2.imread("/home/magnus/phd/3dgs/screenshots/0000/left/1.png", 1)
    nerf = cv2.imread("/home/magnus/Downloads/stereo_dataset_v1_part1/0000/Q/center/IMG_20220818_173906.jpg", 1)

    # Resize original and gs images to match nerf's dimensions
    nerf_resized = cv2.resize(nerf, (original.shape[1], original.shape[0]))
    gs_resized = cv2.resize(gs, (original.shape[1], original.shape[0]))

    val_nerf = PSNR(original, nerf_resized)
    val_3dgs = PSNR(original, gs_resized)
    print(f"PSNR value for nerf is {val_nerf} dB, 3dgs: {val_3dgs} db")


if __name__ == "__main__":
    main()