import cv2
import numpy as np
from pm import PatchMatch

def check_image(image, name="Image"):
    if image is None:
        print(f"{name} data not loaded.")
        return False
    return True

def check_dimensions(img1, img2):
    if img1.shape[:2] != img2.shape[:2]:
        print("Images' dimensions do not correspond.")
        return False
    return True

def main(argv):
    alpha = 0.9
    gamma = 10.0
    tau_c = 10.0
    tau_g = 2.0

    # Reading images
    img1 = cv2.imread(argv[1], cv2.IMREAD_COLOR)
    img2 = cv2.imread(argv[2], cv2.IMREAD_COLOR)

    # Image loading check
    if not check_image(img1, "Image 1") or not check_image(img2, "Image 2"):
        return 1

    # Image sizes check
    if not check_dimensions(img1, img2):
        return 1

    # Processing images
    patch_match = PatchMatch(alpha, gamma, tau_c, tau_g)
    patch_match.set(img1, img2)
    patch_match.process(3)
    patch_match.postprocess()

    disp1 = patch_match.get_left_disparity_map()
    disp2 = patch_match.get_right_disparity_map()

    disp1 = cv2.normalize(disp1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disp2 = cv2.normalize(disp2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    try:
        cv2.imwrite("left_disparity.png", disp1)
        cv2.imwrite("right_disparity.png", disp2)
    except Exception as e:
        print("Disparity save error.")
        print(str(e))
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
