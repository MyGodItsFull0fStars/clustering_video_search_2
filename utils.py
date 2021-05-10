from typing import List

import cv2
from matplotlib import pyplot as plt
import glob


def get_1D_histogram(image, bin_size: int = 256, pixel_min_range: int = 0, pixel_max_range: int = 256) -> list:
    blue, green, red = cv2.split(image)
    hist_blue = cv2.calcHist([blue], [0], None, [bin_size], [pixel_min_range, pixel_max_range])
    hist_green = cv2.calcHist([green], [0], None, [bin_size], [pixel_min_range, pixel_max_range])
    hist_red = cv2.calcHist([red], [0], None, [bin_size], [pixel_min_range, pixel_max_range])
    weight: float = 0.33
    histogram = weight * (hist_blue + hist_green + hist_red)

    return histogram


def get_2D_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = [0, 1]
    bins = [180, 256]
    hs_range = [0, 180, 0, 256]
    histogram = cv2.calcHist([hsv_image], channels, None, bins, hs_range)

    return histogram


def plot_2D_histogram(histogram):
    plt.imshow(histogram, interpolation='nearest')
    plt.show()


def get_image_from_file_name(file_name: str):
    return cv2.imread(file_name)


def get_keyframe_image_list(high_keyframes: bool) -> List:
    file_names: List[str] = get_key_frame_file_names(high_keyframes)

    return [get_image_from_file_name(img_name) for img_name in file_names]


def get_key_frame_file_names(high_keyframes: bool) -> List[str]:
    folder_name: str = 'highkey' if high_keyframes else 'key'
    file_names: list = glob.glob('{}/*.jpg'.format(folder_name))
    # because lexical sorting fails, the file names are split and only the number is used for the sort comparison
    file_names.sort(key=lambda x: int(x.split('_')[1]))

    return file_names
