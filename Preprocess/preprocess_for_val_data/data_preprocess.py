import random
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor    # concurrent and itertools are belong to python version, no need to install
import itertools
from concurrent.futures import as_completed
import cv2


def random_crop(img, crop_size=(64, 64)):
    width, height = img.size

    # 計算裁剪的左上角座標
    left = random.randint(0, max(0, width - crop_size[0]))
    top = random.randint(0, max(0, height - crop_size[1]))

    # 裁剪圖片
    img_cropped = img.crop((left, top, left + crop_size[0], top + crop_size[1]))

    return img_cropped

def window_scaning(img, window_size=32):
    img_windows = []
    height, width = img.shape[:2]
    for i in range(0, width, window_size):
        for j in range(0, height, window_size):
            if i + window_size <= width and j + window_size <= height:
                img_windows.append(img[j:j+window_size, i:i+window_size])
    return img_windows

def process_image(image_name, images_folder_path, images_output_folder_path, crop_size, iteration):
    img = cv2.imread(os.path.join(images_folder_path, image_name))
    original_image_name = image_name.split('.')[0]
    img_windows = window_scaning(img, crop_size[0])
    for i, img_window in enumerate(img_windows):
        
        img_window_gray = cv2.cvtColor(img_window, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(img_window_gray >= 215)
        black_pixels = np.sum(img_window_gray <= 40)
        white_pixels_percentage = white_pixels / (crop_size[0] * crop_size[1])
        black_pixels_percentage = black_pixels / (crop_size[0] * crop_size[1])
        # if white_pixels_percentage <= 0.1 and black_pixels_percentage <= 0.1:
        if white_pixels_percentage <= 0.45:
            image_name = original_image_name + f'_{iteration}_{i}.png'
            cv2.imwrite(os.path.join(images_output_folder_path, image_name), img_window)


if __name__ == '__main__':
#     multiprocessing.freeze_support()

    images_folder_path = "../data/val_data_without_background/"
    images_output_folder_path = "/peterstone/data/val_data_window_croped/validation/val_data_32_32"
    image_names = os.listdir(images_folder_path)
    crop_size=(32, 32)
    iteration = 1

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        
        args_list = list(itertools.product(range(iteration), image_names))
        for args in args_list:
            future = executor.submit(process_image, args[1], images_folder_path, images_output_folder_path, crop_size, args[0])    # args[0] = iteration = 1, args[1] = all the image names
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures), desc='Overall Progress'):
            pass

    # if your computer can not support multi processing, please uncomment the following code

    # for image_name in image_names:
    #     process_image(image_name, images_folder_path, images_output_folder_path, crop_size, iteration)
    