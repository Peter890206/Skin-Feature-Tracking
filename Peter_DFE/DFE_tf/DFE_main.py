"""
This code performs Deep Feature Encoding (DFE) tracking on a sequence of images and calculates the Euclidean distance error between the tracking results and the ground truth.

It iterates over different area sizes and performs the following steps for each area size:
1. Sets the path to the input images and retrieves the list of filenames using `sorted_alphanumeric`.
2. Initializes variables such as `window_size` and `center_set` (initial center position for tracking).
3. Sets the path to save the tracking results and creates the directory if it doesn't exist.
4. Performs DFE tracking using the `DFE_tracking` function from the `DFE_tracking_tf_Peter` or `DFE_tracking` module.
5. Saves the tracking results as a numpy file with a specific naming convention.
6. Records the total time taken for tracking.

After tracking is completed, the code proceeds to calculate the Euclidean distance error between the tracking results and the ground truth:
1. Loads the ground truth and cotracker predictions from their respective paths.
2. Extracts the x and y coordinates from the ground truth, cotracker predictions, and tracking results.
3. Calculates the distance for Cotracker-DFE revision by subtracting the tracking results from the area size.
4. Revises the cotracker predictions by subtracting the Cotracker-DFE revision distances.
5. Calculates the Euclidean distance between the revised Cotracker-DFE predictions and the ground truth.
6. Computes the total error by summing the revised distances and appends it to the list of total errors.
7. Prints the total error for each iteration.

Finally, the code saves the total time and total errors as numpy files in the "./result" directory.
"""

import cv2
import numpy as np
import os
import re
import pandas as pd
import time
import tqdm
import DFE_tracking as dfe_original    # the original dfe tracking algorithm
import DFE_tracking_tf_Peter as dfe    # the dfe tracking algorithm made by Peter
import sys
sys.path.append("..")
from utils.utils_joscha import *
import time

# start_time = time.time()
total_time = []
total_errors = []
tf.random.set_seed(1234)

# Iterate over different area sizes
for i in range(24, 25, 2):
    start_time = time.time()
    path = f"./crop_image_handmole_{i}_{i}_avg/"    #./crop_image_handmole_24_24_avg/    ./pd_frames_reduced/    ./crop_image_handmole_20_20/
    filenames=sorted_alphanumeric(os.listdir(path))
    filenames = filenames[0:] ##
    print(len(filenames))

    window_size=(31,31)
    print(path)
    print(filenames[0])

    # coords = np.load("./coords.npy")
    # print(coords)
    # center_set = [coords[0], coords[1]]
    area_size = i

    # Initial center position for tracking
    center_set = [[area_size, area_size]]      #[[24, 24]]    [[220, 238]]
    rect_list = []

    # Path to save the tracking results
    save_path = "./result/"    #./result/
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    img_path = path
    center = center_set
    # first_name = "handmole_DFE_tf_40_subpixel"
    first_name = "handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref"
    # name = [first_name + f"_point1_{center[0][0]}_{center[0][1]}", first_name + f"_point2_{center[1][0]}_{center[1][1]}"]
    name = [first_name + f"_{center[0][0]}_{center[0][1]}"]
    # dfe.DFE_multiple_tracking(img_path, center, save_path, name)
    prediction = dfe.DFE_tracking(img_path, center)    #change the tracking function here

    np.save(save_path + name[0] + ".npy", prediction)

    end_time = time.time()
    total_time.append(end_time-start_time)
    print(f"Total time: {end_time-start_time}")


    """
    here you can load the ground truth to print the Euclidean distance error.
    Even more, you can run mutiple test of DFE tracking in the for loop, and save the total errors.
    If you only want to run one test, you can change the for loop condition.
    If you don't want to calculate the error here, just comment it out.
    """

    # Load the ground truth and calculate the Euclidean distance error
    Ground_truth_path = "../Process_Video_DfeTracking/result/Peter_handmole_ground_truth_mean.npy"
    cotracker_path = "../Process_Video_DfeTracking/result/Handmole_cotracker_pred_220_238.npy"
    cotracker = np.load(cotracker_path)
    Ground_truth = np.load(Ground_truth_path)

    cotracker_x = cotracker[:, 0]
    cotracker_y = cotracker[:, 1]   
    Ground_truth_x = Ground_truth[:, 0]
    Ground_truth_y = Ground_truth[:, 1]  
    prediction_x = prediction[:, 0]
    prediction_y = prediction[:, 1]

    # Calculate the distance for Cotracker-DFE revision
    distance_for_efficientnet_revise_x = area_size - prediction_x
    distance_for_efficientnet_revise_y = area_size - prediction_y
    cotracker_revised_x = cotracker_x - distance_for_efficientnet_revise_x
    cotracker_revised_y = cotracker_y - distance_for_efficientnet_revise_y
    revised_distances = np.sqrt((cotracker_revised_x - Ground_truth_x)**2 + (cotracker_revised_y - Ground_truth_y)**2)
    # DFE_distances = np.sqrt((prediction_x - Ground_truth_x)**2 + (prediction_y - Ground_truth_y)**2)
    # total_error = np.sum(DFE_distances)

    # Calculate the total error
    total_error = np.sum(revised_distances)
    total_errors.append(total_error)
    print("total_error = ", total_error)

# Save the total time and total errors
np.save("./result/total_time.npy", total_time)
np.save("./result/total_errors.npy", total_errors)