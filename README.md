# Skin-Feature-Tracking
My thesis project
## Summary

- This folder contains four deep learning tracking folders: "Co-tracker" and "Peter_DFE".

- The folder "Co-tracker" contains the codes of Co-Tracker clone from their github (https://github.com/facebookresearch/co-tracker.git), you can check details in the README.md file in this folder.

- The folder "Peter_DFE" contains two version of DFE - "DFE_tf" and "DFE_torch", both of them include the tracking and training code, you can check details in the README.md file in this folder.

- The folder "dfe_ground_truth" contains the ground truth data for tracking and the code for generating ground truth datas, you can check details in the README.md file in this folder.

- The folder "Preprocess" contains the codes for preprocessing video datas for Co-tracker, DFE and generating validation data, you can check details in the README.md file in this folder.

- The folder "Result_analysis" contains the codes for plot and analysis the tracking results, you can check details in the README.md file in this folder.

## Usage for Cotracker

Here are the steps to use Cotracker to generate the tracking results, if you have any questions, please check the README.md file in the  subfolder for details.

**1.** Prepare a input video for Cotracker.

**2.** Run the **choose_tracking_point.py** in "./Preprocess/preprocess_for_cotracker" to generate the "total_centers_for_cropping.csv" which includes the video name, the center coordinate for cropping and the cropped video width and height.

**3.** Run the **crop_video.py** in "./Preprocess/preprocess_for_cotracker" to generate the cropped video for Cotracker, you also need to put the "total_centers_for_cropping.csv" and the input videos in the same folder. (Based on our GPU limitation, our cropped video size is 360*360.)

**4.** Run the **choose_tracking_point.py** again to point the tracking points in each cropped video in the folder, after that, you will get a csv file named "sticker_coords_for_cotracker.csv" in the same folder with cropped videos.

**5.** Copy the cropped videos and the "sticker_coords_for_cotracker.csv" to the intput folder of Cotracker.

**6.** Run the **Peter_demo.py** in the "./Co-tracker/co-tracker" to generate csv files, whose amount is the same as total points you tracked, in the "output_csv" folder and the videos with marker in the "videos" folder.

## Usage for DFE
 
Please follow the instructions in the subfolder "DFE_torch".


