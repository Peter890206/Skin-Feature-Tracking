# Preprocess for Cotracker

## Summary 
This folder is prepared for Cotracker.
The **frame2video.py** is to generate the video from frames.
The **crop_video.py** is to crop the video to suitable size for Cotracker.
The **choose_tracking_point.py** is to generate the tracking points in the videos for Cotracker.
[added by Peter]

## How to use **frame2video.py**?
**1.** You need to prepare a folder of frames, and change the input path and output path in the **frame2video.py**.
The width and height need to be the same as your frames in the folder.
The output video fps is set to 10.

## How to use **crop_video.py**?
**1.** Firstly, check the enviroment included the package "moviepy".

**2.** Change the video input path and output path.

**3.** Setup the width and height of the output video.
If you want to generate the video for Cotracker, the width and height limit is 400 and 400, and the input sequence length is 2400 frames.
You also need to prepare a csv file named "total_centers_for_cropping.csv", which includes the video name and the center coordinate.
If you don't have the "total_centers_for_cropping.csv", you can first run the **choose_tracking_point.py** to generate it.

## How to use **choose_tracking_point.py**?

**1.** Firstly, change the input folder path and the output path for csv file, "total_centers_for_cropping.csv" for center, "sticker_coords_for_cotracker.csv" for tracking points.

**2.** After you run the code, you can see the first frame in your video.
At the process, you can use the keyboard "w", "s", "a", "d" to move the point, and "z" to switch the stride from 1 to 10 or 10 to 1.
After you move to the point, you can click "Enter" to save the point, and use "w", "s", "a", "d" again to move the point on the first frame of next video.
If you onlt have one video, the code will start after you click "Enter".

**3.** The output is a csv file which contains the video name and point coordinates for tracking.
If you are generating the center for cropping ground truth frames, you should update the csv file in NAS "DigitalUPDRS/QPD_Shared/dfe_ground_truth/crop_coords_for_labeling.csv".
If you are generating the coordinates of tracking points for Cotracker or center for cropping videos, you should upload the csv file in our "nordlinglab-digitalupdrs-data" repository.
 

### Enviroment setup

The python version here is what I was using in my computer.
It should be fine if you use the other version, but if you have any problem, please use the same version.

**frame2video.py and choose_tracking_point.py** need the following packages:
1. python 3.9.13
2. opencv-python 4.8.1.78
3. regex 2022.7.9
4. pandas 1.4.4

**crop_video.py** need the following packages:
1. python 3.8.18
2. moviepy 1.0.3
3. pandas 1.4.4



