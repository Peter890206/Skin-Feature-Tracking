# Preprocess for DFE

## Summary 
This folder is prepared for DFE.
The **crop_image.py** is to crop the images centered on the predicted coordinates from Cotracker.
The cropped images are used for DFE, we call this approach "Cotracker-DFE" in my thesis.
You can choose the size of the cropped image, we usually set it to 49 * 49 (w = 24).
The **video_frame2image.py** is to convert the video to images for DFE.
[added by Peter]

## How to use **crop_image.py**?
**1.** Firstly, you need to prepare a folder with images.

**2.** Setup the width of cropped image, if you want to generate several size of cropped image, you can setup the parameter in the for loop condition.

**3.** Run the code.

## How to use **video_frame2image.py**?
**1.** Firstly, setup the input video path and output path.

**2.** Setup the total number of frames in the output video and the interval between two frames.

**3.** Run the code.



### Notes

- For **video_frame2image.py**, the limitation of the output frames amount is shorter than 100000 frames.
If you want to generate more than this number, you need to change the if condition in the code.

- For **crop_image.py**, the output format is png, you can't change it to jpg because the jpg format will change the image quality every time.
Accordingly, if you are using the jpg format, you will get the different results every time using the same data.






