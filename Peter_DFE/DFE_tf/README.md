# DFE Tensorflow Version

## Summary 
This folder contains the code for running the Deep Feature Encoding (DFE) Tensorflow version, including training and tracking.

## Tracking

If you want to run the DFE tracking code, please prepare the input data first.
1. If you want to track the whole video, you can use **crop_video.py** in "\nordlinglab-digitalupdrs\Preprocess\preprocess_for_cotracker" to crop the video to suitable size.
Subsequently, you can use the **video_frame2image.py** in "\nordlinglab-digitalupdrs\Preprocess\preprocess_for_DFE" to convert the video to images.
2. If you want to use Cotracker-DFE, you need to use the **crop_image.py** and following the steps in **README.md** in "\nordlinglab-digitalupdrs\Preprocess\preprocess_for_DFE".

We have four codes here for tracking, including: **DFE_human_select.ipynb**, **DFE_main.py**, **DFE_tracking.py** and **DFE_tracking_tf_Peter.py**.

- The **DFE_human_select.ipynb** is almost the same as the **DFE_human_select.ipynb** in "\nordlinglab-digitalupdrs\Process_Video_DfeTracking".
But I removed the multiple points tracking part.
In this code, you can select the point you want to track by using the keyboard to move it, and call a function to track.
The function for tracking is in the **DFE_tracking.py** or **DFE_tracking_tf_Peter.py**.
At the end, it will save a npy file, which include the coordinate of each frame.

- The **DFE_main.py** is a py file for tracking.
You need to input the coordinate of first frame and call the function to track.
The function for tracking is in the **DFE_tracking.py** or **DFE_tracking_tf_Peter.py**.
At the end, it will save a npy file, which include the coordinate of each frame.
Moreover, if you want, it can print and save the Euclidean distance between the predicition and the ground truth.
Additionally, if you want to test different area size of Cotracker-DFE, you can change the **area_size** in the for loop condition.
It can save all the Euclidean distance of every area size to a npy file.

- The **DFE_tracking.py** includes the original tracking function and the tracking algorithm made by Jacob.

- The **DFE_tracking_tf_Peter.py** includes two tracking functions:
1. The original tracking function without some unnecessary parts.
2. The optimized tracking function with Rank-SSR algorithm purposed by Peter. 

[added by Peter]

## Training
The **DFE_train.ipynb** are used for training the DFE model.

The **DFE_train.ipynb** is a Jupyter Notebook file that includes the code for training the DFE model. 
It loads the dataset, defines the model architecture, and trains the model using the provided dataset. 
The trained model is saved at specified intervals during the training process.

### Enviroment setup

- **Option 1**:
1. python 3.9.13
2. numpy 1.23.5
3. opencv-python 4.8.1.78
4. regex 2022.7.9
5. pandas 1.4.4
6. scipy 1.10.0
7. tensorflow 2.5.0

- **Option 2 (Recommended)**:
Using Dockerfile in this folder to build the image.
Firstly, you need to cd to this folder.
Run the following command to build the image:
    `sudo docker build -t dfe_tf_image .`
The dfe_tf_image here is the image name, you can change it.
Once you build the image, you can run the following command to run the image:
    `sudo docker run -it --gpus=all --ipc=host -v (your folder path):(folder path in the container) --name=(your container name) dfe_tf_image bash`
If you want to check the command detailly, please check the presentation in "NordlingLab/All_Presentations/Presentations_Lab/Peter_docker_introduction_231003.pptx" in our NAS.

- **Option 3 (Most Recommended)**:
Download and load the image tar file.
The image.tar file can be found in the folder "DigitalUPDRS/QPD_Shared/Peter_DFE_Docker_Env" in our NAS.
You can just download it and use following command to load the image:
    `sudo docker load -i image.tar`
Once you build the image, you can run the following command to run the image:
    `sudo docker run -it --gpus=all --ipc=host -v (your folder path):(folder path in the container) --name=(your container name) dfe_tf_image bash`
If you want to check the command detailly, please check the presentation in "NordlingLab/All_Presentations/Presentations_Lab/Peter_docker_introduction_231003.pptx" in our NAS.



## How to use **DFE_human_select.ipynb**?
**1.** Firstly, check you are in the correct enviroment.

**2.** Setup the input folder path and output folder path.

**3.** Using "w", "a", "d", "s" to move and select the target point you want to track.

**4.** Run the code.

## How to use **DFE_main.py**?
**1.** Firstly, check you are in the correct enviroment.

**2.** Setup the input folder path, output folder path and the area size of Cotracker-DFE.
If you are not using Cotracker-DFE, just ignore the for-loop condition.

**3.** Select the tracking function you want to use and setup the coordinate of first frame.

**4.** If you want to calculate the Euclidean distance between the prediction and the ground truth here, you need to change the path of the ground truth.
If not, please comment it out.

**5.** Run the code.








