# DFE Pytorch Version (Main)

## Summary 
This folder contains the code for running the Deep Feature Encoding (DFE) Pytorch version, including training and tracking.
The **DFE_tracking_torch_Peter.py** is used for tracking, and **DFE_train_torch.py** is used for training DFE autoencoder.
Futhermore, the **DFE_eval_torch.py** is used for evaluation.

### Tracking

If you want to run the DFE tracking code, please prepare the input data first.
1. If you want to track the whole video, you can use **crop_video.py** in "Skin-Feature-Tracking\Preprocess\preprocess_for_cotracker" to crop the video to suitable size.
Subsequently, you can use the **video_frame2image.py** in "Skin-Feature-Tracking\Preprocess\preprocess_for_DFE" to convert the video to images.
2. If you want to use Cotracker-DFE, you need to use the **crop_image.py** and following the steps in **README.md** in "Skin-Feature-Tracking\Preprocess\preprocess_for_DFE".

### Training

If you want to train the DFE autoencoder, please prepare the training dataset first.
If you want to create your own dataset, you can use the **data_preprocess.py** in "Skin-Feature-Tracking\Preprocess\preprocess_for_val_data" and follow the instruction in the **README.md**.

[added by Peter]

### Enviroment setup

- **Option 1**:
1. python 3.10.11
2. numpy 1.24.3
3. opencv-python-headless 4.9.0.80
4. pillow = 9.4.0
5. scipy 1.12.0
6. timm 0.5.4
7. torch 2.0.1
8. torchvision 0.15.2

- **Option 2 (Recommended)**:
Using Dockerfile in this folder to build the image.
Firstly, you need to cd to this folder.
Run the following command to build the image:
    `sudo docker build -t dfe_torch_image .`
The dfe_torch_image here is the image name, you can change it.
Once you build the image, you can run the following command to run the image:
    `sudo docker run -it --gpus=all --ipc=host -v (your folder path):(folder path in the container) --name=(your container name) dfe_torch_image bash`


## How to use **DFE_tracking_torch_Peter.py**?

**1.** Firstly, check you are in the correct enviroment.

**2.** Setup the configuration in the parser, such as the input and output path and center coordinates.

**3.** Select the model backbone you want to use, and load the correspond checkpoint path.
If you want to load the training weights, you can download them from "DigitalUPDRS/QPD_Shared/Peter_DFE_Training_Weights" in our NAS.

**4.** Select the tracking approahs you want to use, such as ordered or dynamic.
If you don't want to use both of them, just keep it empty.

**5.** Change the ground path and cotracker prediction path in the main function in the buttom of the code.
If you are not using Cotracker-DFE, just comment the part of Cotracker out.

**6.** Using `python DFE_tracking_torch_Peter.py` to run the code.

## How to use **DFE_train_torch.py**?
**1.** Firstly, check you are in the correct enviroment.

**2.** Setup the configuration in the parser, such as the input and output path, batch size, learning rate and epochs.

**3.** Select the model backbone you want to use, and load the correspond checkpoint path (for finetune).
If you want to load the pretrain weights, you can download them from "DigitalUPDRS/QPD_Shared/Peter_DFE_Training_Weights" in our NAS.

**4.** Using `python -m torch.distributed.run --nproc_per_node=n DFE_train_torch.py` to run the code, where the n is the number of GPUS you want to use.
If you don't want to use distributed training, using `python -m torch.distributed.run --nproc_per_node=1 DFE_train_torch.py` to run the code.

## How to use **DFE_eval_torch.py**?

**1.** Firstly, check you are in the correct enviroment.

**2.** Setup the configuration in the parser, such as the input and output path.
We used the test data to evaluate our model, where the data is create from "validation_data_create.py" in "Skin-Feature-Tracking\Preprocess\preprocess_for_val_data".

**3.** Select the model backbone you want to use, and load the correspond checkpoint path (for finetune).
If you want to load the pretrain weights, you can download them from "DigitalUPDRS/QPD_Shared/Peter_DFE_Training_Weights" in our NAS.

**4.** Using `python DFE_eval_torch.py` to run the code.


##Troubleshooting:

If when you use the distributed training, you find that all of the GPUS stock in 100% usage, you can check the following step:
1. Check you are in the correct environment.
2. Check your environment variables: "NCCL_P2P_DISABLE=1", by using `echo $NCCL_P2P_DISABLE`.
If you want to set the environment variables, using `export NCCL_P2P_DISABLE=1`.
3. Check the IOMMU in the BIOS is disable.

##Notes:

If you want to change the model structure, please check "Skin-Feature-Tracking\Peter_DFE\models".








