# Preprocess

## Summary

- This folder contains three preprocess main folders: "preprocess_for_cotracker", "preprocess_for_DFE" and "preprocess_for_val_data".

- The folder "preprocess_for_cotracker" contains all the code for generating the data for Co-Tracker, you can check details in the README.md file in this folder.

- The folder "preprocess_for_DFE" contains all the code for generating the data for DFE, you can check details in the README.md file in this folder.

- The folder "preprocess_for_val_data" contains the codes for generating the validation (test) data for DFE from our experimental videos, you can check details in the README.md file in this folder.


### Enviroment setup

- **crop_video.py** in "preprocess_for_cotracker" need the following packages:
1. python 3.8.18
2. moviepy 1.0.3
3. pandas 1.4.4

- The other codes need the following packages:
1. python 3.9.13
2. opencv-python 4.8.1.78
3. regex 2022.7.9
4. pandas 1.4.4
5. numpy 1.23.5
6. tqdm 4.64.1