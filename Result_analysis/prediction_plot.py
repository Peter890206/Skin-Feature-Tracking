import numpy as np
import matplotlib.pyplot as plt
import re
import os
import cv2

def sorted_alphanumeric(data):
    # Sorts filenames of in alphanumeric order
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def convertRGB2CIELab(image):
    # Converts uint8 RGB image to float32 CIELAB format
    image = np.array(image).astype('float32')
    if np.max(image) > 1:
        image *= 1/255
    Lab_image = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
    Lab_image[:,:,0]=Lab_image[:,:,0]/100 # Normalize L to be betweeen 0 and 1
    Lab_image[:,:,1]=(Lab_image[:,:,1]+127)/(2*127)
    Lab_image[:,:,2]=(Lab_image[:,:,2]+127)/(2*127)
    return Lab_image

Jose_ground_truth_path = "../dfe_ground_truth/bbox_centers_pd.npy"
Mean_ground_truth_path = "../Peter_handmole_ground_truth_mean.npy"
dfe_prediction_path = "../handmole_DFE_tf_40_subpixel_220_238_preds.npy"
cotracker_prediction_path = "../Handmole_cotracker_pred_220_238.npy"
frames_path = "../dfe_ground_truth/pd_frames_reduced"

filenames=sorted_alphanumeric(os.listdir(frames_path))

Jose_ground_truth = np.load(Jose_ground_truth_path)
Mean_ground_truth = np.load(Mean_ground_truth_path)
dfe_prediction = np.load(dfe_prediction_path)
cotracker_prediction = np.load(cotracker_prediction_path)

scale = 30
crop_size = 31
window_size = crop_size * scale

for i, image_filename in enumerate(filenames):
    image = cv2.imread(os.path.join(frames_path, image_filename))

    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # image=convertRGB2CIELab(image)

    center_x, center_y = Mean_ground_truth[i]
    center_Jose_x, center_Jose_y = Jose_ground_truth[i]
    prediction_x, prediction_y = dfe_prediction[i]
    cotracker_x, cotracker_y = cotracker_prediction[i]
    distance_Jose_x = center_Jose_x - int(center_x)
    distance_Jose_y = center_Jose_y - int(center_y)
    distance_prediction_x = prediction_x - int(center_x)
    distance_prediction_y = prediction_y - int(center_y)
    distance_cotracker_x = cotracker_x - int(center_x)
    distance_cotracker_y = cotracker_y - int(center_y)

    crop_image = image[int(center_y) - crop_size//2:int(center_y) + crop_size//2 + 1, int(center_x) - crop_size//2:int(center_x) + crop_size//2 + 1]
    crop_image = cv2.resize(crop_image, (window_size, window_size), interpolation=cv2.INTER_NEAREST)
    # Mark the center coordinates with a red dot
    new_center_mean = (crop_size//2 * scale + int((center_x - int(center_x)) * scale) + scale//2, crop_size//2 * scale + int((center_y - int(center_y)) * scale) + scale//2)
    print("new_center_mean", new_center_mean)
    new_center_Jose = (crop_size//2 * scale + scale//2 + int(distance_Jose_x * scale), crop_size//2 * scale + scale//2 + int(distance_Jose_y * scale))
    print("new_center_Jose", new_center_Jose)
    new_prediction_center = (crop_size//2 * scale + int(distance_prediction_x * scale) + scale//2, crop_size//2 * scale + int(distance_prediction_y * scale) + scale//2)
    new_cotracker_center = (crop_size//2 * scale + int(distance_cotracker_x * scale) + scale//2, crop_size//2 * scale + int(distance_cotracker_y * scale) + scale//2)
    cv2.circle(crop_image, new_center_mean, 5, (0, 0, 255), -1)
    cv2.circle(crop_image, new_center_Jose, 3, (0, 255, 255), -1)
    cv2.circle(crop_image, new_prediction_center, 5, (255, 0, 0), -1)
    cv2.circle(crop_image, new_cotracker_center, 4, (0, 255, 0), -1)
    cv2.imshow("Cropped Image", crop_image)
    cv2.waitKey(0)
    cv2.imwrite(f'./prediction_ground_truth_comparison/threshold_2_comparison_frame{i}.jpg', crop_image)
    cv2.destroyAllWindows()



