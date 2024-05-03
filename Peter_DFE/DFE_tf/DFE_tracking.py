import cv2

import numpy as np
import os
import re
import pandas as pd
import time
import sys
sys.path.append("..")
from utils.utils_joscha import *
import tqdm
import tensorflow





def search_global_minimum(pos, encodings, ref_encoding, img_cielab, encoder, center = None, text = ""):
    """
    Searches for the global minimum Sum of Squared Residuals (SSR) between the reference crop and all the crops in the current frame.

    Args:
        pos (list): List of positions of the center of each crop in the current frame.
        encodings (numpy.ndarray): Encodings of all the crops in the current frame.
        ref_encoding (numpy.ndarray): Encoding of the reference crop.
        img_cielab (numpy.ndarray): Current frame image in CIELab color space.
        encoder (tensorflow.keras.Model): Encoder model used for generating encodings.
        center (list, optional): Center position of the previous frame's prediction. 

    Returns:
        tuple: A tuple containing the following elements:
            - pred (list): Position of the crop with the global minimum SSR.
            - unique_error (float): SSR value of the predicted crop.
            - candidates (list): List of candidate crop positions.
            - sorted_ssr (numpy.ndarray): Sorted SSR values of all the crops.

    Steps:
        1. Calculate the SSR between the reference encoding and all the crop encodings in the current frame.
        2. If a center position is provided, apply SSR distance lock to limit the search area around the center.
        3. Iterate over unique SSR values in ascending order:
            - For each candidate crop with the current SSR value:
                - Perform quadratic interpolation using the crop position, CIELab image, and reference encoding.
                - Check if the interpolation coefficients satisfy the conditions for a global minimum.
                - If conditions are met, store the curvature and candidate crop position.
            - If any candidates are found, break the loop.
        4. Select the candidate crop with the maximum curvature as the prediction.
        5. Return the predicted position, SSR value, candidate positions, and sorted SSR values.
    """
    # Calculate the SSR between the reference encoding and all the crop encodings
    ssr=np.sum((encodings-ref_encoding)**2,axis=1)
    # Apply SSR distance lock if a center position is provided
    if isinstance(center, list):
        ssr = ssr_distance_lock(ssr, center, 10, pos, orgimg_shape)
    curvatures = []; candidates = []
    # Iterate over unique SSR values in ascending order (from smallest to largest)
    for unique_error in np.unique(np.sort(ssr)):
        for candidate in np.where(ssr==unique_error)[0]:
            # Perform quadratic interpolation
            _,c,_,_=quadratic_interpolation_deep_encodings(pos[candidate],img_cielab,3,encoder,ref_encoding)
            
            # Check if interpolation coefficients are valid
            if c is None or not hasattr(c, "__getitem__"):  
                continue
            # elif c.any() == None:
            #     continue
            # if np.isnan(c).any():
            #     continue
            D = 4*c[5]*c[4]-c[3]**2

            # Check if the interpolation coefficients satisfy the conditions for a global minimum
            if D>0 and c[4]>0:
                curvatures.append(D)
                candidates.append(candidate)

        # If any candidates are found, break the loop      
        if bool(curvatures):
            # print("curvatures: ", curvatures)
            # print("candidates: ", candidates)
            # print(f'Selected Curvature {str(np.max(curvatures))} and SSR value {unique_error}'+ text)
            break

    # Select the candidate crop with the maximum curvature as the prediction
    pred = pos[candidates[np.argmax(curvatures)]]
    # print(pred)
    # print("------------------")
    return pred, unique_error, candidates, np.sort(ssr)




def ssr_distance_lock(ssr, center, desired_range, pos): #center must be 2*1 array
    """
    Applies a distance lock to the Sum of Squared Residuals (SSR) values based on the center position and desired range.

    Args:
        ssr (numpy.ndarray): SSR values for each crop position.
        center (list): Center position of the previous frame's prediction (2,).
        desired_range (int): Desired range around the center position.
        pos (list): List of positions of the center of each crop in the current frame.

    Returns:
        numpy.ndarray: Updated SSR values with distance lock applied.

    Steps:
        1. Define the target area around the center position based on the desired range.
        2. Find the maximum SSR value.
        3. Create a new SSR array by adding the maximum SSR value to the original SSR values.
        4. Iterate over each crop position:
            - If the crop position is within the target area, keep the original SSR value.
            - Otherwise, assign the maximum SSR value to the crop position.
        5. Return the updated SSR values.
    """
    # Define the target area around the center position
    target_area = [
        center[0]-desired_range, center[0]+desired_range, center[1]-desired_range, center[1]+desired_range 
    ]
    # Find the maximum SSR value
    max_ssr = np.max(ssr)
    # Create a new SSR array by adding the maximum SSR value to the original SSR values
    new_ssr = ssr + max_ssr
    # Iterate over each crop position
    for i,p in enumerate(pos):
        new_ssr[i] = ssr[i] + max_ssr
        # Check if the crop position is within the target area
        if p[0] > target_area[0] and p[0] < target_area[1]:   # pos 1 = x
            if p[1] > target_area[2] and p[1] < target_area[3]:
                new_ssr[i] = ssr[i]    # Keep the original SSR value if within the target area
    return new_ssr




def calculate_ssr_with_previous_frame(pos, encodings, img_cielab, img_cielab_clone, encoder, center_previous_pred, window_size):
    """
    calculate_ssr_with_previous_frame is a function to calculate the SSR value between current frame and the previous frame.
    I used this for checking the change of SSR value.
    """
    center_previous_pred = [round(center_previous_pred[0]), round(center_previous_pred[1])]
    ref_crop=np.array(img_cielab_clone[int(center_previous_pred[1]-window_size[0]/2):int(center_previous_pred[1]+window_size[0]/2)
                        ,int(center_previous_pred[0]-window_size[1]/2):int(center_previous_pred[0]+window_size[1]/2),:])
    ref_encoding=np.squeeze(encoder.predict(np.expand_dims(ref_crop,axis=0)))
    pred, unique_error, _, _ = search_global_minimum(pos, encodings, ref_encoding, img_cielab, encoder, text = " extra crop", center = center_previous_pred)


    return unique_error




def DFE_tracking(img_path, center_set):
    """
    Main function for Original Deep Feature Encoding (DFE) tracking.

    Args:
        img_path (str): Path to the directory containing the input images.
        center_set (list): List of initial center positions for tracking.

    Returns:
        numpy.ndarray: Array of predicted positions (sub-pixel accuracy) for each frame.

    Steps:
        1. Initialize variables and load the first frame.
        2. Convert the first frame to CIELab color space and extract the reference crop.
        3. Load the pre-trained DFE model and create the encoder.
        4. Extract the reference encoding from the reference crop.
        5. Initialize the reference encodings list with the reference encoding.
        6. Iterate over the frames starting from the second frame:
            - Load the current frame and convert it to CIELab color space.
            - Generate crops and their positions from the current frame.
            - Convert the crops to encodings using the encoder.
            - Compare the encodings with the reference encodings to find the best match.
            - Apply sub-pixel refinement using quadratic interpolation.
            - Update the reference encodings list if necessary.
            - Append the predicted position to the list of predictions.
        7. Return the array of predicted positions.
    """
    # Initialize variables and load the first frame
    center = center_set[0]
    start_frame = 0
    error_number = 0
    path = img_path
    filenames=sorted_alphanumeric(os.listdir(path))

    print(f"Got {len(filenames)} files")

    window_size=(31,31)
    img=cv2.imread(path+filenames[start_frame])

    global orgimg_shape
    orgimg_shape = img.shape

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=convertRGB2CIELab(img)
    
    # Extract the reference crop from the first frame
    x1 = int(center[0]-np.floor(window_size[0]/2))
    x2 = int(center[0]+np.ceil(window_size[0]/2))
    y1 = int(center[1]-np.floor(window_size[1]/2))
    y2 = int(center[1]+np.ceil(window_size[1]/2))
    ref_crop=np.array(img[y1-1:y2-1,x1-1:x2-1,:])
  
    print("ref_crop = ", ref_crop.shape)
    print(f"Center at {center}, crop corr is {x1, x2, y1, y2}")
    
    model=load_model("../checkpoints/DFE_hand.h5", custom_objects = {"tensorflow": tensorflow})
    encoder=Model(model.input, model.layers[int(len(model.layers)/2)-1].output) #get only half of the network
    encoder.compile(optimizer='adamax',loss='mse')
    ref_encoding=np.squeeze(encoder.predict(np.expand_dims(ref_crop,axis=0)))

    # print("ref_encoding = ", ref_encoding)
    print(ref_encoding.shape)
 

    # Initialize the reference encodings list with the reference encoding
    ref_encodings = [ref_encoding]
    
    stride=1
    dfe_preds=[center]
    dfe_preds_subpixel = [center]
    all_ssr = []
    all_ssr_previous = []
    token = 0
    center_previous_pred = center
    print("start creating ssr_data")
    ssr_data = []
    print("finish creating ssr_data")
    error_frames = []
    all_coords = []
    
    # Iterate over the frames starting from the second frame
    for f in range(1,len(filenames)):
        if f != 1:
            img_cielab_clone = img_cielab.copy() # save previous frame for extra crop
        img=cv2.imread(path+filenames[f])
        img_cielab=convertRGB2CIELab(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # width_step = int((img_cielab.size[0]-window_size[0])/stride)
        pos=[]
        crops=[]
        
        # Generate crops and their positions from the current frame
        for i in range(int((img_cielab.shape[0]-window_size[1])/stride)):
            for j in range(int((img_cielab.shape[1]-window_size[0])/stride)):
                x=int(j+np.ceil(window_size[0]/2))    # np.ceil(window_size[0]/2) = 16
                y=int(i+np.ceil(window_size[1]/2))
                crops.append(np.array(img_cielab[i:i+window_size[1],j:j+window_size[0],:]))
                pos.append([x,y]) #position of centers

     

        crops=np.array(crops)
        # Convert the crops to encodings using the encoder
        encodings=np.squeeze(encoder.predict(crops,batch_size=2**9,verbose=1)) 
        
        compare_ssr = [] 
        compare_curvature = [] 
        compare_pred = []
        

        # Compare the encodings with the reference encodings to find the best match
        for ref_num in range(len(ref_encodings)):
            pred, unique_error, candidates, sorted_ssr = search_global_minimum(pos, encodings, ref_encodings[ref_num], img_cielab, encoder, center = dfe_preds[-1])  
            #pred = pos[candidates[np.argmax(curvatures)]]
            # coords.append([pos[candidate] for candidate in candidates])
            compare_ssr.append(unique_error)
            compare_pred.append(pred)
        # all_coords.append(coords)
        # all_ssr.append(sorted_ssr)
        compare_ssr = np.array(compare_ssr)
        if token == 0:
            thresholdSSR = unique_error
            # dfe_preds.append(pred)
            print(f"initial threshold {unique_error}")
            token += 1

        # Update the reference encodings list if necessary
        if abs(thresholdSSR - np.min(compare_ssr)) > 20:
            print(f"change threshold {np.min(compare_ssr)} to {thresholdSSR}")
            error_number += 1
            error_frames.append(f)
            token += 1
            center_previous_pred = dfe_preds[-1] # for previous frame
            center_previous_pred = [round(center_previous_pred[0]), round(center_previous_pred[1])]
            ref_crop=np.array(img_cielab_clone[int(center_previous_pred[1]-window_size[0]/2):int(center_previous_pred[1]+window_size[0]/2)
                               ,int(center_previous_pred[0]-window_size[1]/2):int(center_previous_pred[0]+window_size[1]/2),:])
            ref_encoding=np.squeeze(encoder.predict(np.expand_dims(ref_crop,axis=0)))
            ref_encodings.append(ref_encoding)
            pred, unique_error = search_global_minimum(pos, encodings, ref_encoding, img_cielab, encoder, text = " extra crop", center = dfe_preds[-1])
            compare_ssr = np.append(compare_ssr, unique_error)
            thresholdSSR = np.min(compare_ssr)
            compare_pred.append(pred)
        
        pred = compare_pred[np.argmin(compare_ssr)]
        
        dis_from_previous = np.sqrt(np.sum((np.array(dfe_preds[-1])-np.array(pred))**2))
        num = 0
        index = np.argsort(compare_ssr)
        if dis_from_previous > 20:
            error_number += 1
            error_frames.append(f)
        if f != 1:
            while dis_from_previous > 20:
                print("dis_from_previous > 20")
                
                if num > len(compare_ssr)-1:
                    token += 1
                    center_previous_pred = dfe_preds[-1] # for previous frame
                    center_previous_pred = [round(center_previous_pred[0]), round(center_previous_pred[1])]
                    ref_crop=np.array(img_cielab_clone[int(center_previous_pred[1]-window_size[0]/2):int(center_previous_pred[1]+window_size[0]/2)
                                    ,int(center_previous_pred[0]-window_size[1]/2):int(center_previous_pred[0]+window_size[1]/2),:])
                    ref_encoding=np.squeeze(encoder.predict(np.expand_dims(ref_crop,axis=0)))
                    ref_encodings.append(ref_encoding)
                    pred, unique_error, candidates, ssr_sort = search_global_minimum(pos, encodings, ref_encoding, img_cielab, encoder, center = dfe_preds[-1], text = " extra crop")
                    compare_ssr = np.append(compare_ssr, unique_error)
                    compare_pred.append(pred)
                    pred = compare_pred[np.argmin(compare_ssr)]
                    print(f"Fail to find suitable point, select {pred}")
                    break
                pred = compare_pred[index[num]]
                dis_from_previous = np.sqrt(np.sum((np.array(dfe_preds[-1])-np.array(pred))**2))
                num += 1
        dis_from_previous = np.sqrt(np.sum((np.array(dfe_preds[-1])-np.array(pred))**2))
        ref_encodings = ref_encodings[1::] if len(ref_encodings) > 12 else ref_encodings
        compare_ssr = compare_ssr[1::] if len(compare_ssr) > 12 else compare_ssr

        final_encoding = ref_encodings[np.argmin(compare_ssr)]
        
        ssr = np.sum((encodings-final_encoding)**2,axis=1)
        ssr_data.append(ssr)

        # Apply sub-pixel refinement using quadratic interpolation
        pred_subpixel,c,A,B=quadratic_interpolation_deep_encodings(pred,img_cielab,3,encoder,final_encoding)
        pred_subpixel = pred_subpixel[0]
            
        # if f > 30 and f < 35:
        #     print("pred = ", pred)
            # print("pred_subpixel = ", pred_subpixel)
        # if f != 1:
        #     ssr_with_previous_frame = calculate_ssr_with_previous_frame(pos, encodings, img_cielab, img_cielab_clone, encoder, center_previous_pred = dfe_preds[-1], window_size = window_size)
        #     all_ssr_previous.append(ssr_with_previous_frame)
        #     all_ssr_previous_save = np.array(all_ssr_previous)
        #     np.save(save_path + f"{name[0]}_previous_ssr.npy", all_ssr_previous_save)
        # all_ssr.append(unique_error)
        # all_ssr_save = np.array(all_ssr)
        # np.save(save_path + f"{name[0]}_ssr.npy", all_ssr_save)

        # print(f"Image path: {path}")
        # print(f"Select point at: {pred} with dis {dis_from_previous} for {filenames[f]}" +'\t')


        # Append the predicted position to the list of predictions
        dfe_preds.append(pred)
        dfe_preds_subpixel.append(pred_subpixel)
        dfe_preds_save = np.array(dfe_preds_subpixel)
        
        # np.save(save_path + f"{name[0]}_preds.npy", dfe_preds_save)
        
    # print("error_number = ", error_number)
        # 
        # if f > 5:
        #     ref_encodings.pop(3)
        # center_this_pred = dfe_preds[-1] # get the pred of current frame
        # # print(center_this_pred)
        # temp_center = [round(center_this_pred[0]), round(center_this_pred[1])]
        # ref_crop=np.array(img_cielab[int(temp_center[1]-window_size[0]/2):int(temp_center[1]+window_size[0]/2)
        #                     ,int(temp_center[0]-window_size[1]/2):int(temp_center[0]+window_size[1]/2),:])
        # ref_encoding=np.squeeze(encoder.predict(np.expand_dims(ref_crop,axis=0)))
        # ref_encodings.append(ref_encoding)


        # Update the reference encodings list with the predicted position
        if f in [2,4,6,8]:
            center_this_pred = dfe_preds[-1] # get the pred of current frame
            # print(center_this_pred)
            temp_center = [round(center_this_pred[0]), round(center_this_pred[1])]
            ref_crop=np.array(img_cielab[int(temp_center[1]-window_size[0]/2):int(temp_center[1]+window_size[0]/2)
                                ,int(temp_center[0]-window_size[1]/2):int(temp_center[0]+window_size[1]/2),:])
            ref_encoding=np.squeeze(encoder.predict(np.expand_dims(ref_crop,axis=0)))
            ref_encodings.append(ref_encoding)
    return dfe_preds_save


