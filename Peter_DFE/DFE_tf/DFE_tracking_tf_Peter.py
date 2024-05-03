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

def quadratic_interpolation_deep_encodings_Peter(center, window_pts, ssr):
    """
    Performs quadratic interpolation for deep feature encodings.

    Args:
        center (list or numpy.ndarray): Center position of the current window.
        window_pts (numpy.ndarray): Coordinates of the points in the window relative to the center. The shape is (9, 2).
        ssr (numpy.ndarray): Sum of Squared Residuals (SSR) values for each window in window_pts.

    Returns:
        tuple: A tuple containing the following elements:
            - pred (numpy.ndarray): Predicted sub-pixel position.
            - C (numpy.ndarray): Coefficients of the quadratic surface.
            - A (numpy.ndarray): Matrix A used in the quadratic surface fitting.
            - B (numpy.ndarray): Vector B used in the quadratic surface fitting.

    Steps:
        1. Prepare the data for quadratic surface fitting:
            - Subtract the center coordinates from the window points.
            - Construct matrix A using the window points and their products.
        2. Find the minimum of the quadratic surface using the `find_min_surface` function.
        3. If the coefficients (C) are not a valid numpy array, return the predicted position (pred) and NaN values for C, A, and B.
        4. Clip the predicted position (pred) to be within the range [-1, 1] and add the center coordinates.
        5. Return the predicted sub-pixel position (pred), coefficients (C), matrix A, and vector B.
    """

    # Prepare data for quadratic surface fitting
    window_pts[:,0]-=center[0]
    window_pts[:,1]-=center[1]
    A=np.c_[(np.ones(ssr.shape),window_pts[:,0],window_pts[:,1],np.prod(window_pts,axis=1),
                window_pts[:,0]**2,window_pts[:,1]**2)]  
    
    # Find the minimum of the quadratic surface                    
    pred, C, A, B=find_min_surface(A,ssr)

    # Check if the coefficients (C) are not a valid numpy array
    if not type(C) is np.ndarray: 
        return pred, np.NaN, A, B
    
    # Clip the predicted position (pred) to be within the range [-1, 1] and add the center coordinates
    pred=np.clip(pred,-1,1)+center

    return pred, C, A, B

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

def calculate_sorted_candidates(ref_encoding, encodings, center, pos, offsets, width_step, height_step, total_rank_range):
    """
    Calculates sorted candidates for each reference encoding.

    Args:
        ref_encoding (numpy.ndarray): Reference encoding of shape (1, encoding_dim).
        encodings (numpy.ndarray): Encodings of all the crops in the current frame of shape (num_crops, encoding_dim).
        center (list or numpy.ndarray): Center position of the previous frame's prediction of shape (2,).
        pos (numpy.ndarray): Positions of the center of each crop in the current frame of shape (num_crops, 2).
        offsets (numpy.ndarray): Offsets used for generating candidate groups of shape (9,).
        width_step (int): Number of horizontal steps of the image.
        height_step (int): Number of vertical steps of the image.
        total_rank_range (int): Total number of top candidates to consider.

    Returns:
        tuple: A tuple containing the following elements:
            - candidates (list): List of candidate positions sorted by curvature.
            - unique_errors (list): List of unique SSR values corresponding to the candidates.
            - ssr (numpy.ndarray): SSR values for all the crop positions of shape (num_crops,).

    Steps:
        1. Expand the reference encoding to shape (1, encoding_dim).
        2. Calculate the Sum of Squared Residuals (SSR) between the reference encoding and all the crop encodings.
        3. Apply SSR distance lock if a center position is provided.
        4. Iterate over unique SSR values in ascending order (from smallest to largest):
            - For each candidate crop with the current SSR value:
                - Generate candidate groups using the offsets and candidate position for array calculation.
                - Ensure the candidate groups are within the valid range.
                - Extract the SSR values and window points for the candidate groups.
                - Perform quadratic interpolation using the candidate position, window points, and SSR values.
                - Check if the interpolation coefficients satisfy the conditions for a global minimum.
                - If conditions are met, store the curvature and candidate position.
            - If any candidates are found, sort them by curvature in descending order (from largest to smallest).
            - If the number of candidates reaches the total rank range, break the loop.
        5. Return the sorted candidates, unique SSR values, and SSR values for all crop positions.
    """
    
    # Expand the reference encoding to shape (1, encoding_dim)
    ref_encoding = np.expand_dims(ref_encoding, axis=0)

    # Calculate the Sum of Squared Residuals (SSR) between the reference encoding and all the crop encodings
    ssr=np.sum((encodings-ref_encoding)**2,axis=1)
    # print("SSR shape: ", ssr.shape)

    # Apply SSR distance lock if a center position is provided
    if isinstance(center, list):
        # print("start lock")
        ssr = ssr_distance_lock(ssr, center, 10, pos)
    
    candidates = []; preds = []; unique_errors = []
    # Iterate over unique SSR values in ascending order
    for unique_error in np.unique(np.sort(ssr)):
        curvatures = []
        for candidate in np.where(ssr==unique_error)[0]:
            offset_meshgrid, candidate_meshgrid = np.meshgrid(offsets, candidate)
            cadidate_group = candidate_meshgrid + offset_meshgrid
            # print("cadidate_group.shape", cadidate_group.shape)

            # Ensure the candidate groups are within the valid range
            error_index = np.logical_and(cadidate_group >= 0, cadidate_group < width_step*height_step)
            cadidate_group = cadidate_group * error_index    # make sure the index is in range, else set it to 0
            cadidate_group = np.squeeze(cadidate_group)
            ssr_for_quadratic = ssr[cadidate_group]
            # print("cadidate_group.shape", cadidate_group.shape)
            # print("pos shape: ", pos.shape)
            window_pts = pos[cadidate_group]

            # Perform quadratic interpolation
            _,c,_,_=quadratic_interpolation_deep_encodings_Peter(pos[candidate], window_pts, ssr_for_quadratic)
            
            # Check if the interpolation coefficients satisfy the conditions for a global minimum
            if c is None or not hasattr(c, "__getitem__"):  
                continue
            
            D = 4*c[5]*c[4]-c[3]**2
            if D>0 and c[4]>0:
                curvatures.append((D, candidate))
                unique_errors.append(unique_error)
            # else:
                # print("None")

        # If any candidates are found, sort them by curvature in descending order
        if curvatures:
            curvatures.sort(reverse=True)
            candidates.extend([candidate for _, candidate in curvatures])    #append the candidate from the highest curvature to the lowest
        
        # If the number of candidates reaches the total rank range, break the loop
        if len(candidates) >= total_rank_range:
            candidates = candidates[0:total_rank_range]
            unique_errors = unique_errors[0:total_rank_range]
            break
    return candidates, unique_errors, ssr

def search_global_minimum_average(pos, encodings, ref_encodings, width_step, height_step, center, total_rank_range = 5, unique_rank_range = 3):
    """
    Searches for the global minimum by averaging the results from multiple reference encodings.

    Args:
        pos (list): Positions of the center of each crop in the current frame of shape (num_crops, 2).
        encodings (numpy.ndarray): Encodings of all the crops in the current frame of shape (num_crops, encoding_dim).
        ref_encodings (list): List of reference encodings, each of shape (encoding_dim,).
        width_step (int): Number of horizontal steps of the image.
        height_step (int): Number of vertical steps of the image.
        center (list or numpy.ndarray): Center position of the previous frame's prediction of shape (2,).
        total_rank_range (int, optional): Total number of top candidates to consider. Defaults to 5.
        unique_rank_range (int, optional): Number of unique candidates to consider. Defaults to 3.

    Returns:
        tuple: A tuple containing the following elements:
            - pred (numpy.ndarray): Predicted position of the global minimum of shape (2,).
            - pred_ssr_min (float): Minimum SSR value corresponding to the predicted position.
            - new_ref_encoding (numpy.ndarray): New reference encoding based on the predicted position of shape (encoding_dim,).
            - window_pts_min (numpy.ndarray): Window points around the predicted position of shape (9, 2).
            - new_ssr_for_quadratic (numpy.ndarray): SSR values for quadratic interpolation around the predicted position of shape (9,).
            - all_candidates (numpy.ndarray): All candidate positions considered during the search of shape (num_ref_encodings, total_rank_range).

    Steps:
        1. Convert the input data to numpy arrays if necessary.
        2. Define offsets for generating candidate groups.
        3. For each reference encoding:
            - Calculate sorted candidates, unique SSR values, and SSR values using the `calculate_sorted_candidates` function.
            - Store the results in separate lists.
        4. Find unique candidates from the top candidates of each reference encoding.
        5. Calculate the ranks for each unique candidate based on their positions in the candidate lists.
        6. Find the candidate(s) with the minimum rank.
        7. For each candidate with the same minimum rank:
            - Find the reference encoding index that gives the minimum SSR value for the candidate.
            - Update the predicted position, minimum SSR value, and new reference encoding if necessary.
        8. Generate candidate groups around the predicted position.
        9. Extract the window points and SSR values for the candidate groups.
        10. Return the predicted position, minimum SSR value, new reference encoding, window points, SSR values for quadratic interpolation, and all candidates.
    """
    
    pos = np.array(pos)

    # Convert reference encodings and encodings to numpy arrays if necessary
    ref_encodings = [ref_encoding.cpu().numpy() if type(ref_encoding) is not np.ndarray else ref_encoding for ref_encoding in ref_encodings]
    if type(encodings) is not np.ndarray:
        encodings = encodings.cpu().numpy()

    all_candidates = []
    all_unique_ssr = []
    all_ssr = []

    # Define offsets for generating candidate groups
    offsets = np.array([-width_step - 1, -width_step, -width_step + 1, -1, 0, 1, width_step - 1, width_step, width_step + 1])
    for ref_encoding in ref_encodings:
        # print("ref_encoding sum: ", np.sum(ref_encoding))
        candidates, unique_errors, ssr = calculate_sorted_candidates(ref_encoding, encodings, center, pos, offsets, width_step, height_step, total_rank_range)
        all_candidates.append(candidates)
        all_unique_ssr.append(unique_errors)
        all_ssr.append(ssr)

    all_candidates = np.array(all_candidates)
    # print("all_candidates shape: ", all_candidates.shape)

    # Find unique candidates from the top candidates of each reference encoding
    unique_candidates = np.unique(all_candidates[:,0:unique_rank_range])

    """
    calculate the ranks for each reference vectors
    """
    ranks = []
    rank = 0

    # Calculate the ranks for each unique candidate based on their positions in the candidate lists
    for unique_candidate in unique_candidates:
        for i, candidates in enumerate(all_candidates):
            if not unique_candidate in candidates:
                rank += total_rank_range + 2
            else:
                rank += np.where(candidates==unique_candidate)[0][0]
        ranks.append(rank)
        rank = 0

    # Find the candidate(s) with the minimum rank
    rank_min_indexs = np.where(ranks==np.min(ranks))[0]
    pred_ssr_min = 0
    pred_candidate_min = 0
    pred_ssr = 0
    pred_ssr_sum = 0
    pred_ssr_sum_min = 0
    reference_encoding_index = -1
    new_ref_encoding_index = 0

    # Iterate over each candidate with the minimum rank
    for rank_min_index in rank_min_indexs:
        # print("rank_min_index = ", rank_min_index)
        pred_candidate = unique_candidates[rank_min_index]
        pred_ssr_index_min = -1
        pred_ssr_min = 0

        # Find the reference encoding index that gives the minimum SSR value for the candidate
        for i, candidates in enumerate(all_candidates):
            if not pred_candidate in candidates:
                continue
            else:
                # print("pred_candidate = ", pred_candidate)
                pred_ssr_index = np.where(candidates==pred_candidate)[0][0]
                pred_ssr_sum += all_unique_ssr[i][pred_ssr_index]
                if pred_ssr_index_min == -1:
                    pred_ssr_index_min = pred_ssr_index
                    pred_ssr = all_unique_ssr[i][pred_ssr_index]
                    pred_ssr_min = pred_ssr
                    reference_encoding_index = i
                    reference_encoding = encodings[pred_candidate]
                else:
                    if pred_ssr_index <= pred_ssr_index_min:
                        pred_ssr = all_unique_ssr[i][pred_ssr_index]
                        if pred_ssr < pred_ssr_min:
                            # print("i =", i, "pred_ssr_index = ", pred_ssr_index, "pred_candidate =", pred_candidate)
                            reference_encoding_index = i
                            reference_encoding = encodings[pred_candidate]
                            pred_ssr_min = pred_ssr

        # Update the predicted position, minimum SSR value, and new reference encoding if necessary
        if pred_ssr_sum_min == 0:
            # print("pred_ssr_sum = ", pred_ssr_sum)
            pred_ssr_sum_min = pred_ssr_sum
            pred_candidate_min = pred_candidate
            new_ref_encoding_index = reference_encoding_index
            new_ref_encoding = reference_encoding
        else:
            if pred_ssr_sum < pred_ssr_sum_min:
                pred_ssr_sum_min = pred_ssr_sum
                pred_candidate_min = pred_candidate
                new_ref_encoding_index = reference_encoding_index
                new_ref_encoding = reference_encoding
        pred_ssr_sum = 0

    # Generate candidate groups around the predicted position
    offset_min_meshgrid, candidate_min_meshgrid = np.meshgrid(offsets, pred_candidate_min)
    cadidate_min_group = offset_min_meshgrid + candidate_min_meshgrid

    # Ensure the candidate groups are within the valid range
    error_index = np.logical_and(cadidate_min_group >= 0, cadidate_min_group < width_step*height_step)
    cadidate_min_group = cadidate_min_group * error_index 
    cadidate_min_group = np.squeeze(cadidate_min_group)

    # Extract the predicted position, window points, and SSR values for the candidate groups
    pred = pos[pred_candidate_min]
    window_pts_min = pos[cadidate_min_group]
    new_ssr_for_quadratic = all_ssr[new_ref_encoding_index][cadidate_min_group]
    

    return pred, pred_ssr_min, new_ref_encoding, window_pts_min, new_ssr_for_quadratic, all_candidates


def calculate_ssr_with_previous_frame(pos, encodings, img_cielab, img_cielab_clone, encoder, center_previous_pred, window_size):
    """
    calculate_ssr_with_previous_frame is a function to calculate the SSR value between current frame and the previous frame.
    I used this for checking the change of SSR value.
    """
   
    center_previous_pred = [round(center_previous_pred[0]), round(center_previous_pred[1])]
    ref_crop=np.array(img_cielab_clone[int(center_previous_pred[1]-window_size[0]/2):int(center_previous_pred[1]+window_size[0]/2)
                        ,int(center_previous_pred[0]-window_size[1]/2):int(center_previous_pred[0]+window_size[1]/2),:])
    ref_encoding=np.squeeze(encoder.predict(np.expand_dims(ref_crop,axis=0)))
    pred, unique_error = search_global_minimum(pos, encodings, ref_encoding, img_cielab, encoder, text = " extra crop", center = center_previous_pred)


    return unique_error

def DFE_tracking(img_path, center_set):
    """
    Performs Optimized Deep Feature Encoding (DFE) tracking on a sequence of images.

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
            - Search for the global minimum position using the `search_global_minimum_average` function.
            - Apply sub-pixel refinement using quadratic interpolation.
            - Append the predicted position to the list of predictions.
            - Update the reference encodings list by removing the middle encoding and appending the new reference encoding.
        7. Return the array of predicted positions (sub-pixel accuracy).
    """

    # Initialize variables and load the first frame
    center = center_set[0]
    start_frame = 0
    path = img_path
    filenames=sorted_alphanumeric(os.listdir(path))
    print(f"Got {len(filenames)} files")

    window_size=(31,31)
    img=cv2.imread(path+filenames[start_frame])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # Convert the first frame to CIELab color space
    img=convertRGB2CIELab(img)
    
    # Extract the reference crop from the first frame
    x1 = int(center[0]-window_size[0]/2)
    x2 = int(center[0]+window_size[0]/2)
    y1 = int(center[1]-window_size[1]/2)
    y2 = int(center[1]+window_size[1]/2)
    ref_crop=np.array(img[y1:y2,x1:x2,:])
    
    print(f"Center at {center}, crop corr is {x1, x2, y1, y2}")
    
    # Load the pre-trained DFE model and create the encoder
    model=load_model("../checkpoints/DFE_hand.h5", custom_objects = {"tensorflow": tensorflow})
    encoder=Model(model.input, model.layers[int(len(model.layers)/2)-1].output) #get only half of the network
    encoder.compile(optimizer='adamax',loss='mse')

    # Extract the reference encoding from the reference crop
    ref_encoding=np.squeeze(encoder.predict(np.array([ref_crop]), batch_size=2**9, verbose=1))
    print(ref_encoding.shape)

    # Initialize the reference encodings list with the reference encoding
    ref_encodings = [ref_encoding]
    
    
    stride=1
    dfe_preds=[center]
    dfe_preds_subpixel=[center]
    ref_len = 14

    # Iterate over the frames starting from the second frame
    for f in range(1,len(filenames)):
        if f != 1:
            img_cielab_clone = img_cielab.copy() # save previous frame for extra crop

        # Load the current frame and convert it to CIELab color space
        img=cv2.imread(path+filenames[f])
        img_cielab=convertRGB2CIELab(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        width_step = int((img_cielab.shape[0]-window_size[0])/stride)
        height_step = int((img_cielab.shape[1]-window_size[1])/stride)
        pos=[]
        crops=[]
        # print("img_cielab.shape", img_cielab.shape)

        # Generate crops and their positions from the current frame
        for i in range(int((img_cielab.shape[0]-window_size[1])/stride)):
            for j in range(int((img_cielab.shape[1]-window_size[0])/stride)):
                x=int(j+np.ceil(window_size[0]/2))
                y=int(i+np.ceil(window_size[1]/2))
                crops.append(np.array(img_cielab[i:i+window_size[1],j:j+window_size[0],:]))
                pos.append([x,y]) #position of centers

        crops=np.array(crops)
        # print("crops.shape", crops.shape)

        # Convert the crops to encodings using the encoder
        encodings=np.squeeze(encoder.predict(crops,batch_size=2**9,verbose=1)) # Convert all crops to encodings
        
        print(f"Got {len(ref_encodings)} ref_crops")

        # Search for the global minimum position using the `search_global_minimum_average` function
        pred, pred_ssr_min, new_ref_encoding, window_pts_min, new_ssr_for_quadratic, all_candidates = search_global_minimum_average(pos, encodings, ref_encodings, width_step, height_step, center = dfe_preds[-1])
        
        # Apply sub-pixel refinement using quadratic interpolation
        pred_subpixel,c,A,B=quadratic_interpolation_deep_encodings_Peter(pred, window_pts_min, new_ssr_for_quadratic)
        pred_subpixel = pred_subpixel[0]

        # Append the predicted position to the list of predictions
        dfe_preds.append(pred)
        dfe_preds_subpixel.append(pred_subpixel)
        
        # np.save(save_path + f"{name[0]}_preds.npy", dfe_preds_save)

        # Update the reference encodings list by removing the middle encoding and appending the new reference encoding (head-tail approach)
        if f > ref_len-1:
            ref_encodings.pop(int(ref_len/2))
            # ref_encodings.pop((ref_len-2))
        ref_encodings.append(new_ref_encoding)

        # Update the reference encodings list by appending the new reference encoding until the reference length is reached (ordered approach)
        # if f < ref_len:
        #     ref_encodings.append(new_ref_encoding)

    # Return the array of predicted positions (sub-pixel accuracy)
    dfe_preds_save = np.array(dfe_preds_subpixel)
    return dfe_preds_save

 





def search_global_minimum_average_old_version(pos, encodings, ref_encodings, img_cielab, encoder, center = None, text = "", total_rank_range = 5, unique_rank_range = 3):
    """
    Searches for the global minimum by averaging the results from multiple reference encodings (old version).

    Args:
        pos (list): Positions of the center of each crop in the current frame.
        encodings (numpy.ndarray): Encodings of all the crops in the current frame.
        ref_encodings (list): List of reference encodings.
        img_cielab (numpy.ndarray): Current frame image in CIELab color space.
        encoder (tensorflow.keras.Model): Encoder model used for generating encodings.
        center (list, optional): Center position of the previous frame's prediction. Defaults to None.
        total_rank_range (int, optional): Total number of top candidates to consider. Defaults to 5.
        unique_rank_range (int, optional): Number of unique candidates to consider. Defaults to 3.

    Returns:
        tuple: A tuple containing the following elements:
            - pred (numpy.ndarray): Predicted position of the global minimum.
            - new_ref_encoding_index (int): Index of the new reference encoding.

    Steps:
        1. Initialize lists to store candidates, unique SSR values, and SSR values for each reference encoding.
        2. For each reference encoding:
            - Calculate the SSR values between the reference encoding and all the crop encodings.
            - Apply SSR distance lock if a center position is provided.
            - Iterate over unique SSR values in ascending order:
                - For each candidate crop with the current SSR value:
                    - Perform quadratic interpolation using the crop position, CIELab image, and reference encoding.
                    - Check if the interpolation coefficients satisfy the conditions for a global minimum.
                    - If conditions are met, store the curvature and candidate crop position.
            - If any candidates are found, sort them by curvature in descending order.
            - If the number of candidates reaches the total rank range, break the loop.
            - Append the candidates, unique SSR values, and SSR values to their respective lists.
        3. Convert the candidate lists to a numpy array.
        4. Find unique candidates from the top candidates of each reference encoding.
        5. Calculate the ranks for each unique candidate based on their positions in the candidate lists.
        6. Find the candidate(s) with the minimum rank.
        7. For each candidate with the minimum rank:
            - Find the reference encoding index that gives the minimum SSR value for the candidate.
            - Update the predicted position, minimum SSR value, and new reference encoding index if necessary.
        8. Return the predicted position and the new reference encoding index.
    """
    
    all_candidates = []
    all_unique_ssr = []
    all_ssr = []
    # For each reference encoding
    for ref_encoding in ref_encodings:
        # Calculate the SSR values between the reference encoding and all the crop encodings
        ssr=np.sum((encodings-ref_encoding)**2,axis=1)
        if isinstance(center, list):
            # Apply SSR distance lock if a center position is provided
            ssr = ssr_distance_lock(ssr, center, 10, pos)

        candidates = []
        preds = []
        unique_errors = []

        # Iterate over unique SSR values in ascending order
        for unique_error in np.unique(np.sort(ssr)):
            curvatures = []
            for candidate in np.where(ssr==unique_error)[0]:
                # Perform quadratic interpolation
                _,c,_,_=quadratic_interpolation_deep_encodings(pos[candidate],img_cielab,3,encoder,ref_encoding)
                # Check if the interpolation coefficients are valid
                if c is None or not hasattr(c, "__getitem__"): 
                    continue
              
                # Check if the interpolation coefficients satisfy the conditions for a global minimum
                D = 4*c[5]*c[4]-c[3]**2
                if D>0 and c[4]>0:
                    curvatures.append((D, candidate))
                    unique_errors.append(unique_error)

            # If any candidates are found, sort them by curvature in descending order
            if curvatures:
                curvatures.sort(reverse=True)
                candidates.extend([candidate for _, candidate in curvatures])    #append the candidate from the highest curvature to the lowest
            
            # If the number of candidates reaches the total rank range, break the loop
            if len(candidates) >= total_rank_range:
                candidates = candidates[0:total_rank_range]
                unique_errors = unique_errors[0:total_rank_range]
                break
        
        # Append the candidates, unique SSR values, and SSR values to their respective lists
        all_candidates.append(candidates)
        all_unique_ssr.append(unique_errors)
        all_ssr.append(unique_errors)

    # Convert the candidate lists to a numpy array
    all_candidates = np.array(all_candidates)
    # Find unique candidates from the top candidates of each reference encoding
    unique_candidates = np.unique(all_candidates[:,0:unique_rank_range])
    ranks = []
    rank = 0

    # Calculate the ranks for each unique candidate based on their positions in the candidate lists
    for unique_candidate in unique_candidates:
        for i, candidates in enumerate(all_candidates):
            if not unique_candidate in candidates:
                rank += total_rank_range + 2
            else:
                rank += np.where(candidates==unique_candidate)[0][0]
        ranks.append(rank)
        rank = 0

    # Find the candidate(s) with the minimum rank
    rank_min_indexs = np.where(ranks==np.min(ranks))[0]

    pred_ssr_min = 0
    pred_candidate_min = 0
    pred_ssr = 0
    pred_ssr_sum = 0
    pred_ssr_sum_min = 0
    
    # For each candidate with the minimum rank
    for rank_min_index in rank_min_indexs:
        # print("rank_min_index = ", rank_min_index)
        pred_candidate = unique_candidates[rank_min_index]
        pred_ssr_index_min = -1
        pred_ssr_min = 0

        # Find the reference encoding index that gives the minimum SSR value for the candidate
        for i, candidates in enumerate(all_candidates):
            if not pred_candidate in candidates:
                continue
            else:
                # print("pred_candidate = ", pred_candidate)
                pred_ssr_index = np.where(candidates==pred_candidate)[0][0]
                pred_ssr_sum += all_unique_ssr[i][pred_ssr_index]
                if pred_ssr_index_min == -1:
                    pred_ssr_index_min = pred_ssr_index
                    pred_ssr = all_unique_ssr[i][pred_ssr_index]
                    pred_ssr_min = pred_ssr
                    reference_encoding_index = i
                else:
                    if pred_ssr_index <= pred_ssr_index_min:
                        pred_ssr = all_unique_ssr[i][pred_ssr_index]
                        if pred_ssr < pred_ssr_min:
                            # print("i =", i, "pred_ssr_index = ", pred_ssr_index, "pred_candidate =", pred_candidate)
                            reference_encoding_index = i
                            pred_ssr_min = pred_ssr

        # Update the predicted position, minimum SSR value, and new reference encoding index if necessary
        if pred_ssr_sum_min == 0:
            # print("pred_ssr_sum = ", pred_ssr_sum)
            pred_ssr_sum_min = pred_ssr_sum
            pred_candidate_min = pred_candidate
            new_ref_encoding_index = reference_encoding_index
            
        else:
            if pred_ssr_sum < pred_ssr_sum_min:
                pred_ssr_sum_min = pred_ssr_sum
                pred_candidate_min = pred_candidate
                new_ref_encoding_index = reference_encoding_index
                
        pred_ssr_sum = 0

    # Get the predicted position based on the minimum candidate
    pred = pos[pred_candidate_min]
    
    
    return pred, new_ref_encoding_index

def DFE_tracking_old_version(img_path, center_set, save_path, name, number=1):
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
            - Append the predicted position to the list of predictions.
        7. Return the array of predicted positions.
    """
    # Initialize variables and load the first frame
    center = center_set[0]
    start_frame = 0
    path = img_path
    filenames=sorted_alphanumeric(os.listdir(path))
    print(f"Got {len(filenames)} files")

    window_size=(31,31)
    img=cv2.imread(path+filenames[start_frame])
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=convertRGB2CIELab(img)
    
    # Extract the reference crop from the first frame
    x1 = int(center[0]-window_size[0]/2)
    x2 = int(center[0]+window_size[0]/2)
    y1 = int(center[1]-window_size[1]/2)
    y2 = int(center[1]+window_size[1]/2)
    ref_crop=np.array(img[y1:y2,x1:x2,:])
    
    print(f"Center at {center}, crop corr is {x1, x2, y1, y2}")
    
    model=load_model("../checkpoints/DFE_hand.h5", custom_objects = {"tensorflow": tensorflow})
    encoder=Model(model.input, model.layers[int(len(model.layers)/2)-1].output) #get only half of the network
    encoder.compile(optimizer='adamax',loss='mse')
    ref_encoding=np.squeeze(encoder.predict(np.array([ref_crop]), batch_size=2**9, verbose=1))
    print(ref_encoding.shape)

    # Initialize the reference encodings list with the reference encoding
    ref_encodings = []
    for i in range(number):
        ref_encodings.append(ref_encoding)
    
    stride=1
    dfe_preds=[center]
    dfe_preds_subpixel=[center]
    ref_len = 1
    # start_time=time.perf_counter()

    # Iterate over the frames starting from the second frame
    for f in range(1,len(filenames)):
        if f != 1:
            img_cielab_clone = img_cielab.copy() # save previous frame for extra crop
        img=cv2.imread(path+filenames[f])
        img_cielab=convertRGB2CIELab(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        width_step = int((img_cielab.shape[0]-window_size[0])/stride)
        height_step = int((img_cielab.shape[1]-window_size[1])/stride)
        pos=[]
        crops=[]
        # print("img_cielab.shape", img_cielab.shape)

        # Generate crops and their positions from the current frame
        for i in range(int((img_cielab.shape[0]-window_size[1])/stride)):
            for j in range(int((img_cielab.shape[1]-window_size[0])/stride)):
                x=int(j+np.ceil(window_size[0]/2))
                y=int(i+np.ceil(window_size[1]/2))
                crops.append(np.array(img_cielab[i:i+window_size[1],j:j+window_size[0],:]))
                pos.append([x,y]) #position of centers

        crops=np.array(crops)
        # print("crops.shape", crops.shape)

        # Convert the crops to encodings using the encoder
        encodings=np.squeeze(encoder.predict(crops,batch_size=2**9,verbose=1)) # Convert all crops to encodings
        
        print(f"Got {len(ref_encodings)} ref_crops")

        
        # Compare the encodings with the reference encodings to find the best match
        pred, new_ref_encoding_index = search_global_minimum_average_old_version(pos, encodings, ref_encodings, img_cielab, encoder, center = dfe_preds[-1])
        final_ref_encoding = ref_encodings[new_ref_encoding_index]

        # Apply sub-pixel refinement using quadratic interpolation
        pred_subpixel,c,A,B=quadratic_interpolation_deep_encodings(pred, img_cielab, 3, encoder, final_ref_encoding)
        pred_subpixel = pred_subpixel[0]

        # Append the predicted position to the list of predictions
        dfe_preds.append(pred)
        dfe_preds_subpixel.append(pred_subpixel)
        
        # np.save(save_path + f"{name[0]}_preds.npy", dfe_preds_save)

        # Update the reference encodings list by removing the middle encoding and appending the new reference encoding (head-tail approach)
        if f > ref_len-1:
            ref_encodings.pop(int(ref_len/2))
            # ref_encodings.pop((ref_len-2))
        center_this_pred = dfe_preds[-1]
        temp_center = [round(center_this_pred[0]), round(center_this_pred[1])]
        new_ref_crop=np.array(img_cielab[int(temp_center[1]-window_size[0]/2):int(temp_center[1]+window_size[0]/2)
                            ,int(temp_center[0]-window_size[1]/2):int(temp_center[0]+window_size[1]/2),:])
        new_ref_encoding=np.squeeze(encoder.predict(np.expand_dims(new_ref_crop,axis=0)))
        ref_encodings.append(new_ref_encoding)

        # if f < ref_len:
        #     ref_encodings.append(new_ref_encoding)

    # Return the array of predicted positions (sub-pixel accuracy)
    dfe_preds_save = np.array(dfe_preds_subpixel)
    return dfe_preds_save