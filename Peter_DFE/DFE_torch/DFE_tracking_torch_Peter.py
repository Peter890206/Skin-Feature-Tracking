from PIL import Image
import numpy as np
import os
import re
import time
from tqdm import tqdm
# import kornia
from torch.utils.data import Dataset
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from models.EfficientNet_model import efficientnet_b0
from models.Resnet34_model import resnet34, resnet50
from models.Swin_transformer_model import swin_transformer_base
from models.swin_transformer_v2 import SwinTransformerV2
from models.ConvNeXt_model import convnext_base
from models.EfficientNetV2_model import efficientnetv2_s
from utils.utils_Peter import *
from Dataset import Dataset_for_tracking


def get_args_parser():
    parser = argparse.ArgumentParser('DFE_EfficientNet for encoding', add_help=False)
    parser.add_argument('--batch_size', default=2048, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * 4 * # gpus')

    parser.add_argument('--window_size', default=32, type=int,
                        help='images input size')
    
    parser.add_argument('--center_coord', default=[24, 24], type=int,
                        help='center coordinates of the area cropped from prediction of Cotracker')
    
    parser.add_argument('--name', default='swin_30_30_100')    #resnet34_30_30_100

    parser.add_argument('--data_name', default='04941364_R_B_little_finger_nail',
                        help='data name for saving')

    # * Finetuning params
    parser.add_argument('--checkpoint_path', default='../checkpoints/model_finetune_swin_cnn_adamw_30_30_100_10_30_2048_49.pth',    
                        help='checkpoint')    
    # model_finetune_efficientnet_adamw_0_0_100_15_35_10_30_2048_47.pth, model_finetune_efficientnet_cnn_adamw_30_30_100_10_30_2048_49.pth, 
    # model_finetune_resnet34_adamw_100_10_30_2048_48.pth, model_finetune_resnet34_cnn_adamw_30_30_100_10_30_2048_48.pth
    # model_finetune_resnet50_cnn_adamw_100_10_30_2048_49.pth, model_finetune_resnet50_cnn_adamw_30_30_100_10_30_2048_47.pth
    # model_finetune_ConvNeXt_cnn_adamw_100_10_30_2048_48.pth, model_finetune_ConvNeXt_cnn_adamw_30_30_100_10_30_2048_48.pth
    # model_finetune_swin_cnn_adamw_100_10_30_2048_49.pth, model_finetune_swin_cnn_adamw_30_30_100_10_30_2048_49.pth
    # model_finetune_efficientnetv2_cnn_adamw_100_10_30_2048_47.pth, model_finetune_efficientnetv2_cnn_adamw_30_30_100_10_30_2048_48.pth
    # Dataset parameters
    parser.add_argument('--data_path', default='../Process_Video_DfeTracking/04941364_R_B_little_finger_nail_crop_image_handmole_24_24_avg/', type=str,
                        help='dataset path')    #./crop_image_handmole_24_24_avg/    ./04941364_R_B_crop_image_handmole_24_24_avg/
    
    parser.add_argument('--output_dir', default='../Process_Video_DfeTracking/result',
                        help='you should not add "/" at the end of the path.')
    
    parser.add_argument('--tracking_type', default='dynamic',
                        help='type of tracking, ordered or dynamic')    # the dynamic here means the head-tail tracking algorithm in the paper
    
    parser.add_argument('--device', default='cuda', 
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=16, type=int,
                        help='num of workers in dataloader')

    parser.set_defaults(pin_mem=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU. \
                        But this will require more memory.')
    # distributed training parameters
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser



def calculate_sorted_candidates(ref_encoding, encodings, center, pos, offsets, width_step, height_step, total_rank_range):
    """
    Calculate a sorted list of candidate indices based on their sum of squared residuals (SSR) with respect to a reference encoding.

    Args:
        ref_encoding (np.ndarray): The reference encoding to compare against.
        encodings (np.ndarray): The set of encodings to compare with the reference encoding.
        center (list): The center coordinates to prioritize in the search.
        pos (list): The list of positions corresponding to each encoding.
        offsets (np.ndarray): The offsets to consider when calculating candidate groups.
        width_step (int): The step size along the width dimension.
        height_step (int): The step size along the height dimension.
        total_rank_range (int): The maximum number of candidates to return.

    Returns:
        list: A sorted list of candidate indices based on their SSR.
        list: A list of unique SSR values corresponding to the sorted candidates.
        np.ndarray: The SSR values for all encodings.
    """
    # Add an extra dimension to the reference encoding
    ref_encoding = np.expand_dims(ref_encoding, axis=0)
    # Calculate SSR for all encodings
    ssr=np.sum((encodings-ref_encoding)**2,axis=1)
    
    # Lock the search to a certain area around the center if provided
    if isinstance(center, list):
        ssr = ssr_distance_lock(ssr, center, 10, pos)
    
    candidates = []
    unique_errors = []

    # Iterate over unique SSR values in ascending order
    for unique_error in np.unique(np.sort(ssr)):
        curvatures = []
        for candidate in np.where(ssr==unique_error)[0]:
            # Calculate the candidate group around the current candidate
            offset_meshgrid, candidate_meshgrid = np.meshgrid(offsets, candidate)
            cadidate_group = candidate_meshgrid + offset_meshgrid
            error_index = np.logical_and(cadidate_group >= 0, cadidate_group < width_step*height_step)
            cadidate_group = cadidate_group * error_index    # Ensure indices are in range, else set it to 0
            cadidate_group = np.squeeze(cadidate_group)

            # Extract SSR and position values for the candidate group for acceleration
            ssr_for_quadratic = ssr[cadidate_group]
            window_pts = pos[cadidate_group]

            # Perform quadratic interpolation for the current candidate
            _,c,_,_=quadratic_interpolation_deep_encodings(pos[candidate], window_pts, ssr_for_quadratic)
            
            if c is None or not hasattr(c, "__getitem__"):  # 检查 c 是否为 None 或不可下标访问
                continue
            
            # Calculate the curvature and append the candidate if valid
            D = 4*c[5]*c[4]-c[3]**2
            if D>0 and c[4]>0:
                curvatures.append((D, candidate))
                unique_errors.append(unique_error)
        
        # Sort the candidates in descending order of curvature and append them to the list
        if curvatures:
            curvatures.sort(reverse=True)
            candidates.extend([candidate for _, candidate in curvatures])    
        
        # Stop if the desired number of candidates is reached   
        if len(candidates) >= total_rank_range:
            candidates = candidates[0:total_rank_range]
            unique_errors = unique_errors[0:total_rank_range]
            break

    return candidates, unique_errors, ssr

def search_global_minimum_average(pos, encodings, ref_encodings, width_step, height_step, center, total_rank_range = 5, unique_rank_range = 3):
    """
    Find the global minimum candidate by averaging the sum of squared residuals (SSR) across multiple reference encodings.

    Args:
        pos (np.ndarray): The list of positions corresponding to each encoding.
        encodings (np.ndarray): The set of encodings to search for the global minimum.
        ref_encodings (list): A list of reference encodings to compare against.
        width_step (int): The step size along the width dimension.
        height_step (int): The step size along the height dimension.
        center (list, optional): The center coordinates to prioritize in the search.
        total_rank_range (int, optional): The maximum number of candidates to consider. Default is 5.
        unique_rank_range (int, optional): The number of unique candidates to consider for ranking. Default is 3.

    Returns:
        tuple:
            - np.ndarray: The predicted position of the global minimum candidate.
            - float: The minimum SSR value of the predicted candidate.
            - np.ndarray: The new reference encoding corresponding to the predicted candidate.
            - np.ndarray: The positions of the window points around the predicted candidate.
            - np.ndarray: The SSR values of the window points around the predicted candidate.
            - list: A list of all candidate indices for each reference encoding.
    """

    # Prepare the input data
    pos = np.array(pos)
    ref_encodings = [ref_encoding.cpu().numpy() if type(ref_encoding) is not np.ndarray else ref_encoding for ref_encoding in ref_encodings]
    if type(encodings) is not np.ndarray:
        encodings = encodings.cpu().numpy()
    all_candidates = []
    all_unique_ssr = []
    all_ssr = []
    offsets = np.array([-width_step - 1, -width_step, -width_step + 1, -1, 0, 1, width_step - 1, width_step, width_step + 1])

    # Calculate sorted candidates for each reference encoding
    for ref_encoding in ref_encodings:
        candidates, unique_errors, ssr = calculate_sorted_candidates(ref_encoding, encodings, center, pos, offsets, width_step, height_step, total_rank_range)
        all_candidates.append(candidates)
        all_unique_ssr.append(unique_errors)
        all_ssr.append(ssr)

    all_candidates = np.array(all_candidates)
    unique_candidates = np.unique(all_candidates[:,0:unique_rank_range])

    """
    Calculate the ranks for each reference vector based on their unique candidates.
    The rank is computed by summing the indices of the unique candidates in each reference vector.
    A lower rank indicates a better candidate.
    """
    ranks = []
    rank = 0
    for unique_candidate in unique_candidates:
        for i, candidates in enumerate(all_candidates):
            if not unique_candidate in candidates:
                rank += total_rank_range + 2    # Add a large value if the candidate is not present
            else:
                rank += np.where(candidates==unique_candidate)[0][0]    # Add the index of the candidate
        ranks.append(rank)
        rank = 0
    
    # Find the candidate with the minimum rank and corresponding SSR value
    rank_min_indexs = np.where(ranks==np.min(ranks))[0]
    pred_ssr_min = 0
    pred_candidate_min = 0
    pred_ssr = 0
    pred_ssr_sum = 0
    pred_ssr_sum_min = 0
    reference_encoding_index = -1
    new_ref_encoding_index = 0

    # Find the candidate with the minimum sum of SSR across reference encodings
    for rank_min_index in rank_min_indexs:
        pred_candidate = unique_candidates[rank_min_index]
        pred_ssr_index_min = -1
        pred_ssr_min = 0

        # Find the reference encoding with the minimum SSR for the predicted candidate
        for i, candidates in enumerate(all_candidates):
            if not pred_candidate in candidates:
                continue
            else:
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
                            reference_encoding_index = i
                            reference_encoding = encodings[pred_candidate]
        
        # Update the minimum sum of SSR and corresponding candidate
        if pred_ssr_sum_min == 0:
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

    # Calculate the window points and SSR values around the predicted candidate for acceleration in the next step
    offset_min_meshgrid, candidate_min_meshgrid = np.meshgrid(offsets, pred_candidate_min)
    cadidate_min_group = offset_min_meshgrid + candidate_min_meshgrid
    error_index = np.logical_and(cadidate_min_group >= 0, cadidate_min_group < width_step*height_step)
    cadidate_min_group = cadidate_min_group * error_index 
    cadidate_min_group = np.squeeze(cadidate_min_group)
    pred = pos[pred_candidate_min]
    window_pts_min = pos[cadidate_min_group]
    new_ssr_for_quadratic = all_ssr[new_ref_encoding_index][cadidate_min_group]

    return pred, pred_ssr_min, new_ref_encoding, window_pts_min, new_ssr_for_quadratic, all_candidates

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

def DFE_tracking(img_path, encoder, device, center_set, args, ref_len):
    """
    Perform Deep Feature Exploration (DFE) tracking on a sequence of images.

    This function takes an image path, a pre-trained encoder model, and an initial center point as input. 
    It then tracks the point of interest across the image sequence using the DFE algorithm. 
    The tracking can be performed in either "ordered" or "dynamic" mode, where the reference encodings are updated differently.

    Args:
        img_path (str): The path to the directory containing the image sequence.
        encoder (torch.nn.Module): The pre-trained encoder model used for encoding the image crops.
        device (torch.device): The device to use for computations (e.g., CPU or GPU).
        center_set (list): A list containing the initial center coordinates for tracking.
        args (argparse.Namespace): The command-line arguments containing various configuration options.
        ref_len (int): The length of the reference encoding list for dynamic tracking.

    Returns:
        tuple:
            - np.ndarray: The predicted sub-pixel coordinates for each frame in the sequence.
            - list: A list of candidate indices for each frame, used for visualization or analysis.
    """
    
    center = center_set[0]
    start_frame = 0
    path = img_path
    filenames=sorted_alphanumeric(os.listdir(path))
    print(f"Got {len(filenames)} files")

    window_size = (args.window_size, args.window_size)
    img_path = path + filenames[start_frame]
    img = Image.open(img_path)
    
    # Crop the initial reference image
    x1 = int(center[0]-window_size[0]/2)
    x2 = int(center[0]+window_size[0]/2)
    y1 = int(center[1]-window_size[1]/2)
    y2 = int(center[1]+window_size[1]/2)
    ref_crop = img.crop((x1, y1, x2, y2))

    print(f"Center at {center}, crop corr is {x1, x2, y1, y2}")
    
    # Load the pre-trained model weights
    pretrained_weights = torch.load(args.checkpoint_path, map_location='cpu')
    encoder_state_dict = {k: v for k, v in pretrained_weights.items() if 'encoder' in k}
    modified_weights = {}
    for k, v in encoder_state_dict.items():
        if k.startswith('module.encoder'):
            new_key = k[len('module.encoder.'):]  # remove 'module.encoder' prefix
            modified_weights[new_key] = v

    encoder.load_state_dict(modified_weights, strict=True)

    # Disable gradient computation for the encoder
    for param in encoder.parameters():
        param.requires_grad = False

    encoder.to(device)
    encoder.eval()

    # Encode the initial reference crop
    ref_crop_data = Dataset_for_tracking([ref_crop])    #[ref_crop]
    ref_data_loader = torch.utils.data.DataLoader(
            ref_crop_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    ref_encoding=np.squeeze(predict(encoder, ref_data_loader, device))
    print("ref_encoding shape", ref_encoding.shape)
    
    # Initialize the list of reference encodings
    ref_encodings = [ref_encoding]
    stride=1
    # Initialize the list of predicted centers and sub-pixel centers
    dfe_preds=[center]
    dfe_preds_subpixel = [center]
    # Initialize the list to store candidate indices for visualization
    all_candidates_save = []
    x_offset = np.ceil(window_size[0]/2)
    y_offset = np.ceil(window_size[1]/2)
    height_step = int((img.size[1]-window_size[1])/stride)
    width_step = int((img.size[0]-window_size[0])/stride)
    
    # Iterate over the image sequence
    for f in tqdm(range(1,len(filenames))):
    
        img_path = path + filenames[f]
        img_cielab = Image.open(img_path)
        pos=[]
        crops=[]
        
        # Extract crops from the current frame
        for i in range(height_step):
            for j in range(width_step):
                x=int(j+x_offset)
                y=int(i+y_offset)
                crops.append(img_cielab.crop((j, i, j+window_size[0], i+window_size[1])))
                pos.append([x,y]) #position of centers

        crops_data = Dataset_for_tracking(crops)
        data_loader = torch.utils.data.DataLoader(
            crops_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        # Encode the crops from the current frame
        encodings = predict(encoder, data_loader, device)
        encodings=np.squeeze(encodings) 

        # Find the global minimum candidate using the search_global_minimum_average function
        pred, pred_ssr_min, new_ref_encoding, window_pts_min, new_ssr_for_quadratic, all_candidates = search_global_minimum_average(pos, encodings, ref_encodings, width_step, height_step, center = dfe_preds[-1])
        
        # Perform quadratic interpolation to get sub-pixel accuracy
        pred_subpixel,c,A,B=quadratic_interpolation_deep_encodings(pred, window_pts_min, new_ssr_for_quadratic)
        pred_subpixel = pred_subpixel[0]

        # Append the predicted center and sub-pixel center to the lists
        dfe_preds.append(pred)
        dfe_preds_subpixel.append(pred_subpixel)

        # Update the reference encodings based on the tracking type
        if ref_len != 1:
            if args.tracking_type == "ordered":
                if f < ref_len:
                    ref_encodings.append(new_ref_encoding)
            elif args.tracking_type == "dynamic":
                if f > ref_len-1:
                    ref_encodings.pop(int(ref_len/2))    # Remove the middle reference encoding
                #     # ref_encodings.pop((ref_len-2))
                ref_encodings.append(new_ref_encoding)

            
            # if f in [2,4,6,8]:
            #     ref_encodings.append(new_ref_encoding)

        # Store the candidate indices for visualization
        all_candidates_save.append(all_candidates)
        
    dfe_preds_save = np.array(dfe_preds_subpixel)
    return dfe_preds_save, all_candidates_save
 
if __name__ == '__main__':
    """
    Main function for running the Deep Feature Exploration (DFE) tracking algorithm.

    This function serves as the entry point for executing the DFE tracking algorithm. 
    It performs the following steps:

    1. Parse command-line arguments for configuration options using the `get_args_parser` function.
    2. Print the current model, data, and tracking type being used.
    3. Initialize lists to store total time and total errors for multiple iterations.
    4. Set the initial center coordinates and device for computation.
    5. Initialize the encoder model (e.g., EfficientNetV2) for encoding the image crops.
    6. Run multiple iterations of the DFE tracking algorithm:
        a. Call the `DFE_tracking` function to perform tracking and obtain predicted coordinates and candidate indices.
        b. Calculate the total error by comparing the predicted coordinates with the ground truth.
        c. Measure the total time taken for the current iteration.
        d. Append the total error and total time to the respective lists.
    7. Save the total time and total errors for the ordered and dynamic tracking types to separate files.

    The main function serves as a convenient way to run the DFE tracking algorithm with different configurations and
    evaluate its performance by comparing the predicted coordinates with the ground truth. 
    The total time and total errors are saved for further analysis or comparison with other tracking algorithms.

    Note: Make sure you are in the correct environment, and provide the correct paths for the checkpoint, data, and ground truth files.
    """

    args = get_args_parser()
    args = args.parse_args()
    print(f"Now using Model: {args.name}, Data: {args.data_name}, Tracking type: {args.tracking_type}")
    total_time = []
    total_errors = []
    center_set = [args.center_coord]    # Set the initial center coordinates
    device = torch.device(args.device)
    encoder = swin_transformer_base(num_classes=0)    #change the model encoder here
    
    # Run multiple iterations
    for i in range(1, 20):
        print("Start iteration: ", i)
        start_time = time.time()

        # Perform DFE tracking and get predicted coordinates and candidate indices
        prediction, all_candidates_save = DFE_tracking(args.data_path, encoder, device, center_set, args, i)
        end_time = time.time()
        print(f"Total time: {end_time-start_time}")
        # np.save(os.path.join(args.output_dir, f"{args.name}_preds.npy"), prediction)

        # Calculate total error by comparing with ground truth
        Ground_truth_path = "../cotracker_pred_and_ground_truth/04941364_R_B_little_finger_nail_coordinates_mean.npy"    #04941364_R_B_ground_truth_coordinates_mean.npy, Peter_handmole_ground_truth_mean.npy
        cotracker_path = "../cotracker_pred_and_ground_truth/04941364_R_B_little_finger_nail_cotracker_pred_coords.npy"    #04941364_R_B_cotracker_pred_178_116.npy, Handmole_cotracker_pred_220_238.npy
        cotracker = np.load(cotracker_path)
        Ground_truth = np.load(Ground_truth_path)

        cotracker_x = cotracker[:, 0]
        cotracker_y = cotracker[:, 1]   
        Ground_truth_x = Ground_truth[:, 0]
        Ground_truth_y = Ground_truth[:, 1]  
        prediction_x = prediction[:, 0]
        prediction_y = prediction[:, 1]

        # Adjust the predicted coordinates of Cotracker-DFE based on the area size
        distance_for_efficientnet_revise_x = args.center_coord[0] - prediction_x
        distance_for_efficientnet_revise_y = args.center_coord[1] - prediction_y
        cotracker_revised_x = cotracker_x - distance_for_efficientnet_revise_x
        cotracker_revised_y = cotracker_y - distance_for_efficientnet_revise_y

        # Calculate the revised distances between the adjusted Cotracker-DFE predictions and ground truth
        revised_distances = np.sqrt((cotracker_revised_x - Ground_truth_x)**2 + (cotracker_revised_y - Ground_truth_y)**2)
        # DFE_distances = np.sqrt((prediction_x - Ground_truth_x)**2 + (prediction_y - Ground_truth_y)**2)

        cotracker_error = np.sum(np.sqrt((cotracker_x - Ground_truth_x)**2 + (cotracker_y - Ground_truth_y)**2))
        print("cotracker_error = ", cotracker_error)
        total_error = np.sum(revised_distances)
        print("total_error = ", total_error)
        total_errors.append(total_error)
        total_time.append(end_time-start_time)
        

    total_errors = np.array(total_errors)
    total_time = np.array(total_time)

    # Save total time and total errors for ordered and dynamic tracking types
    if args.tracking_type == "ordered":
        np.save(f"{args.output_dir}/{args.data_name}_{args.name}_ordered_total_time.npy", total_time)
        np.save(f"{args.output_dir}/{args.data_name}_{args.name}_ordered_total_errors.npy", total_errors)
    elif args.tracking_type == "dynamic":
        np.save(f"{args.output_dir}/{args.data_name}_{args.name}_dynamic_total_time.npy", total_time)
        np.save(f"{args.output_dir}/{args.data_name}_{args.name}_dynamic_total_errors.npy", total_errors)
    else:
        np.save(f"{args.output_dir}/{args.data_name}_{args.name}_total_time.npy", total_time)
        np.save(f"{args.output_dir}/{args.data_name}_{args.name}_total_errors.npy", total_errors)
 





# """
# #old algorithm for tracking
# """
# def search_global_minimum(pos, encodings, ref_encodings, img_cielab, encoder, center = None, text = ""):
#     for ref_encoding in ref_encodings:
#         if type(ref_encoding) is not np.ndarray:
#             ref_encoding = ref_encoding.cpu().numpy()
#         if type(encodings) is not np.ndarray:
#             encodings = encodings.cpu().numpy()
#         ref_encoding = np.expand_dims(ref_encoding, axis=0)
#         # print("ref_encoding.shape", ref_encoding.shape)
#         # print("encodings.shape", encodings.shape)
#         ssr=np.sum((encodings-ref_encoding)**2,axis=1)
#         compare_ssr = []; compare_pred = []
#         curvatures = []; candidates = []
#         for unique_error in np.unique(np.sort(ssr)):
#             for candidate in np.where(ssr==unique_error)[0]:
#                 _,c,_,_=quadratic_interpolation_deep_encodings(pos[candidate],img_cielab,3,encoder,ref_encoding)
#                 # print("c = ", c)
#                 if c is None or not hasattr(c, "__getitem__"):  # 检查 c 是否为 None 或不可下标访问
#                     continue
#                 # elif c.any() == None:
#                 #     continue
#                 # if np.isnan(c).any():
#                 #     continue
#                 D = 4*c[5]*c[4]-c[3]**2
#                 if D>0 and c[4]>0:
#                     curvatures.append(D)
#                     candidates.append(candidate)
                    
#             if bool(curvatures):
#                 # print("curvatures: ", curvatures)
#                 print("candidates: ", candidates)
#                 print(f'Selected Curvature {str(np.max(curvatures))} and SSR value {unique_error}'+ text)
#                 break
#         pred = pos[candidates[np.argmax(curvatures)]]
#         compare_ssr.append(unique_error)
#         compare_pred.append(pred)
#         # print(pred)
#         # print("------------------")
#     compare_ssr = np.array(compare_ssr)
#     pred = compare_pred[np.argmin(compare_ssr)]
#     final_encoding = ref_encodings[np.argmin(compare_ssr)]
#     return pred, final_encoding


# """
# failed algorithm (the result is not good)
# """
# def calculate_average_ssr_rank(encodings, ref_encodings_previous_frame, candidates, width_step, height_step, area_distance):

#     candidates = np.array(candidates)
#     if area_distance != 0:
#         offsets = np.array([-width_step - area_distance, -width_step, -width_step + area_distance, -area_distance, 0, area_distance, width_step - area_distance, width_step, width_step + area_distance])
#     else:
#         offsets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    
#     # candidates_meshgrid, offset_meshgrid = np.meshgrid(candidates, offsets)
#     # cadidates_groups = candidates_meshgrid + offset_meshgrid
#     offset_meshgrid, candidates_meshgrid = np.meshgrid(offsets, candidates)
#     cadidates_groups = candidates_meshgrid + offset_meshgrid
#     # print("cadidates_groups.shape", cadidates_groups.shape)
#     error_index = np.logical_and(cadidates_groups >= 0, cadidates_groups < width_step*height_step)    
#     True_index_rows = np.sum(error_index, axis = 1)
#     cadidates_groups = cadidates_groups * error_index    # make sure the index is in range, else set it to 0
    
#     encoding_groups = encodings[cadidates_groups]
#     encoding_groups = encoding_groups * np.expand_dims(error_index, axis=-1)
#     # print("encoding_groups.shape", encoding_groups.shape)
#     # encoding_groups = np.transpose(encoding_groups, (1, 0, 2))
#     # print("Encoding groups shape: ", encoding_groups.shape)
#     ssr_totals = []
#     for i, encoding_group in enumerate(encoding_groups):
#         square_distance = (encoding_group - ref_encodings_previous_frame)**2
#         # square_distance = square_distance * error_index[i]    # make sure the ssr value of error index is 0
#         # ssr_total=(np.sum(square_distance, axis=1)) / True_index_rows[i]    # True_index_rows[i] = the number of index in the range
#         ssr_total=np.mean(np.sum(square_distance, axis=1))
#         ssr_totals.append(ssr_total)
#     ssr_totals = np.array(ssr_totals)
#     sorted_indices = np.argsort(ssr_totals)

#     return candidates[sorted_indices], encoding_groups[sorted_indices], ssr_totals[sorted_indices]



# def search_global_minimum_average_multiple_area(pos, img_cielab, encoder, encodings, ref_encodings, width_step, height_step, area_distance, center, total_rank_range = 5, unique_rank_range = 3):
#     all_candidates = []
#     all_encoding_groups_sorted = []
#     all_ssr_totals = []
#     if type(encodings) is not np.ndarray:
#         encodings = encodings.cpu().numpy()
#         # encodings = encodings.numpy()
#     # print("Encoding shape: ", encodings.shape)
    
#     for ref_encoding in ref_encodings:
#         if type(ref_encoding) is not np.ndarray:
#             ref_encoding = ref_encoding.cpu().numpy()
#         # print("Encoding shape: ", encodings.shape)
#         ref_encoding_group = ref_encoding
#         ref_encoding_center = ref_encoding[4]    # 4 is the center
#         ref_encoding_center = np.expand_dims(ref_encoding_center, axis=0)
#         # print("Ref encoding shape: ", ref_encoding_center.shape)
#         ssr=np.sum((encodings-ref_encoding_center)**2,axis=1)
#         # print("SSR shape: ", ssr.shape)
#         if isinstance(center, list):
#             # print("start lock")
#             ssr = ssr_distance_lock(ssr, center, 10, pos)
        
#         candidates = []; preds = []; unique_errors = []
#         for unique_error in np.unique(np.sort(ssr)):
#             curvatures = []
#             for candidate in np.where(ssr==unique_error)[0]:
#                 # print("start quadratic_interpolation_deep_encodings")
#                 _,c,_,_=quadratic_interpolation_deep_encodings(pos[candidate], img_cielab, 3, encoder, ref_encoding[4])
#                 # print("finish quadratic_interpolation_deep_encodings")
                
#                 if c is None or not hasattr(c, "__getitem__"):  # 检查 c 是否为 None 或不可下标访问
#                     continue
                
#                 D = 4*c[5]*c[4]-c[3]**2
#                 if D>0 and c[4]>0:
#                     curvatures.append((D, candidate))
#                     unique_errors.append(unique_error)
#                 else:
#                     print("None")
#             if curvatures:
#                 curvatures.sort(reverse=True)
#                 candidates.extend([candidate for _, candidate in curvatures])    #append the candidate from the highest curvature to the lowest
            
                
#             if len(candidates) >= total_rank_range:
#                 candidates = candidates[0:total_rank_range]
#                 break
            
#             #not using quadratic interpolation
            
#             # for candidate in np.where(ssr==unique_error)[0]:
#             #     # print("Candidate: ", candidate)
#             #     candidates.append(candidate)
#             #     unique_errors.append(unique_error)
#             # if len(candidates) >= total_rank_range:
#             #     candidates = candidates[0:total_rank_range]
#             #     break
#         # print("len candidates: ", len(candidates))   
        
#         candidates_sorted, encoding_groups_sorted, ssr_totals = calculate_average_ssr_rank(encodings, ref_encoding_group, candidates, width_step, height_step, area_distance)

#         all_candidates.append(candidates_sorted)
#         all_encoding_groups_sorted.append(encoding_groups_sorted)
#         all_ssr_totals.append(ssr_totals)
#     all_candidates = np.array(all_candidates)
#     # print("all_candidates shape: ", all_candidates.shape)
#     unique_candidates = np.unique(all_candidates[:,0:unique_rank_range])
#     ranks = []
#     rank = 0
#     for unique_candidate in unique_candidates:
#         for i, candidates in enumerate(all_candidates):
#             if not unique_candidate in candidates:
#                 rank += total_rank_range + 2
#             else:
#                 rank += np.where(candidates==unique_candidate)[0][0]
#         ranks.append(rank)
#         rank = 0
    
#     rank_min_indexs = np.where(ranks==np.min(ranks))[0]
#     pred_ssr_min = 0
#     pred_candidate_min = 0
#     pred_ssr = 0
#     for rank_min_index in rank_min_indexs:
#         pred_candidate = unique_candidates[rank_min_index]
#         for i, candidates in enumerate(all_candidates):
#             if not pred_candidate in candidates:
#                 continue
#             else:
#                 index = np.where(candidates==pred_candidate)[0][0]
#                 pred_total_ssr = all_ssr_totals[i][index]
#                 ref_encodings_group = all_encoding_groups_sorted[i][index]
#                 break
#         if pred_ssr_min == 0:
#             pred_ssr_min = pred_total_ssr
#             pred_candidate_min = pred_candidate
#             new_ref_encodings_group = ref_encodings_group
#         if pred_total_ssr < pred_ssr_min:
#             pred_ssr_min = pred_total_ssr
#             pred_candidate_min = unique_candidates[rank_min_index]
#             new_ref_encodings_group = ref_encodings_group
#     # print("pred_candidate_min ", pred_candidate_min)  
#     pred = pos[pred_candidate_min]
    
#     return pred, pred_ssr_min, new_ref_encodings_group, all_ssr_totals
