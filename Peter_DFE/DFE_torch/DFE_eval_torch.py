"""
DFE Torch Evaluation Script

This script is used to evaluate a Deep Feature Exploration (DFE) model on a test dataset. 
The DFE model is a deep learning architecture designed for image classification tasks. 
The script supports various DFE model variants, such as DFE_EfficientNet, DFE_Resnet34, DFE_Swin_transformer, and more.

The script loads the test dataset, initializes the specified DFE model, and computes the evaluation loss (Mean Squared Error) on the test data. 
It allows for configuring various parameters, such as batch size, input size, window size, and loading a pre-trained model checkpoint.

Usage:
    python DFE_torch_eval.py --batch_size <batch_size> --window_size <window_size> --finetune <checkpoint_path> --data_path <dataset_path> [other_options]

Example:
    python DFE_torch_eval.py --batch_size 12000 --window_size 32 --finetune ../checkpoints/model_checkpoint.pth --data_path /path/to/val_data
    or you setup the config in the code first and run: python DFE_torch_eval.py

Author: Peter
"""

import os
import numpy as np
import torch
import torch.nn as nn    #torch version 2.0.1
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm.optim.optim_factory as optim_factory    #timm version 0.5.4
import argparse
from Dataset import Dataset_for_training
import sys
sys.path.append("..")
from utils.utils_Peter import *
from models.models import DFE_EfficientNet, DFE_Resnet34, DFE_Model, DFE_EfficientNet_vit_decoder, DFE_Swin_transformer, DFE_Swin_transformerv2, DFE_ConvNeXt
import random

# Set a seed for reproducibility
seed = 42

random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

# Function to parse command-line arguments
def get_args_parser():
    parser = argparse.ArgumentParser('Evalution of Cotracker-DFE', add_help=False)
    parser.add_argument('--batch_size', default=12000, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    
    parser.add_argument('--window_size', default=32, type=int,
                        help='input size for DFE model')
    
    # * Finetuning params
    parser.add_argument('--checkpoint', default='../checkpoints/model_finetune_efficientnet_vit_16_16_adamw_100_10_30_2048_97.pth',    #./models/model_877.pth
                        help='checkpoint path')
    # Dataset parameters
    parser.add_argument('--data_path', default='/peterstone/data/val_data_window_croped/validation/val_data_32_32/', type=str,
                        help='dataset path')    
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for testing')

    parser.add_argument('--num_workers', default=16, type=int,
                        help='num of workers in dataloader')

    parser.set_defaults('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU. \
                        But this will require more memory.')

    parser.set_defaults('--persistent_workers', default=True,
                        help='Accelerate data loading by preventing the need to repeatedly close and restart worker processes \
                        for each epoch. But this will require much more memory.')

    # distributed training parameters
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

args = get_args_parser()
args = args.parse_args()

# Set the device 
device = torch.device('cuda')

# Get the filenames and file paths for the test dataset
filenames=sorted_alphanumeric(os.listdir(args.data_path))
print("len(filenames): ", len(filenames))

file_paths = [os.path.join(args.data_path, file) for file in filenames]
total_size = len(file_paths)

print("test len : ", total_size)

# Create the test dataset and data loader
test_dataset = Dataset_for_training(file_paths, args.window_size, None)


print("Start loading data...")
val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, persistent_workers=args.persistent_workers)
print("Starting Setting up model...")

# Initialize the model and move it to the device
model = DFE_EfficientNet_vit_decoder().to(device)

# Load the pre-trained model weights and remove the `module.` prefix to make it compatible with the DFE model
if args.finetune:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

# Define the loss function
criterion = nn.MSELoss().to(device)

# Initialize variables for recording the loss
min_loss = float('inf')
final_epoch = 0
all_train_loss = []
all_val_loss = []

print("Start evaluation...")
model.eval()
total_loss = 0.0

# Iterate over the test dataset
for data_iter_step, (data, labels) in enumerate(val_loader):
    inputs = data.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    total_loss += loss.item()

# Calculate the average loss
average_loss = total_loss / len(val_loader)

print(f'Evaluation Loss: {average_loss}')
