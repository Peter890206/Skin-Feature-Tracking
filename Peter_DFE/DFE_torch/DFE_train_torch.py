"""
DFE Training Script

This script is used to train and fine-tune Deep Feature Exploration (DFE) models on image datasets. 
The DFE models are deep learning architectures designed for image classification tasks. 
The script supports various DFE model variants, such as DFE_EfficientNet, DFE_Resnet34, DFE_Swin_transformer, and more.

The script allows for configuring various training parameters, including batch size, number of epochs, learning rate,
weight decay, and data augmentation techniques like Gaussian noise, Gaussian blur, and random erasing.

During training, the script splits the dataset into training and validation sets, applies data augmentation to the training set,
and trains the specified DFE model using distributed training. 
It also saves the trained model weights and loss curves for both training and validation sets.

Usage:
    python -m torch.distributed.run --nproc_per_node=<num_gpus> DFE_train_torch.py --(configuration you want to change)

Example:
    python -m torch.distributed.run --nproc_per_node=2 DFE_train_torch.py

Note:
    Make sure you are in the correct environment.

Author: Peter
"""

import os
import numpy as np
import torch
import torch.nn as nn    #torch version 2.0.1
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import timm.optim.optim_factory as optim_factory    #timm version 0.5.4

import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append("..")
from models.models import DFE_EfficientNet, DFE_Resnet34, DFE_Model, DFE_EfficientNet_vit_decoder, DFE_Swin_transformerv2, DFE_Swin_transformer, DFE_ConvNeXt, DFE_Resnet50, DFE_EfficientNetV2
from utils.utils_Peter import *
from Dataset import Dataset_for_training

import lr_sched
from torch.utils.data import random_split
import random



# Set a seed for reproducibility (to ensure consistent results across runs)
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
    parser = argparse.ArgumentParser('Pretraining and Finetuning of DFE using different backbones', add_help=False)
    parser.add_argument('--batch_size', default=2048, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=50, type=int, 
                        help='pretrain = 600, finetune = 50')   
    
    parser.add_argument('--window_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',    #pretrain 1.6e-2, finetune 8e-4
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',    #pretrain 1e-3, finetune 5e-5
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,    #0.05
                        help='weight decay (default: 0.05)')

    parser.add_argument('--min_lr', type=float, default=5e-9, metavar='LR',    # 5e-9
                        help='lower lr bound for cyclic schedulers that hit 5e-9')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',    #
                        help='epochs to warmup LR, pretrain = 40 and finetune = 5')
    # * Finetuning params
    parser.add_argument('--finetune', default='../checkpoints/model_pretrain_efficientnetv2_cnn_adamw_100_10_30_2048_592.pth',    #./models/model_efficientnet_pretrain_adamw_0_0_100_15_35_10_30_2048_596.pth
                        help='finetune from checkpoint')
    # Dataset parameters
    parser.add_argument('--data_path', default='/peterstone/data/hand11k_data_window_croped/train/hand1M_data_resized_64_window_croped/', type=str,
                        help='dataset path, pretrain: Imagenet_data_resized_64_JPEG/train/0/, \
                        finetune: hand11k_data_window_croped/train/hand1M_data_resized_64_window_croped/')    
    
    parser.add_argument('--file_name', default='efficientnetv2_cnn_adamw_30_30_100_10_30_2048', type=str,
                        help='file name for saving, the rule is : <backbone>_<decoder>_<optimizer>_<percentage of Gaussian noise> \
                            _<percentage of Gaussian blur>_<percentage of Random Erasing>_<range of Random Erasing(min and max)> \
                            _<batch size>')

    parser.add_argument('--output_dir', default='../checkpoints',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')

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

# Initialize distributed training
dist.init_process_group("nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()
device_id = rank % torch.cuda.device_count()
device = torch.device(device_id)

# Get the file paths for the dataset
filenames=sorted_alphanumeric(os.listdir(args.data_path))
print("len(filenames): ", len(filenames))

file_paths = [os.path.join(args.data_path, file) for file in filenames]
total_size = len(file_paths)

# Split the dataset into training and validation sets
val_size = int(0.15 * total_size)
train_size = total_size - val_size
train_paths, val_paths = random_split(file_paths, [train_size, val_size])
print("train len : ", len(train_paths))
print("val len : ", len(val_paths))

# Define the data augmentation transformations for the training set
train_transform = transforms.Compose([
            transforms.RandomApply([gauss_noise_tensor], p = 0.3),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.2, 1.5))], p = 0.3), 
            transforms.RandomErasing(p = 1.0, scale = (0.1, 0.3)),
            ])

# Create the training and validation datasets
train_dataset = Dataset_for_training(train_paths, args.window_size, train_transform)    #train_transform
val_dataset = Dataset_for_training(val_paths, args.window_size, None)

# Print the dataset length and start loading data
print("len(dataset): ", len(train_dataset))
print("Start loading data...")

# Create data loaders for training and validation sets using DistributedSampler for distributed training
train_sampler = DistributedSampler(train_dataset)
data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=args.pin_mem, persistent_workers=args.persistent_workers)
val_sample = DistributedSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sample, num_workers=args.num_workers, pin_memory=args.pin_mem, persistent_workers=args.persistent_workers)
print("Starting Setting up model...")

# Initialize the DFE model and move it to the device
model = DFE_EfficientNetV2().to(device)

# Set up distributed data parallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model = DistributedDataParallel(model, find_unused_parameters=True)
    model_without_ddp = model.module
else:
    model_without_ddp = model

# Load pre-trained model weights if specified
if args.finetune:
    model.load_state_dict(torch.load(args.finetune, map_location='cpu'), strict=False)

# Define the loss function (Mean Squared Error in this case)
criterion = nn.MSELoss().to(device)

# Set up the optimizer and learning rate 
param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
if args.lr is None:  # only base_lr is specified
        # args.lr = args.blr * args.batch_size * 2 * 4 / 256
        args.lr = args.blr * args.batch_size * 2 / 256
optimizer = optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
# optimizer = optim.Adam(param_groups, lr=args.lr)
# optimizer = optim.Adamax(param_groups, lr=args.lr)

# Initialize variables for recording the loss
min_loss = float('inf')
final_epoch = 0
all_train_loss = []
all_val_loss = []

# Training loop
print("Start training...")
for epoch in tqdm(range(args.epochs)):
    # Set the model to training mode
    model.train()
    total_loss = 0.0
    # Iterate over the training dataset
    for data_iter_step, (data, labels) in enumerate(data_loader):
        # Adjust the learning rate
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move the data and labels to the device
        inputs = data.to(device)
        labels = labels.to(device)
        # inputs = data.view(-1, 3, args.window_size, args.window_size)    # make sure the dimensions match
        # inputs = inputs.to(device)

        # Forward pass through the model
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Average gradients across processes
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= args.world_size

        optimizer.step()

        # Print the loss and learning rate (for every 20 steps)
        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % 20 == 0 and rank == 0:  
            print(f'Epoch {epoch}, Step {data_iter_step}/{len(data_loader)}, Loss: {loss.item()}, lr: {lr}')

        # Accumulate the loss
        total_loss += loss.item()
    
    # Evaluation on the validation set
    model.eval()
    val_total_loss = 0.0
    print("Start validation...")

    # Iterate over the validation dataset
    for data_iter_step, (data, labels) in enumerate(val_loader):
        # print("data_iter_step: ", data_iter_step)
        inputs = data.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                outputs = model.module(inputs)
            else:
                outputs = model(inputs)
            val_loss = criterion(outputs, labels)
        
        # Accumulate and average the validation loss across processes
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_total_loss += val_loss.item()

    print("finish validation")
    average_val_loss = val_total_loss / len(val_loader)
    
    average_val_loss /= args.world_size

    # Save the model with the minimum validation loss
    if rank == 0:
        average_epoch_loss = total_loss / len(data_loader)
        if average_val_loss < min_loss:
            min_loss = average_val_loss
            final_epoch = epoch
            torch.save(model.state_dict(), f'{args.output_dir}/model.pth')
            print("Min loss: ", min_loss)
        
        # Print and store the training and validation losses
        print(f'Epoch {epoch}, Train Loss: {average_epoch_loss}')
        print(f'Epoch {epoch}, Val Loss: {average_val_loss}')
        all_train_loss.append(average_epoch_loss)
        all_val_loss.append(average_val_loss)

# Save the loss curves and rename the saved model file
if rank == 0:
    if args.finetune:
        np.save(f'../Process_Video_DfeTracking/training_loss_npy/train_loss_{args.file_name}_{final_epoch}_finetune.npy', all_train_loss)
        np.save(f'../Process_Video_DfeTracking/training_loss_npy/val_loss_{args.file_name}_{final_epoch}_finetune.npy', all_val_loss)
    else:
        np.save(f'../Process_Video_DfeTracking/training_loss_npy/train_loss_{args.file_name}_{final_epoch}_pretrain.npy', all_train_loss)
        np.save(f'../Process_Video_DfeTracking/training_loss_npy/val_loss_{args.file_name}_{final_epoch}_pretrain.npy', all_val_loss)
    if os.path.exists(f'{args.output_dir}/model.pth'):
        if args.finetune:
            os.rename(f'{args.output_dir}/model.pth', f'{args.output_dir}/model_finetune_{args.file_name}_{final_epoch}.pth')
        else:
            os.rename(f'{args.output_dir}/model.pth', f'{args.output_dir}/model_pretrain_{args.file_name}_{final_epoch}.pth')