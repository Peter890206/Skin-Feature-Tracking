import os
import torch
import torch.distributed as dist
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from cotracker.predictor import CoTrackerPredictor
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd


DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0, help='rank for distributed training')

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


args = parser.parse_args(args=[])
device = torch.device(f'cuda:{args.local_rank}')
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend="nccl", init_method="env://")
print("start synchronizing")
synchronize()
print("end synchronizing")

file_folder = "./assets"
file_names = os.listdir(file_folder)
# csv_file_name = "sticker_coords_for_cotracker.csv"
# csv_path = os.path.join(file_folder, csv_file_name)
# csv_data = pd.read_csv(csv_path)

model = CoTrackerPredictor(
            checkpoint=os.path.join(
                './checkpoints/cotracker_stride_4_wind_8.pth'
            )
        )
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


for file_name in file_names:
    start_time = time.time()
    if file_name.endswith("04941364_R_B_360_ground_truth.mp4"):
        video = read_video_from_path(os.path.join(file_folder, file_name))
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

        video = video.to(device)
        
        # row = csv_data[csv_data['filename'] == file_name]
        # if row['x2'].item() == -1 or row['y2'].item() == -1:
        #     queries = torch.zeros((1, 3)) 
        #     queries[:, 0] = 0.0
        #     queries[0, 1:] = torch.tensor([float(row['x1']), float(row['y1'])])
        # else:
        #     queries = torch.zeros((2, 3)) 
        #     queries[:, 0] = 0.0
        #     queries[0, 1:] = torch.tensor([float(row['x1']), float(row['y1'])])
        #     queries[1, 1:] = torch.tensor([float(row['x2']), float(row['y2'])])

        


        
        queries = torch.tensor([
            # [0., 220., 238.],  # hand mole
            # [0., 107., 223.],  # static face mole
            # [0., 160., 237.],  # static nose tip
            # [0., 105., 277.],  # bike face mole
            # [0., 159., 308.]   # bike nose tip
            # [0., 205., 169.], 
            # [0., 156., 187.],
            # [0., 178., 116.],    #04941364_R_B_handmole_ground_truth_coordinates_mean    5.908s
            # [0., 275., 141.],    #04941364_R_B_right_handmole_coordinates_mean
            # [0., 287., 195.],    #04941364_R_B_red_sticker_coordinates_mean
            [0., 71., 258.],    #04941364_R_B_little_finger_nail_coordinates_mean
            # frame number 10
        ])

        queries = queries.to(device)
        output_name = file_name


        print("start predicting")
        pred_tracks, pred_visibility = model(video, queries=queries[None])
        print("end predicting")

        vis = Visualizer(
            save_dir='./videos',
            linewidth=6,
            mode='cool',
            tracks_leave_trace=-1,
            name = output_name
        )
        vis.visualize(
            video=video,
            tracks=pred_tracks,
            visibility=pred_visibility,
            filename='queries');

        end_time = time.time()
        print(end_time - start_time)
        del video, queries, pred_tracks, pred_visibility
        torch.cuda.empty_cache()

