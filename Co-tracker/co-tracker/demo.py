# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help='rank for distributed training')
    parser.add_argument(
        "--video_path",
        default="./assets/Finger_tapping_video.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/finger_tapping_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker_stride_4_wind_8.pth",
        help="cotracker model",
    )
    parser.add_argument("--grid_size", type=int, default=0, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame ",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    args = parser.parse_args(args=[])
    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    print("start synchronizing")
    synchronize()
    print("end synchronizing")

    # load the input video frame by frame
    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    segm_mask = torch.from_numpy(segm_mask)[None, None]

    model = CoTrackerPredictor(checkpoint=args.checkpoint)
    model = model.to(device)
    video = video.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    

    pred_tracks, pred_visibility = model(
        video,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        backward_tracking=args.backward_tracking,
        # segm_mask=segm_mask
    )
    print("computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame)
