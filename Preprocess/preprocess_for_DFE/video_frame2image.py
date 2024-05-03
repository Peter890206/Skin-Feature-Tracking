# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:59:25 2023

@author: peter
"""

import cv2
import os
# print(cv2.getBuildInformation())

first_frame = 0
total_frame = 500
frame_interval = 1
video_path = "../04941364_R_B_360_ground_truth.mp4"
output_dirname = "../data/frame_for_DFE/"

cap = cv2.VideoCapture(video_path)
isOpened = cap.isOpened()
print("Success load video:", isOpened)
original_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total input video frame:", original_video_frames)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)

i = 0
num = 0
frame_num = 0

while(isOpened):
      flag, frame = cap.read()
      # print(flag)
      if num == first_frame:
            if frame_num % frame_interval == 0:
                
                  if i == total_frame or i + 1 > original_video_frames:
                        break
                  else:
                        i += 1
                  if i < 10:
                        name = "0000" + str(i)
                  elif i < 100:
                        name = "000" + str(i)
                  elif i < 1000:
                        name = "00" + str(i)
                  elif i < 10000:
                        name = "0" + str(i)
                  else:
                        name = str(i)
                  fileName = name + ".png"
                  
                  # print(os.path.join(output_dirname,fileName))
      
                  cv2.imwrite(os.path.join(output_dirname,fileName), frame, [cv2.IMWRITE_JPEG_QUALITY, total_frame])
            frame_num += 1
            num -= 1
      num += 1
        