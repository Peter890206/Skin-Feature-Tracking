import moviepy
import pandas as pd
from moviepy.editor import *
from moviepy.video.fx.all import *

def pad_video(clip,  pos, size=(360, 360), color=(255, 255, 255)):
    padded = clip.on_color(size=size, color=color, pos=pos)
    return padded

folder_path = "C:/nordlinglab-digitalupdrs/Process_Video_Deep_Learning_Tracking_Peter/QuantPD_data/02636542"

csv_file_path = os.path.join(folder_path, "total_centers_for_cropping.csv")
data = pd.read_csv(csv_file_path)

output_folder = os.path.join(folder_path, "crop")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index, row in data.iterrows():
    filename = row['filename']
    x_center = row['x1']
    y_center = row['y1']
    
    video_path = os.path.join(folder_path, filename)
    video = VideoFileClip(video_path)
    
    output_width = 360
    output_height = 360
    x1 = x_center - output_width // 2
    y1 = y_center - output_height // 2

    if x1 + output_width > video.w:
        output_cropped = crop(video, x1=max(x1, 0), y1=max(y1, 0), width=min(360, video.w - max(x1, 0)), height=min(360, video.h - max(y1, 0)))
        output_cropped = pad_video(output_cropped, pos='right')
    elif y1 + output_height > video.h:
        output_cropped = crop(video, x1=max(x1, 0), y1=max(y1, 0), width=min(360, video.w - max(x1, 0)), height=min(360, video.h - max(y1, 0)))
        output_cropped = pad_video(output_cropped, pos='top')
    else:
        output_cropped = crop(video, x1=x1, y1=y1, width=360, height=360)

    output_path = os.path.join(output_folder, filename)
    output_cropped.write_videofile(output_path, codec='libx264', fps=video.fps, threads=4)
    print(f'Processed {filename}')



