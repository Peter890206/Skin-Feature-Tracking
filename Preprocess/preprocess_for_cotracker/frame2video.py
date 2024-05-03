import cv2
import os
import re


def sorted_alphanumeric(data):
    # Sorts filenames of in alphanumeric order
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

# 設定影片的寬度、高度和fps
width, height = 360, 360
fps = 10

# 設定影片的保存路徑和檔名
output_path = "04941364_R_B_360_ground_truth.mp4"

# 設定圖片資料夾的路徑
image_folder = "../dfe_ground_truth/04941364_R_B_360/"

# 取得圖片資料夾中的所有圖片檔案名稱
filenames=sorted_alphanumeric(os.listdir(image_folder ))

# 建立影片寫入器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 逐一讀取圖片並寫入影片
for image_file in filenames:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    video_writer.write(image)

# 釋放影片寫入器
video_writer.release()

print("finish")