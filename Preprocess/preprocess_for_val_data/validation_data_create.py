import cv2

# 初始化變量
cap = cv2.VideoCapture('../../QuantPD_data/04941364/04941364_20240403_Hands_R_B_B.mp4')  # 替換成您的影片路徑
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
output_size = 360
interval = 10    # if you can change this number to generate val data with different interval
x, y = 600, 360  # the initial position of the red marker

def on_trackbar(val):
    global current_frame
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        displayed_frame = process_frame(frame.copy(), x, y)
        cv2.imshow('Video', displayed_frame)


def process_frame(frame, x, y):
    
    cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    return frame

def crop_and_save_image(frame, x, y, frame_number, output_size=360):
   
    height, width = frame.shape[:2]
    crop_width = output_size//2
    start_x = max(x - crop_width, 0)
    start_y = max(y - crop_width, 0)
    end_x = min(x + crop_width, width)
    end_y = min(y + crop_width, height)
    cropped = frame[start_y:end_y, start_x:end_x]
    # change your output paht here
    cv2.imwrite(f'../data/val_data_before_crop/cropped_image_04941364_R_B_frame{frame_number}.png', cropped)

cv2.namedWindow('Video')
cv2.createTrackbar('Frame', 'Video', 0, total_frames-1, on_trackbar)

stride = 1
while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        break  # 如果無法讀取幀，退出循環

    displayed_frame = process_frame(frame.copy(), x, y)
    cv2.imshow('Video', displayed_frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):  # 按q退出
        break
    elif key == ord('w'):  # 向上移動
        y = max(y - stride, 0)
    elif key == ord('s'):  # 向下移動
        y = min(y + stride, frame.shape[0] - stride)
    elif key == ord('a'):  # 向左移動
        x = max(x - stride, 0)
    elif key == ord('d'):  # 向右移動
        x = min(x + stride, frame.shape[1] - stride)
    elif key == ord('z'):
            if stride == 1:
                stride = 10
            else:
                stride = 1

    elif key == ord('j'):  # 代替左箭头 - 倒退1帧
        current_frame = max(current_frame - 1, 0)
    elif key == ord('l'):  # 代替右箭头 - 快进1帧
        current_frame = min(current_frame + 1, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    elif key == ord('i'):  # 代替上箭头 - 快进10帧
        current_frame = min(current_frame + interval, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    elif key == ord('k'):  # 代替下箭头 - 倒退10帧
        current_frame = max(current_frame - interval, 0)
    elif key == ord('\r'):  # Enter鍵，裁剪並保存圖像
        crop_and_save_image(frame, x, y, current_frame, output_size)
cap.release()
cv2.destroyAllWindows()