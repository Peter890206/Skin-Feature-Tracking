import cv2
import pandas as pd
import os
import re

def sorted_alphanumeric(data):
    # Sorts filenames of in alphanumeric order
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

folder_path = "C:/nordlinglab-digitalupdrs/Process_Video_Deep_Learning_Tracking_Peter/QuantPD_data/02636542"
output_name = 'total_centers_for_cropping.csv'   # 'total_centers_for_cropping.csv', 'sticker_coords_for_cotracker.csv'
image_filenames = [f for f in sorted_alphanumeric(os.listdir(folder_path)) if f.endswith('.mp4')]
# image_filenames = sorted_alphanumeric(os.listdir(folder_path))
# path = "../QuantPD_data/19830624_20220214_Hands_L_R_360_360.mp4"
centers = []
for i, filename in enumerate(image_filenames):
# for i in range(1):
    # if i < 54:
    #     continue
    # print(path)
    file_path = os.path.join(folder_path, filename)
    cap = cv2.VideoCapture(file_path)
    ret, first_frame = cap.read()
    orgimg_shape = first_frame.shape
    ImgCopy = first_frame.copy()
    window_size=(31,31)
    initial_point = (int(orgimg_shape[0]/2), int(orgimg_shape[1]/2))
    x1 = int(initial_point[0]-window_size[0]/2)
    x2 = int(initial_point[0]+window_size[0]/2)
    y1 = int(initial_point[1]-window_size[1]/2)
    y2 = int(initial_point[1]+window_size[1]/2)
    stride = 1
    frame_idx = 0
    rect_list = []
    center_set = []
    key = cv2.waitKey(0)
    while True:
        clone = ImgCopy.copy()
        if key == ord('w'):
            y1 -= stride
            y2 -= stride
        if key == ord('s'):
            y1 += stride
            y2 += stride
        if key == ord('a'):
            x1 -= stride
            x2 -= stride
        if key == ord('d'):
            x1 += stride
            x2 += stride
        if key == ord('z'):
            if stride == 1:
                stride = 10
            else:
                stride = 1

        if key == ord('r'):  # refresh img
            rect_list = []
            center_set = []
        if key == 13:
            print("Select center at: " +
                    str([int(x1+window_size[0]/2), int(y1+window_size[0]/2)]))
            print("x1, y1, x2, y2: " + str([x1, y1, x2, y2]))
            center = [int(x1+window_size[0]/2), int(y1+window_size[0]/2)]
            rect_list.append([x1, y1, x2, y2])
            center_set.append(
                [int(x1+window_size[0]/2), int(y1+window_size[0]/2)])
            
    

        for i in range(len(rect_list)):
            coor = rect_list[i]
            cv2.rectangle(clone, (coor[0], coor[1]),
                            (coor[2], coor[3]), (0, 255, 0), 2)
            cv2.circle(clone, (int(coor[0]+window_size[0]/2),
                        int(coor[1]+window_size[1]/2)), 3, (0, 255, 0), 2)

        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(
            clone, (int(x1+window_size[0]/2), int(y1+window_size[1]/2)), 3, (0, 0, 255), 2)
        cv2.putText(clone, "stride: " + str(stride), (10, 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Select thumb first after that select index finger", clone)
        key = cv2.waitKey(0)
        if key == ord('q'):
            centers.append(center_set)
            clone = ImgCopy.copy()
            for i in range(len(rect_list)):
                coor = rect_list[i]
            
                cv2.rectangle(clone, (coor[0], coor[1]),
                                (coor[2], coor[3]), (0,0 , 255), 2)
                # cv2.putText(clone,"Point "+str(i+1), (coor[0], coor[1]),cv2.FONT_HERSHEY_COMPLEX,0.6, (0,0,0), 2)
                # cv2.circle(clone, (int(
                #     coor[0]+window_size[0]/2), int(coor[1]+window_size[1]/2)), 3, (0, 255, 0), 2)

                # img_save_path = os.path.dirname(path[:-1])
                # cv2.imwrite(img_save_path + f"/{initial_point[0]}.jpg", clone)
            # img_save_path = os.path.dirname(path)
            # cv2.imwrite(img_save_path + f"_targets.jpg", clone)

            cv2.destroyAllWindows()
            break
print("total_point =", centers)
data = []
for i, coords in enumerate(centers):
    if len(coords) == 2:
        # 如果有两个点，正常处理
        data.append([image_filenames[i], coords[0][0], coords[0][1], coords[1][0], coords[1][1]])
    elif len(coords) == 1:
        # 如果只有一个点，第二个点的坐标设为-1
        data.append([image_filenames[i], coords[0][0], coords[0][1], -1, -1])

df = pd.DataFrame(data, columns=['filename', 'x1', 'y1', 'x2', 'y2'])
df.to_csv(os.path.join(folder_path, output_name), index=False)
# np.save("./19830624_20220214_Hands_L_R_center.npy", centers)
