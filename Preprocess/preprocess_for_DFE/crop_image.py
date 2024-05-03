import cv2
import os
import numpy as np

"""
The convertRGB2CIELab function is for showing the CIELAB images.
If you just want to crop the image for Cotracker-DFE, please ignore it.
"""
def convertRGB2CIELab(image):
    # Converts uint8 RGB image to float32 CIELAB format
    image = np.array(image).astype('float32')
    if np.max(image) > 1:
        image *= 1/255
    Lab_image = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
    Lab_image[:,:,0]=Lab_image[:,:,0]/100 # Normalize L to be betweeen 0 and 1
    Lab_image[:,:,1]=(Lab_image[:,:,1]+127)/(2*127)
    Lab_image[:,:,2]=(Lab_image[:,:,2]+127)/(2*127)
    return Lab_image

point1_corrds = np.load('../../QuantPD_data/cotracker_pred/04941364_R_B_360_little_finger_nail_cotracker_pred_coords.npy')
# point2_corrds = np.load('../static_nosetip_cotracker_predict_coords_300_420.npy')
# print(point1_corrds[0])

images_dir = "../../dfe_ground_truth/04941364_R_B_360"
for width in range(8, 9, 2):
    i = 0
    w = 16 + width
    output_dir = f"../data/04941364_R_B_little_finger_nail_crop_image_handmole_{w}_{w}_avg"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image_name in os.listdir(images_dir):
        print(i)
        image_dir = os.path.join(images_dir, image_name)
        image = cv2.imread(image_dir)
        # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # image=convertRGB2CIELab(image)
        point1_coord_x = point1_corrds[i][0]
        point1_coord_y = point1_corrds[i][1]
        # point2_coord_x = point2_corrds[i][0]
        # point2_coord_y = point2_corrds[i][1]

        
        # w_test = 16 + 22
        crop_image1 = image[point1_coord_y-w:point1_coord_y+w+1, point1_coord_x-w:point1_coord_x+w+1]
        # crop_image_test = image[point1_coord_y-w_test:point1_coord_y+w_test+1, point1_coord_x-w_test:point1_coord_x+w_test+1]   
        # crop_image2 = image[point2_coord_y-w:point2_coord_y+w+1, point2_coord_x-w:point2_coord_x+w+1]
        
        cv2.imwrite(f"{output_dir}/crop_image_{i}_avg.png", crop_image1)
        # cv2.imwrite(f"./crop_image2_32/crop_image_{i}.jpg", crop_image2)
        i += 1
        cv2.waitKey(100)

