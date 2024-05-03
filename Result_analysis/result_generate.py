import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

plt.style.use(r"./rw_visualization.mplstyle")


def load_data(path):
    return np.load(path)

def extract_coordinates(data):
    return data[:, 0], data[:, 1]

def calculate_distance(coord1_x, coord1_y, ground_truth_x, ground_truth_y):
    return np.sqrt((coord1_x - ground_truth_x)**2 + (coord1_y - ground_truth_y)**2)

def revise_cotracker_distance(coord1_x, coord1_y, cotracker_x, cotracker_y, center_width):   # center_width = (width of window predicted by cotracker - 1) / 2
    return cotracker_x - center_width + coord1_x, cotracker_y - center_width + coord1_y
paths = {
    "ground_truth_point": "../dfe_ground_truth/mean_ground_truth/04941364_R_B_handmole_ground_truth_coordinates_mean.npy",    #../Peter_handmole_ground_truth_mean.npy, 04941364_R_B_ground_truth_coordinates_mean.npy
    "Jose_ground_truth_point": "../dfe_ground_truth/bbox_centers_pd.npy",
    "cotracker_point": "../QuantPD_data/cotracker_pred/04941364_R_B_cotracker_pred_178_116.npy",    #Handmole_cotracker_pred_220_238.npy, 04941364_R_B_cotracker_pred_178_116.npy
    # "dfe_revised_point_24": "../handmole_DFE_tf_revised_40_subpixel_24_24_preds.npy",
    # "dfe_revised_point_20": "../handmole_DFE_tf_revised_40_subpixel_optimized_rank_1ref_20_20.npy",
    # "dfe_revised_point_30": "../handmole_DFE_tf_revised_40_subpixel_optimized_rank_1ref_30_30.npy",
    # "mixmae_revised_point": "../handmole_efficientnet_vit_16_16_adamw_100_10_30_2048_97_finetune_preds.npy",
    # "dfe_point": "../handmole_DFE_tf_1ref_40_subpixel_220_238_preds.npy",
    "optical_flow_point": "../QuantPD_data/handmole_optical_flow_pred_220_238.npy",
    "optical_flow_point2": "../QuantPD_data/L_BPredict_point1_Crop(0, 0)_dots[[220, 238]].npy",
    # "mixmae_point": "../handmole_DFE_mixmae_40_32_pixel_level_CIELAB_0_0008_3ref_area1_220_239_preds.npy",
    # "pips++_point": "../Handmole_predition_pips++.npy",
}

data = {key: load_data(value) for key, value in paths.items()}
# data["pips++_point"] = np.squeeze(data["pips++_point"])
# print(data["pips++_point"].shape)
coordinates = {key: extract_coordinates(value) for key, value in data.items()}

optical_flow_point_list = list(coordinates["optical_flow_point"])
optical_flow_point_list[0] = np.insert(np.array(optical_flow_point_list[0]), 0, 220)
optical_flow_point_list[1] = np.insert(np.array(optical_flow_point_list[1]), 0, 238)
optical_flow_point_list2 = list(coordinates["optical_flow_point2"])
optical_flow_point_list2[0] = np.insert(np.array(optical_flow_point_list2[0]), 0, 220)
optical_flow_point_list2[1] = np.insert(np.array(optical_flow_point_list2[1]), 0, 238)
coordinates["optical_flow_point"] = tuple(optical_flow_point_list)
coordinates["optical_flow_point2"] = tuple(optical_flow_point_list2)
print("x diff: ", np.sum(coordinates["optical_flow_point"][0] - coordinates["optical_flow_point2"][0]))
print("y diff: ", np.sum(coordinates["optical_flow_point"][1] - coordinates["optical_flow_point2"][1]))

# coordinates["ground_truth_point"][0][0] = int(coordinates["ground_truth_point"][0][0])
# coordinates["ground_truth_point"][1][0] = int(coordinates["ground_truth_point"][1][0])
# dfe_revised_24_x, dfe_revised_24_y = revise_cotracker_distance(coordinates["dfe_revised_point_24"][0], coordinates["dfe_revised_point_24"][1], coordinates["cotracker_point"][0], coordinates["cotracker_point"][1], 24)
# def_revised_20_x, def_revised_20_y = revise_cotracker_distance(coordinates["dfe_revised_point_20"][0], coordinates["dfe_revised_point_20"][1], coordinates["cotracker_point"][0], coordinates["cotracker_point"][1], 20)
# dfe_revised_30_x, dfe_revised_30_y = revise_cotracker_distance(coordinates["dfe_revised_point_30"][0], coordinates["dfe_revised_point_30"][1], coordinates["cotracker_point"][0], coordinates["cotracker_point"][1], 30)
# mixmae_revised_x, mixmae_revised_y = revise_cotracker_distance(coordinates["mixmae_revised_point"][0], coordinates["mixmae_revised_point"][1], coordinates["cotracker_point"][0], coordinates["cotracker_point"][1], 24)
distances = {}

distances["cotracker_distances"] = calculate_distance(coordinates["cotracker_point"][0], coordinates["cotracker_point"][1], coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])
# distances["dfe_revised_24_distances"] = calculate_distance(dfe_revised_24_x, dfe_revised_24_y, coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])
# distances["dfe_revised_20_distances"] = calculate_distance(def_revised_20_x, def_revised_20_y, coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])
# distances["dfe_revised_30_distances"] = calculate_distance(dfe_revised_30_x, dfe_revised_30_y, coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])
# distances["dfe_distances"] = calculate_distance(coordinates["dfe_point"][0], coordinates["dfe_point"][1], coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])
# distances["mixmae_revised_distances"] = calculate_distance(mixmae_revised_x, mixmae_revised_y, coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])
# distances["mixmae_distances"] = calculate_distance(coordinates["mixmae_point"][0], coordinates["mixmae_point"][1], coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])
# distances["pips++_distances"] = calculate_distance(coordinates["pips++_point"][0], coordinates["pips++_point"][1], coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])
# distances["optical_flow_distances"] = calculate_distance(coordinates["optical_flow_point"][0], coordinates["optical_flow_point"][1], coordinates["ground_truth_point"][0], coordinates["ground_truth_point"][1])


sorted_distances = {key: np.sort(value)[::-1] for key, value in distances.items()}

# cotracker_revised_subpixel_distances = np.fft.fft(cotracker_revised_subpixel_distances)
# cotracker_revised_pixel_distances = np.fft.fft(cotracker_revised_pixel_distances)

# sampling_rate = 1  # 取樣率，假設為1
# n = len(cotracker_revised_subpixel_distances)  # 資料點數量
# freq = np.fft.fftfreq(n, d=1/sampling_rate)

plt.figure(figsize=(5, 6))
# # plt.plot(DFE_distances, '-', label='DFE distances')
# plt.plot(cotracker_point1_x, '-', label='Cotracker point1')
# plt.plot(cotracker_point2_x, '-', label='Cotracker point2')
# plt.plot(DFE_point1_x, '--', label='DFE point1')
# plt.plot(DFE_point2_x, '--', label='DFE point2')
plt.plot(distances["cotracker_distances"], 'o-', label='Cotracker')
# plt.plot(distances["dfe_revised_24_distances"], 's-', label='Cotracker-DFE')

# plt.plot(distances["dfe_revised_20_distances"], 's-', label='Cotracker-DFE-20')
# plt.plot(distances["dfe_revised_30_distances"], 's-', label='Cotracker-DFE-30')

# plt.plot(distances["dfe_distances"], '^-', label='DFE')

# plt.plot(point1_revised_mixmae_distances, 'D-', label='Cotracker-mixmae')
# plt.plot(point1_distance_optical_flow, ':', label='Point1 distances Optical Flow')
# plt.plot(point1_dfe_mixmae_distances, 'v-', label='DFE-mixmae')
# plt.plot(distances["pips++_distances"], 'v-', label='Pips++')
plt.legend(loc='best', fontsize=30)
plt.xlim(0, 39)
num_points = 25
# x = np.arange(0, len(point1_distances))
# for i in range(0, len(x), len(x)//num_points):
#     plt.scatter(x[i], point1_distances[i], marker='o', s=100, color='C0')
#     plt.scatter(x[i], point1_revised_distances[i], marker='s', s=100, color='C1')
#     plt.scatter(x[i], point1_distances_dfe[i], marker='^', s=100, color='C2')
#     plt.scatter(x[i], point1_revised_mixmae_distances[i], marker='D', s=100, color='C3')

# 添加標籤和標題
plt.xlabel('Frame', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.ylabel('distance between two points (pixel)')
# plt.ylabel('distance between X (pixel)')
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
# plt.title('Absolute distance error (pixels)', fontsize=36)

print("Total_distance_cotracker_point1: ", np.sum(distances["cotracker_distances"]))
# print("Total_distance_cotracker_revised_point1: ", np.sum(distances["dfe_revised_24_distances"]))
# print("Total_distance_DFE_point1: ", np.sum(distances["dfe_distances"]))
# print("Total_distance_mixmae_revised_point1: ", np.sum(distances["mixmae_revised_distances"]))
# print("Total_distance_mixmae_point1: ", np.sum(distances["mixmae_distances"]))
# print("Total_distance_optical_flow_point1: ", np.sum(distances["optical_flow_distances"]))
# print("Total_distance_pips++: ", np.sum(distances["pips++_distances"]))

# 顯示圖表
# plt.figure(2)
# plt.hist(point1_distances, bins=100, alpha=0.5, label='Point1 distances Cotracker')
# plt.hist(point1_revised_distances, bins=100, alpha=0.5, label='Point1 revised Cotracker distances')
# plt.hist(point1_distances_dfe, bins=100, alpha=0.5, label='Point1 distances DFE'
# # plt.plot(cotracker_distances, '-', label='Cotracker distances')
# # plt.plot(cotracker_point1_y, '-', label='Cotracker point1')
# # plt.plot(cotracker_point2_y, '-', label='Cotracker point2')
# # plt.plot(DFE_point1_y, '--', label='DFE point1')
# # plt.plot(DFE_point2_y, '--', label='DFE point2')
# plt.plot(point2_distances, '-', label='Point2 distances Cotracker')
# plt.plot(point2_revised_distances, '--', label='Point2 revised Cotracker distances')
# plt.plot(point2_distances_dfe, ':', label='Point2 distances DFE')
# plt.plot(point2_distance_optical_flow, ':', label='Point2 distances Optical Flow')
# plt.plot(point2_distance_mixmae_revised, ':', label='Point2 distances mixmae revised')


# plt.legend(loc='best')

# # 添加標籤和標題
# plt.xlabel('Frame')
# # plt.ylabel('distance between two points (pixel)')
# # plt.ylabel('distance between Y (pixel)')
# plt.ylabel('Absulate distance of point2 between prediction and Ground truth (pixel)')
# plt.title('distances')

# print(np.abs(cotracker_revised_pixel_distances))
# print(freq)
# plt.figure(10)
# plt.plot(ssr_point1, '-', label='SSR point1')
# plt.hist(distances_between_pixel_subpixel, bins=100, label='distances_between_pixel_subpixel')
# plt.hist(dfe_point2_y, bins=range(220, 250), alpha=0.5, label='DFE point2')

# plt.plot(freq, np.abs(cotracker_revised_pixel_distances), '-', label='cotracker_revised_pixel_distances')
# plt.plot(freq, np.abs(cotracker_revised_subpixel_distances), '--', label='cotracker_revised_subpixel_distances')
# # # plt.plot(cotracker_point2_y, '-', label='Cotracker point2')
# # # plt.plot(DFE_point1_y, '--', label='DFE point1')
# # # plt.plot(DFE_point2_y, '--', label='DFE point2')
# plt.legend(loc='best')
# plt.xlabel('Frame')
# plt.ylabel('ssr')
# # plt.ylabel('distance between two points (pixel)')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.xlim([0, 0.5])
# plt.legend(loc='best')
# plt.figure(figsize=(5, 6))
x = np.arange(0, len(sorted_distances["cotracker_distances"]))
plt.plot(x, sorted_distances["cotracker_distances"], 'o-', label='Cotracker')
# plt.plot(x, sorted_distances["dfe_revised_24_distances"], 's-', label='Cotracker-DFE')
# plt.plot(x, sorted_distances["dfe_distances"], '^-', label='DFE')
# plt.plot(x, sorted_distances["mixmae_revised_distances"], 'D-', label='Cotracker-mixmae')
# plt.yscale('log')  # 使用對數刻度
# plt.ylim(0.1, None)
plt.xlim(0, 39)
plt.xlabel('Number of images with an error larger than the value', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.title('Sorted Values with absolute distance error (pixels)', fontsize=36)
plt.legend(loc = 'best', fontsize=30)
num_points = 15
# for i in range(0, len(x), len(x)//num_points):
    # plt.scatter(x[i], sorted_point1_distances[i], marker='o', s=100, color='C0')
    # plt.scatter(x[i], sorted_point1_revised_distances[i], marker='s', s=100, color='C1')
    # plt.scatter(x[i], sorted_point1_distances_dfe[i], marker='^', s=100, color='C2')
    # plt.scatter(x[i], sorted_point1_revised_mixmae_distances[i], marker='D', s=100, color='C3')

# plt.show()



# 顯示圖表
# plt.show()

"""
plot training an validation loss
"""



# error_frame_path = "../handmole_DFE_tf_revised_40_subpixel_40_40_test_error_frames.npy"
# error_frame = np.load(error_frame_path)
# ssr_path = "../handmole_DFE_tf_revised_40_pixel_34_34_test_ssr.npy"
# ssr = np.load(ssr_path)
# all_coords_path = "../handmole_DFE_tf_revised_40_pixel_34_34_test_all_coords.npy"
# all_coords = np.load(all_coords_path)
# # print(error_frame)
# print("all_coords", all_coords[2])
# print("ssr", ssr[2][0:3])


# width_for_window = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]    #window size = (16 + width) * 2 + 1
# total_error_pixel = [31.7858, 25.8929, 42.5432, 40.2579, 45.6076, 45.6076, 36.0284, 48.6076, 30.7858, 26.1289, 42.1289]
# total_error_subpixel = [35.7270, 49.0791, 30.1408, 51.0590, 44.4551, 61.6590, 41.1224, 59.2892, 73.2503, 57.4733, 39.3044]    #start from 41.1224 have a larger peak nearby 33st frame
# time_pixel = [30.88, 30.37, 32.62, 33.70, 31.79, 30.53, 32.04, 32.20, 32.56, 34.15, 33.95]
# time_subpixel = [32.21, 32.81, 34.74, 34.44, 37.71, 36.04, 36.76, 37.16, 37.19, 41.16, 38.79]

# plt.figure()
# plt.plot(width_for_window, total_error_pixel, 'o-', label='total_error_pixel')
# plt.plot(width_for_window, total_error_subpixel, 's-', label='total_error_subpixel')
# plt.xlim(4, 24)
# plt.xlabel('width for window size (pixel)', fontsize=44)
# plt.ylabel('total error (pixel)', fontsize=44)
# plt.legend()

# plt.figure()
# plt.plot(width_for_window, time_pixel, 'o-', label='time_pixel')
# plt.plot(width_for_window, time_subpixel, 's-', label='time_subpixel')
# plt.xlabel('width for window size (pixel)', fontsize=44)
# plt.ylabel('time (s)', fontsize=44)
# plt.xlim(4, 24)
# plt.legend()
# plt.show()

# distance_between_cotracker_ground_truth_x = coordinates["ground_truth_point"][0] - coordinates["cotracker_point"][0]
# distance_between_cotracker_ground_truth_y = coordinates["ground_truth_point"][1] - coordinates["cotracker_point"][1]
# print("cotracker_x_distances", distance_between_cotracker_ground_truth_x)
# print("cotracker_y_distances", distance_between_cotracker_ground_truth_y)

"""
plot training and validation loss
"""
loss_paths = {
    "EfficientNet_Adamw_70_70_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_70_70_15_35_2048.npy",
    "EfficientNet_Adamw_70_30_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_70_30_15_35_2048.npy",
    "EfficientNet_Adamw_70_100_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_70_100_15_35_2048.npy",
    "EfficientNet_Adamw_70_0_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_70_0_15_35_2048.npy",
    "EfficientNet_Adamw_50_50_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_50_50_15_35_2048.npy",
    "EfficientNet_Adamw_30_70_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_30_70_15_35_2048.npy",
    "EfficientNet_Adamw_30_30_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_30_30_15_35_2048.npy",
    "EfficientNet_Adamw_30_100_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_30_100_15_35_2048.npy",
    "EfficientNet_Adamw_30_0_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_30_0_15_35_2048.npy",
    "EfficientNet_Adamw_100_30_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_100_30_15_35_2048.npy",
    "EfficientNet_Adamw_100_0_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_100_0_15_35_2048.npy",
    "EfficientNet_Adamw_0_70_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_0_70_15_35_2048.npy",
    "EfficientNet_Adamw_0_30_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_0_30_15_35_2048.npy",
    "EfficientNet_Adamw_0_100_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_0_100_15_35_2048.npy",
    "EfficientNet_Adamw_0_0_0": "../Peter_DFE/training_loss_npy/val_loss_adamw_0_0_15_35_2048.npy",
    "EfficientNet_Adamw_0_0_100_20_50": "../Peter_DFE/training_loss_npy/val_loss_adamw_0_0_100_15_35_20_50_2048.npy",
    "EfficientNet_Adamw_0_0_100_10_30": "../Peter_DFE/training_loss_npy/val_loss_adamw_0_0_100_15_35_10_30_2048.npy",
    "EfficientNet_Adamw_30_30_100_10_30": "../Peter_DFE/training_loss_npy/val_loss_adamw_30_30_100_15_35_10_30_2048.npy",
    "adam_0_0_100_15_35_10_30_2048": "../Peter_DFE/training_loss_npy/val_loss_adam_0_0_100_15_35_10_30_2048.npy",
    "adamax_0_0_100_15_35_10_30_2048": "../Peter_DFE/training_loss_npy/val_loss_adamax_0_0_100_15_35_10_30_2048_pretrain.npy",
    "0_0_100_15_35_10_30_2048_291_finetune": "../Peter_DFE/training_loss_npy/val_loss_adamw_0_0_100_15_35_10_30_2048_finetune.npy",
    "0_0_100_15_35_10_30_2048_47_finetune": "../Peter_DFE/training_loss_npy/val_loss_efficientnet_adamw_0_0_100_15_35_10_30_2048_47_finetune.npy",
    "resnet34_adamw_0_0_100_15_35_10_30_2048_finetune": "../Peter_DFE/training_loss_npy/val_loss_resnet34_adamw_0_0_100_15_35_10_30_2048_finetune.npy",
    "0_0_0_15_35_10_30_2048_48_finetune": "../Peter_DFE/training_loss_npy/val_loss_efficientnet_adamw_0_0_0_15_35_10_30_2048_48_finetune.npy",
    "30_0_25_2048": "../Peter_DFE/training_loss_npy/val_loss_efficientnet_adamw_30_0_25_2048_48_pretrain.npy",
    "efficientnet_vit_adamw_100_10_30_2048": "../Peter_DFE/training_loss_npy/val_loss_efficientnet_vit_adamw_100_10_30_2048_97_finetune.npy",
    # "efficientnet_vit_16_adamw_100_10_30_2048": "../Peter_DFE/training_loss_npy/val_loss_efficientnet_vit_16_16_adamw_100_10_30_2048_97_finetune.npy",
    "swin_cnn_adamw_100_10_30_2048": "../Peter_DFE/training_loss_npy/val_loss_swin_cnn_adamw_100_10_30_2048_49_finetune.npy",
    "swin_cnn_adamw_30_30_100_10_30_2048": "../Peter_DFE/training_loss_npy/val_loss_efficientnet_vit_16_16_adamw_100_10_30_2048_97_finetune.npy",


}

data_loss = {key: load_data(value) for key, value in loss_paths.items()}
test_losses = [0.0007321917219087481, 0.0006670656148344278, 0.0006589707336388528, 0.0006784470751881599, 0.0005908077000640333, 
               0.00034902215702459216, 0.0003512469702400267, 0.00036438045208342373, 0.00041653821244835854, 0.014782303012907505,
               0.012820134870707989, 0.00032375985756516457, 0.00034657117794267833, 0.00034352202783338726, 0.00028347145416773856,
               0.0003842534206341952, 0.00031815737020224333, 0.0005244513158686459, 0.011030909605324268, 0.017428671941161156, 
               0.00017212535021826625, 0.0001330557424807921, 0.000121594152005855, 0.00011750611156458035, 0.00034794333623722196,
               0.00332098756916821, 0, 0]

Euclidean_distances = [56.255465336563006, 34.91000955307271, 25.44784679471318, 45.69280034379979, 37.524199318611345, 
                       29.75408986868529, 30.792716291493562, 25.5049190113515, 21.106414481826853, 415.48712780579933,
                       399.4184899997058, 28.031064342330886, 27.39298723172386, 26.41612848582006, 23.133065249895196,
                       24.252673138296466, 24.252673138296466, 21.08115478468134, 381.91577277171007, 251.14421582234095,
                       21.003913978653763, 17.77320380199737, 23.57551310442576, 22.446169867527296, 25.17897389086025, 
                       31.27809660385795, 0, 0]

total_time = [237.33057475090027, 202.10171151161194, 214.9658546447754, 212.70779752731323, 200.25588750839233, 
              231.18116641044617, 185.84633994102478, 136.4427990913391, 181.08564257621765, 177.201669216156,
              181.5913965702057, 185.0553901195526, 236.35819244384766, 140.8409023284912, 174.43701910972595,
              172.63589215278625, 230.70649123191833, 143.46708297729492, 134.71972846984863, 136.13556599617004,
              171.77727699279785, 178.60874390602112, 225.55482721328735, 218.86536264419556, 181.74158143997192,
              167.07763767242432, 0, 0]

i = 0
for key, value in data_loss.items():
    # print("key:", key, " min loss:", np.min(value), "test loss:", test_losses[i], "\n", "Euclidean distance", Euclidean_distances[i], "total time", total_time[i], "\n")
    color = plt.cm.tab20(i / 20)
    plt.plot(value, label=key, color=color)
    i += 1


plt.xlabel('Epochs', fontsize=44)
plt.ylabel('Loss', fontsize=44)
plt.legend(fontsize = 15)
# plt.show()

# plt.figure()
# plt.plot(data_loss["val_loss_efficientnet_vit_adamw_100_10_30_2048_597_pretrain"])
# plt.show()
coords_val_loss = [
    (70, 70, 0.00016300359251823982),
    (70, 30, 0.00015523709324069565),
    (70, 100, 0.00015239848980058094),
    (70, 0, 0.00015633207398303613),
    # (50, 50, 0.00014210190032574817),
    (30, 70, 0.0001037445312461911),
    (30, 30, 0.00010366272181272507),
    (30, 100, 0.00010607913510370864),
    (30, 0, 0.00011545945500426443),
    # (100, 30, 0.014782303012907505),
    # (100, 0, 0.012820134870707989),
    (0, 70, 0.00012099629506616629),
    (0, 30, 0.0001380747571535728),
    (0, 100, 0.00013209039460736782),
    (0, 0, 0.00009647070469095093),
]

coords_test_loss = [
    (70, 70, 0.0007321917219087481),
    (70, 30, 0.0006670656148344278),
    (70, 100, 0.0006589707336388528),
    (70, 0, 0.0006784470751881599),
    # (50, 50, 0.0005908077000640333),
    (30, 70, 0.00034902215702459216),
    (30, 30, 0.0003512469702400267),
    (30, 100, 0.00036438045208342373),
    (30, 0, 0.00041653821244835854),
    # (100, 30, 0.014782303012907505),
    # (100, 0, 0.012820134870707989),
    (0, 70, 0.00032375985756516457),
    (0, 30, 0.00034657117794267833),
    (0, 100, 0.00034352202783338726),
    (0, 0, 0.00028347145416773856),
]

coords_euclidean_distances = [
    (70, 70, 56.255465336563006),
    (70, 30, 34.91000955307271),
    (70, 100, 25.44784679471318),
    (70, 0, 45.69280034379979),
    # (50, 50, 37.524199318611345),
    (30, 70, 29.75408986868529),
    (30, 30, 30.792716291493562),
    (30, 100, 25.5049190113515),
    (30, 0, 21.106414481826853),
    # (100, 30, 415.48712780579933),
    # (100, 0, 399.4184899997058),
    (0, 70, 28.031064342330886),
    (0, 30, 28.249999999999996),
    (0, 100, 27.39298723172386),
    (0, 0, 26.41612848582006),
]

# 转换为numpy数组方便处理
coords_val_loss_array = np.array(coords_val_loss)
coords_test_loss_array = np.array(coords_test_loss)
coords_euclidean_distances_array = np.array(coords_euclidean_distances)

# 获取X和Y坐标的唯一值并排序
unique_val_xs = np.unique(coords_val_loss_array[:, 0])
unique_val_ys = np.unique(coords_val_loss_array[:, 1])
unique_test_xs = np.unique(coords_test_loss_array[:, 0])
unique_test_ys = np.unique(coords_test_loss_array[:, 1])
unique_euclidean_distances_xs = np.unique(coords_euclidean_distances_array[:, 0])
unique_euclidean_distances_ys = np.unique(coords_euclidean_distances_array[:, 1])

# 创建热图数据矩阵
heatmap_data_val = np.full((len(unique_val_ys), len(unique_val_xs)), np.nan)
heatmap_data_test = np.full((len(unique_test_ys), len(unique_test_xs)), np.nan)
heatmap_data_euclidean_distances = np.full((len(unique_euclidean_distances_ys), len(unique_euclidean_distances_xs)), np.nan)

# 填充热图数据
for x, y, loss in coords_val_loss:
    x_index = np.where(unique_val_xs == x)[0][0]
    y_index = np.where(unique_val_ys[::-1] == y)[0][0]
    heatmap_data_val[y_index, x_index] = loss

for x, y, loss in coords_test_loss:
    x_index = np.where(unique_test_xs == x)[0][0]
    y_index = np.where(unique_test_ys[::-1] == y)[0][0]
    heatmap_data_test[y_index, x_index] = loss

for x, y, loss in coords_euclidean_distances:
    x_index = np.where(unique_euclidean_distances_xs == x)[0][0]
    y_index = np.where(unique_euclidean_distances_ys[::-1] == y)[0][0]
    heatmap_data_euclidean_distances[y_index, x_index] = loss
# 绘制热图
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data_val, annot=True, xticklabels=unique_val_xs, yticklabels=unique_val_ys[::-1], cmap="Spectral", fmt=".2e", annot_kws={"size": 30})
# # plt.title("Validaion Loss Heatmap")
# plt.xlabel("Gaussian Noise (percentage)", fontsize=44)
# plt.ylabel("Gaussian Blur (percentage)", fontsize=44)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)

# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data_test, annot=True, xticklabels=unique_test_xs, yticklabels=unique_test_ys[::-1], cmap="Spectral", fmt=".2e", annot_kws={"size": 30})
# # plt.title("Test Loss Heatmap")
# plt.xlabel("Gaussian Noise (percentage)", fontsize=44)
# plt.ylabel("Gaussian Blur (percentage)", fontsize=44)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)

# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data_euclidean_distances, annot=True, xticklabels=unique_euclidean_distances_xs, yticklabels=unique_euclidean_distances_ys[::-1], cmap="Spectral", fmt=".5f", annot_kws={"size": 30})
# # plt.title("Euclidean Distance Heatmap")
# plt.xlabel("Gaussian Noise (percentage)", fontsize=44)
# plt.ylabel("Gaussian Blur (percentage)", fontsize=44)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.show()

"""
plot relabeled ground truth
"""


ground_truth_paths = {
    "Rex": "../dfe_ground_truth/04941364_R_B_red_sticker_5_people/04941364_R_B_red_sticker_coordinates_Rex.npy",
    # "Benson": "../dfe_ground_truth/handmole_ground_truth_coordinates_benson.npy",
    "Peter": "../dfe_ground_truth/04941364_R_B_red_sticker_5_people/04941364_R_B_red_sticker_coordinates_Peter.npy",
    "Joey": "../dfe_ground_truth/04941364_R_B_red_sticker_5_people/04941364_R_B_red_sticker_coordinates_Joey.npy",
    "David": "../dfe_ground_truth/04941364_R_B_red_sticker_5_people/04941364_R_B_red_sticker_coordinates_David.npy",
    # "Dallen": "../dfe_ground_truth/04941364_R_B_red_sticker_5_people/04941364_R_B_red_sticker_coordinates_Dallen.npy",
}

data_ground_truth = {key: load_data(value) for key, value in ground_truth_paths.items()}
# print(data_ground_truth["Joey"])
ground_truth_coordinates = {key: extract_coordinates(value) for key, value in data_ground_truth.items()}
total_sum = np.zeros((71, 2))

all_ground_truth = []
# 將每個key的value相加
for key, value in ground_truth_coordinates.items():
    # print(key, ":", len(value))
    total_sum += np.array(value).T
    all_ground_truth.append(np.array(value).T)
ground_truth_mean = total_sum / len(ground_truth_coordinates)
print(ground_truth_mean.shape)

all_ground_truth = np.array(all_ground_truth).astype(float)
all_ground_truth -= ground_truth_mean    #calculate distance between annotations of five people with ground_truth_mean
print(all_ground_truth.shape)

reshaped_ground_truth = all_ground_truth.reshape(-1, 2)
print(reshaped_ground_truth.shape)
# std_dev_x = np.std(all_ground_truth[:, :, 0], axis=0)
# std_dev_y = np.std(all_ground_truth[:, :, 1], axis=0)
std_dev_x = np.std(reshaped_ground_truth[:, 0], axis=0)
std_dev_y = np.std(reshaped_ground_truth[:, 1], axis=0)
print(np.sum(std_dev_x))
print(np.sum(std_dev_y))
# print(len(ground_truth_coordinates))
# print(ground_truth_mean)

plt.figure()
plt.plot(ground_truth_coordinates["Rex"][0], label="Rex")
# plt.plot(ground_truth_coordinates["Benson"][0], label="Benson")
plt.plot(ground_truth_coordinates["Peter"][0], label="Peter")
plt.plot(ground_truth_coordinates["Joey"][0], label="Joey")
plt.plot(ground_truth_coordinates["David"][0], label="David")
# plt.plot(ground_truth_coordinates["Dallen"][0], label="Dallen")
# plt.plot(coordinates["Jose_ground_truth_point"][0], label="Jose_ground_truth_point")
plt.xlabel('frame')
plt.ylabel('x coordinate')
plt.legend(loc="best")

plt.figure()
plt.plot(ground_truth_coordinates["Rex"][1], label="Rex")
# plt.plot(ground_truth_coordinates["Benson"][1], label="Benson")
plt.plot(ground_truth_coordinates["Peter"][1], label="Peter")
plt.plot(ground_truth_coordinates["Joey"][1], label="Joey")
plt.plot(ground_truth_coordinates["David"][1], label="David")
# plt.plot(ground_truth_coordinates["Dallen"][1], label="Dallen")
# plt.plot(coordinates["Jose_ground_truth_point"][1], label="Jose_ground_truth_point")
plt.xlabel('frame')
plt.ylabel('y coordinate')

plt.legend(loc="best")
plt.show()

# plt.figure()

# plt.plot(ground_truth_mean[:, 0], label="x coordinate")
# plt.plot(ground_truth_mean[:, 1], label="y coordinate")
# plt.xlabel('frame')
# plt.ylabel('y coordinate')
# plt.legend(loc="best")
# plt.show()

# ground_truth_labeling_distance = calculate_distance(ground_truth_coordinates["Peter"][0], ground_truth_coordinates["Peter"][1], ground_truth_mean[:, 0], ground_truth_mean[:, 1])
# print("sum ground_truth_labeling_distance", np.sum(ground_truth_labeling_distance))
# plt.figure()

# plt.plot(ground_truth_labeling_distance, label="ground_truth_labeling_distance")

# plt.xlabel('frame')
# plt.ylabel("Euclidean distance (pixel)")
# plt.legend(loc="best")
# plt.show()
# plt.figure()
# plt.hist(ground_truth_labeling_distance, bins=100)
# plt.show()

# np.save("../dfe_ground_truth/mean_ground_truth/04941364_R_B_little_finger_nail_coordinates_mean.npy", ground_truth_mean)

# ground_truth_mean = np.load("../dfe_ground_truth/mean_ground_truth/04941364_R_B_ground_truth_coordinates_mean.npy")
print("ground_truth_mean[0]: ", ground_truth_mean[0])

ssr_path = "../QuantPD_data/handmole_DFE_tf_40_subpixel_220_238_ssr.npy"
ssr_prevoius_path = "../QuantPD_data/handmole_DFE_tf_40_subpixel_220_238_previous_ssr.npy"
ssr = np.load(ssr_path)
ssr_prevoius = np.load(ssr_prevoius_path)

# plt.figure()
plt.plot(ssr, label="SSR with first frame", linewidth=2.5)
plt.plot(range(1, len(ssr_prevoius) + 1), ssr_prevoius, label="SSR with previous frame", linewidth=2.5)
plt.xlabel("Frame", fontsize=44)
plt.ylabel("SSR", fontsize=44)
plt.legend(loc="best", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.show()


"""
calculate the chi square distance
"""

def chi_square_distance(distance_x, distance_y, sigma_x = 0.253, sigma_y = 0.279):    #handmole 40 frame: sigma_x = 0.253, sigma_y = 0.279    04941364_R_B 71 frame: sigma_x = 0.244, sigma_y = 0.251
    return ((distance_x**2)/(sigma_x**2) + (distance_y**2)/(sigma_y**2))

def calculate_distance_for_chi(coord1_x, coord1_y, ground_truth_x, ground_truth_y):
    return (coord1_x - ground_truth_x), (coord1_y - ground_truth_y)


DFE_tf_paths = {
    "ground_truth_point": "../dfe_ground_truth/bbox_centers_pd.npy",    #../Peter_handmole_ground_truth_mean.npy
    "cotracker_point": "../QuantPD_data/cotracker_pred/handmole_cotracker_predict_coords.npy",
    "dfe_revised_point": "../QuantPD_data/handmole_DFE_tf_revised_40_pixel_20_20_preds.npy",
    "dfe_point": "../QuantPD_data/handmole_DFE_tf_40_pixel_0315_220_239_preds.npy",
}

DFE_tf_data = {key: load_data(value) for key, value in DFE_tf_paths.items()}
DFE_tf_coordinates = {key: extract_coordinates(value) for key, value in DFE_tf_data.items()}
DFE_tf_revised_x, DFE_tf_revised_y = revise_cotracker_distance(DFE_tf_coordinates["dfe_revised_point"][0], DFE_tf_coordinates["dfe_revised_point"][1], DFE_tf_coordinates["cotracker_point"][0], DFE_tf_coordinates["cotracker_point"][1], 20)

DFE_tf_distances = {}
DFE_tf_distances["cotracker_distances"] = calculate_distance_for_chi(DFE_tf_coordinates["cotracker_point"][0], DFE_tf_coordinates["cotracker_point"][1], DFE_tf_coordinates["ground_truth_point"][0], DFE_tf_coordinates["ground_truth_point"][1])
DFE_tf_distances["dfe_revised_distances"] = calculate_distance_for_chi(DFE_tf_revised_x, DFE_tf_revised_y, DFE_tf_coordinates["ground_truth_point"][0], DFE_tf_coordinates["ground_truth_point"][1])
DFE_tf_distances["dfe_distances"] = calculate_distance_for_chi(DFE_tf_coordinates["dfe_point"][0], DFE_tf_coordinates["dfe_point"][1], DFE_tf_coordinates["ground_truth_point"][0], DFE_tf_coordinates["ground_truth_point"][1])

DFE_tf_distances["dfe_revised_distances_cotracker_base"] = calculate_distance_for_chi(DFE_tf_revised_x, DFE_tf_revised_y, DFE_tf_coordinates["cotracker_point"][0], DFE_tf_coordinates["cotracker_point"][1])
DFE_tf_distances["dfe_distances_cotracker_base"] = calculate_distance_for_chi(DFE_tf_coordinates["dfe_point"][0], DFE_tf_coordinates["dfe_point"][1], DFE_tf_coordinates["cotracker_point"][0], DFE_tf_coordinates["cotracker_point"][1])

DFE_tf_distances["cotracker_distances_dfe_base"] = calculate_distance_for_chi(DFE_tf_coordinates["cotracker_point"][0], DFE_tf_coordinates["cotracker_point"][1], DFE_tf_coordinates["dfe_point"][0], DFE_tf_coordinates["dfe_point"][1])
DFE_tf_distances["dfe_revised_distances_dfe_base"] = calculate_distance_for_chi(DFE_tf_revised_x, DFE_tf_revised_y, DFE_tf_coordinates["dfe_point"][0], DFE_tf_coordinates["dfe_point"][1])

DFE_tf_distances["cotracker_distances_revised_base"] = calculate_distance_for_chi(DFE_tf_coordinates["cotracker_point"][0], DFE_tf_coordinates["cotracker_point"][1], DFE_tf_revised_x, DFE_tf_revised_y)
DFE_tf_distances["dfe_distances_revised_base"] = calculate_distance_for_chi(DFE_tf_coordinates["dfe_point"][0], DFE_tf_coordinates["dfe_point"][1], DFE_tf_revised_x, DFE_tf_revised_y)



DFE_tf_distances["cotracker_distances_sum"] = calculate_distance(DFE_tf_coordinates["cotracker_point"][0], DFE_tf_coordinates["cotracker_point"][1], DFE_tf_coordinates["ground_truth_point"][0], DFE_tf_coordinates["ground_truth_point"][1])
DFE_tf_distances["dfe_revised_distances_sum"] = calculate_distance(DFE_tf_revised_x, DFE_tf_revised_y, DFE_tf_coordinates["ground_truth_point"][0], DFE_tf_coordinates["ground_truth_point"][1])
DFE_tf_distances["dfe_distances_sum"] = calculate_distance(DFE_tf_coordinates["dfe_point"][0], DFE_tf_coordinates["dfe_point"][1], DFE_tf_coordinates["ground_truth_point"][0], DFE_tf_coordinates["ground_truth_point"][1])
# print("Total_Cotracker_Euclidean_distances", np.sum(DFE_tf_distances["cotracker_distances_sum"]))
# print("Total_dfe_Euclidean_distance: ", np.sum(DFE_tf_distances["dfe_distances_sum"]))
# print("Total_dfe_revised_Euclidean_distance: ", np.sum(DFE_tf_distances["dfe_revised_distances_sum"]))
# print(DFE_tf_distances["cotracker_distances"].shape)
total_x_chi_cotracker = 0
total_x_chi_dfe = 0
total_x_chi_dfe_revised = 0

total_x_chi_dfe_cotracker_base = 0
total_x_chi_dfe_revised_cotracker_base = 0

total_x_chi_cotracker_dfe_base = 0
total_x_chi_dfe_revised_dfe_base = 0

total_x_chi_cotracker_revised_base = 0
total_x_chi_dfe_revised_base = 0

for i in range(40):
    x_chi_cotracker = chi_square_distance(DFE_tf_distances["cotracker_distances"][0][i], DFE_tf_distances["cotracker_distances"][1][i])
    total_x_chi_cotracker += x_chi_cotracker
    # print(i)
    x_chi_dfe = chi_square_distance(DFE_tf_distances["dfe_distances"][0][i], DFE_tf_distances["dfe_distances"][1][i])
    total_x_chi_dfe += x_chi_dfe
    x_chi_dfe_revised = chi_square_distance(DFE_tf_distances["dfe_revised_distances"][0][i], DFE_tf_distances["dfe_revised_distances"][1][i])
    total_x_chi_dfe_revised += x_chi_dfe_revised

    x_chi_dfe_cotracker_base = chi_square_distance(DFE_tf_distances["dfe_distances_cotracker_base"][0][i], DFE_tf_distances["dfe_distances_cotracker_base"][1][i])
    total_x_chi_dfe_cotracker_base += x_chi_dfe_cotracker_base
    x_chi_dfe_revised_cotracker_base = chi_square_distance(DFE_tf_distances["dfe_revised_distances_cotracker_base"][0][i], DFE_tf_distances["dfe_revised_distances_cotracker_base"][1][i])
    total_x_chi_dfe_revised_cotracker_base += x_chi_dfe_revised_cotracker_base

    x_chi_cotracker_dfe_base = chi_square_distance(DFE_tf_distances["cotracker_distances_dfe_base"][0][i], DFE_tf_distances["cotracker_distances_dfe_base"][1][i])
    total_x_chi_cotracker_dfe_base += x_chi_cotracker_dfe_base
    x_chi_dfe_revised_dfe_base = chi_square_distance(DFE_tf_distances["dfe_revised_distances_dfe_base"][0][i], DFE_tf_distances["dfe_revised_distances_dfe_base"][1][i])
    total_x_chi_dfe_revised_dfe_base += x_chi_dfe_revised_dfe_base

    x_chi_cotracker_revised_base = chi_square_distance(DFE_tf_distances["cotracker_distances_revised_base"][0][i], DFE_tf_distances["cotracker_distances_revised_base"][1][i])
    total_x_chi_cotracker_revised_base += x_chi_cotracker_revised_base
    x_chi_dfe_revised_base = chi_square_distance(DFE_tf_distances["dfe_distances_revised_base"][0][i], DFE_tf_distances["dfe_distances_revised_base"][1][i])
    total_x_chi_dfe_revised_base += x_chi_dfe_revised_base
    
# print("--------------------------------")
# print("Chi_square_distance_cotracker: ", total_x_chi_cotracker)
# print("Chi_square_distance_dfe: ", total_x_chi_dfe)
# print("Chi_square_distance_dfe_revised: ", total_x_chi_dfe_revised)
# print("--------------------------------")
# print("Chi_square_distance_dfe_cotracker_base: ", total_x_chi_dfe_cotracker_base)
# print("Chi_square_distance_dfe_revised_cotracker_base: ", total_x_chi_dfe_revised_cotracker_base)
# print("--------------------------------")
# print("Chi_square_distance_cotracker_dfe_base: ", total_x_chi_cotracker_dfe_base)
# print("Chi_square_distance_dfe_revised_dfe_base: ", total_x_chi_dfe_revised_dfe_base)
# print("--------------------------------")
# print("Chi_square_distance_cotracker_revised_base: ", total_x_chi_cotracker_revised_base)
# print("Chi_square_distance_dfe_revised_base: ", total_x_chi_dfe_revised_base)



"""
plot different area size of Cotracker-DFE
"""
areas_paths = {
    "ground_truth_point": "../QuantPD_data/Peter_handmole_ground_truth_mean.npy",    #../Peter_handmole_ground_truth_mean.npy
    "cotracker_point": "../QuantPD_data/cotracker_pred/Handmole_cotracker_pred_220_238.npy",
    "dfe_revised_point_20": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_20_20.npy",
    "dfe_revised_point_22": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_22_22.npy",
    "dfe_revised_point_24": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_24_24.npy",
    "dfe_revised_point_26": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_26_26.npy",
    "dfe_revised_point_28": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_28_28.npy",
    "dfe_revised_point_30": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_30_30.npy",
    "dfe_revised_point_32": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_32_32.npy",
    "dfe_revised_point_34": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_34_34.npy",
    "dfe_revised_point_36": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_36_36.npy",
    "dfe_revised_point_38": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_38_38.npy",
    "dfe_revised_point_40": "../QuantPD_data/handmole_DFE_tf_revised_40_subpixel_optimized_rank_14ref_40_40.npy",
}

areas_data = {key: load_data(value) for key, value in areas_paths.items()}

areas_coordinates = {key: extract_coordinates(value) for key, value in areas_data.items()}
areas_coordinates["ground_truth_point"][0][0] = int(areas_coordinates["ground_truth_point"][0][0])
areas_coordinates["ground_truth_point"][1][0] = int(areas_coordinates["ground_truth_point"][1][0])
def_revised_20_x, def_revised_20_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_20"][0], areas_coordinates["dfe_revised_point_20"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 20)
def_revised_22_x, def_revised_22_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_22"][0], areas_coordinates["dfe_revised_point_22"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 22)
dfe_revised_24_x, dfe_revised_24_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_24"][0], areas_coordinates["dfe_revised_point_24"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 24)
dfe_revised_26_x, dfe_revised_26_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_26"][0], areas_coordinates["dfe_revised_point_26"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 26)
dfe_revised_28_x, dfe_revised_28_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_28"][0], areas_coordinates["dfe_revised_point_28"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 28)
dfe_revised_30_x, dfe_revised_30_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_30"][0], areas_coordinates["dfe_revised_point_30"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 30)
dfe_revised_32_x, dfe_revised_32_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_32"][0], areas_coordinates["dfe_revised_point_32"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 32)
dfe_revised_34_x, dfe_revised_34_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_34"][0], areas_coordinates["dfe_revised_point_34"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 34)
dfe_revised_36_x, dfe_revised_36_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_36"][0], areas_coordinates["dfe_revised_point_36"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 36)
dfe_revised_38_x, dfe_revised_38_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_38"][0], areas_coordinates["dfe_revised_point_38"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 38)
dfe_revised_40_x, dfe_revised_40_y = revise_cotracker_distance(areas_coordinates["dfe_revised_point_40"][0], areas_coordinates["dfe_revised_point_40"][1], areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], 40)

areas_distances = {}

areas_distances["cotracker_areas_distances"] = calculate_distance(areas_coordinates["cotracker_point"][0], areas_coordinates["cotracker_point"][1], areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_20_areas_distances"] = calculate_distance(def_revised_20_x, def_revised_20_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_22_areas_distances"] = calculate_distance(def_revised_22_x, def_revised_22_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_24_areas_distances"] = calculate_distance(dfe_revised_24_x, dfe_revised_24_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_26_areas_distances"] = calculate_distance(dfe_revised_26_x, dfe_revised_26_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_28_areas_distances"] = calculate_distance(dfe_revised_28_x, dfe_revised_28_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_30_areas_distances"] = calculate_distance(dfe_revised_30_x, dfe_revised_30_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_32_areas_distances"] = calculate_distance(dfe_revised_32_x, dfe_revised_32_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_34_areas_distances"] = calculate_distance(dfe_revised_34_x, dfe_revised_34_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_36_areas_distances"] = calculate_distance(dfe_revised_36_x, dfe_revised_36_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_38_areas_distances"] = calculate_distance(dfe_revised_38_x, dfe_revised_38_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])
areas_distances["dfe_revised_40_areas_distances"] = calculate_distance(dfe_revised_40_x, dfe_revised_40_y, areas_coordinates["ground_truth_point"][0], areas_coordinates["ground_truth_point"][1])   


# plt.figure(figsize=(5, 6))


plt.plot(areas_distances["dfe_revised_20_areas_distances"], 's-', label='Cotracker-DFE_20', color='blue')
plt.plot(areas_distances["dfe_revised_22_areas_distances"], 's-', label='Cotracker-DFE_22', color='orange')
plt.plot(areas_distances["dfe_revised_24_areas_distances"], 's-', label='Cotracker-DFE_24', color='green')
plt.plot(areas_distances["dfe_revised_26_areas_distances"], 's-', label='Cotracker-DFE_26', color='red')
plt.plot(areas_distances["dfe_revised_28_areas_distances"], 's-', label='Cotracker-DFE_28', color='purple')
plt.plot(areas_distances["dfe_revised_30_areas_distances"], 's-', label='Cotracker-DFE_30', color='brown')
plt.plot(areas_distances["dfe_revised_32_areas_distances"], 's-', label='Cotracker-DFE_32', color='pink')
plt.plot(areas_distances["dfe_revised_34_areas_distances"], 's-', label='Cotracker-DFE_34', color='gray')
plt.plot(areas_distances["dfe_revised_36_areas_distances"], 's-', label='Cotracker-DFE_36', color='cyan')
plt.plot(areas_distances["dfe_revised_38_areas_distances"], 's-', label='Cotracker-DFE_38', color='magenta')
plt.plot(areas_distances["dfe_revised_40_areas_distances"], 's-', label='Cotracker-DFE_40', color='olive')
plt.legend(loc='best', fontsize=23)
plt.xlim(0, 39)
plt.xlabel('Frame', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.ylabel('Euclidean distance (pixel)', fontsize=44)
# plt.show()

# total_time = np.load('../QuantPD_data/EfficientNet_dynamic_total_time.npy')
# total_errors = np.load('../QuantPD_data/EfficientNet_dynamic_total_errors.npy')
# print("total_time: ", total_time)
# print("total_errors: ", total_errors)



"""
plot different tracking type and different backbones of Cotracker-DFE
"""
tracking_type_paths = {
    "efficientnet_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/efficientnet_100_ordered_total_errors.npy",
    "efficientnet_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/efficientnet_100_dynamic_total_errors.npy",
    "efficientnet_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/efficientnet_30_30_100_ordered_total_errors.npy",    #../Peter_handmole_ground_truth_mean.npy
    "efficientnet_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/efficientnet_30_30_100_dynamic_total_errors.npy",
    "resnet34_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/resnet34_100_ordered_total_errors.npy",
    "resnet34_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/resnet34_100_dynamic_total_errors.npy",
    "resnet34_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/resnet34_30_30_100_ordered_total_errors.npy",
    "resnet34_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/resnet34_30_30_100_dynamic_total_errors.npy",
    "resnet50_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/resnet50_100_ordered_total_errors.npy",
    "resnet50_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/resnet50_100_dynamic_total_errors.npy",
    "resnet50_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/resnet50_30_30_100_ordered_total_errors.npy",
    "resnet50_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/resnet50_30_30_100_dynamic_total_errors.npy",
    "swin_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/swin_100_ordered_total_errors.npy",
    "swin_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/swin_100_dynamic_total_errors.npy",
    "swin_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/swin_30_30_100_ordered_total_errors.npy",
    "swin_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/swin_30_30_100_dynamic_total_errors.npy",
    "Convnext_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/Convnext_100_ordered_total_errors.npy",
    "Convnext_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/Convnext_100_dynamic_total_errors.npy",
    "Convnext_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/Convnext_30_30_100_ordered_total_errors.npy",
    "Convnext_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/pd_handmole/Convnext_30_30_100_dynamic_total_errors.npy",



}


tracking_type_04941364_R_B_paths = {
    "04941364_R_B_efficientnet_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_efficientnet_100_ordered_total_errors.npy",
    "04941364_R_B_efficientnet_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_efficientnet_100_dynamic_total_errors.npy",
    "04941364_R_B_efficientnet_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_efficientnet_30_30_100_ordered_total_errors.npy",    #../Peter_handmole_ground_truth_mean.npy
    "04941364_R_B_efficientnet_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_efficientnet_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet34_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_resnet34_100_ordered_total_errors.npy",
    "04941364_R_B_resnet34_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_resnet34_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet34_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_resnet34_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_resnet34_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_resnet34_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet50_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_resnet50_100_ordered_total_errors.npy",
    "04941364_R_B_resnet50_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_resnet50_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet50_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_resnet50_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_resnet50_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_resnet50_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_swin_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_swin_100_ordered_total_errors.npy",
    "04941364_R_B_swin_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_swin_100_dynamic_total_errors.npy",
    "04941364_R_B_swin_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_swin_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_swin_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_swin_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_Convnext_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_Convnext_100_ordered_total_errors.npy",
    "04941364_R_B_Convnext_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_Convnext_100_dynamic_total_errors.npy",
    "04941364_R_B_Convnext_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_Convnext_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_Convnext_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_handmole/04941364_R_B_Convnext_30_30_100_dynamic_total_errors.npy",
    
} 

tracking_type_04941364_R_B_right_handmole_paths = {
    "04941364_R_B_efficientnet_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_efficientnet_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_efficientnet_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_efficientnet_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_efficientnet_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_efficientnet_30_30_100_ordered_total_errors.npy",    #../Peter_handmole_ground_truth_mean.npy
    "04941364_R_B_efficientnet_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_efficientnet_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet34_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_resnet34_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_resnet34_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_resnet34_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet34_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_resnet34_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_resnet34_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_resnet34_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet50_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_resnet50_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_resnet50_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_resnet50_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet50_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_resnet50_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_resnet50_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_resnet50_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_swin_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_swin_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_swin_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_swin_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_swin_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_swin_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_swin_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_swin_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_Convnext_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_Convnext_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_Convnext_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_Convnext_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_Convnext_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_Convnext_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_Convnext_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_Convnext_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_efficientnetv2_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_efficientnetv2_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_efficientnetv2_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_efficientnetv2_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_efficientnetv2_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_efficientnetv2_30_30_100_ordered_total_errors.npy", 
    "04941364_R_B_efficientnetv2_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_right_handmole/04941364_R_B_360_right_handmole_efficientnetv2_30_30_100_dynamic_total_errors.npy",
} 

tracking_type_04941364_R_B_red_sticker_paths = {
    "04941364_R_B_efficientnet_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_efficientnet_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_efficientnet_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_efficientnet_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_efficientnet_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_efficientnet_30_30_100_ordered_total_errors.npy",    #../Peter_handmole_ground_truth_mean.npy
    "04941364_R_B_efficientnet_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_efficientnet_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet34_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_resnet34_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_resnet34_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_resnet34_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet34_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_resnet34_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_resnet34_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_resnet34_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet50_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_resnet50_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_resnet50_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_resnet50_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_resnet50_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_resnet50_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_resnet50_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_resnet50_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_swin_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_swin_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_swin_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_swin_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_swin_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_swin_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_swin_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_swin_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_Convnext_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_Convnext_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_Convnext_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_Convnext_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_Convnext_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_Convnext_30_30_100_ordered_total_errors.npy",
    "04941364_R_B_Convnext_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_Convnext_30_30_100_dynamic_total_errors.npy",
    "04941364_R_B_efficientnetv2_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_efficientnetv2_0_0_100_ordered_total_errors.npy",
    "04941364_R_B_efficientnetv2_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_efficientnetv2_0_0_100_dynamic_total_errors.npy",
    "04941364_R_B_efficientnetv2_30_30_100_ordered": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_efficientnetv2_30_30_100_ordered_total_errors.npy", 
    "04941364_R_B_efficientnetv2_30_30_100_dynamic": "../QuantPD_data/tracking_result_multi_ref_npy/04941364_R_B_red_sticker/04941364_R_B_red_sticker_efficientnetv2_30_30_100_dynamic_total_errors.npy",
} 

tracking_type_data = {key: load_data(value) for key, value in tracking_type_04941364_R_B_red_sticker_paths.items()}
tracking_type_04941364_R_B_data = {key: load_data(value) for key, value in tracking_type_04941364_R_B_paths.items()}
length_of_ref = list(range(1, 20))

right_handmole_cotracker_pred = np.load("../QuantPD_data/cotracker_pred/04941364_R_B_right_handmole_cotracker_pred_coords.npy")
red_sticker_cotracker_pred = np.load("../QuantPD_data/cotracker_pred/04941364_R_B_red_sticker_cotracker_pred_coords.npy")
print(right_handmole_cotracker_pred.shape)
red_sticker_ground_truth = np.load("../dfe_ground_truth/mean_ground_truth/04941364_R_B_red_sticker_coordinates_mean.npy")
total_error_red_sticker_cotracker = np.sum(calculate_distance(red_sticker_cotracker_pred[:, 0], red_sticker_cotracker_pred[:, 1], red_sticker_ground_truth[:, 0], red_sticker_ground_truth[:, 1]))
total_error_red_sticker_for_plot = np.ones((19, 1)) * total_error_red_sticker_cotracker
print(total_error_red_sticker_cotracker)
print(len(length_of_ref))
print(len(tracking_type_data["04941364_R_B_Convnext_100_ordered"]))
plt.figure()
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_efficientnet_100_ordered"], 'o-', label='efficientnet_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_efficientnet_100_dynamic"], 's-', label='efficientnet_100_head-tail')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_efficientnet_30_30_100_ordered"], 'o--', label='efficientnet_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_efficientnet_30_30_100_dynamic"], 's--', label='efficientnet_30_30_100_head-tail')
plt.plot(length_of_ref, total_error_red_sticker_for_plot, '^-', label='Cotracker')
plt.xlim(1, 19)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_resnet34_100_ordered"], 'o-', label='resnet34_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_resnet34_100_dynamic"], 's-', label='resnet34_100_head-tail')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_resnet34_30_30_100_ordered"], 'o--', label='resnet34_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_resnet34_30_30_100_dynamic"], 's--', label='resnet34_30_30_100_head-tail')
plt.plot(length_of_ref, total_error_red_sticker_for_plot, '^-', label='Cotracker')
plt.xlim(1, 19)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_resnet50_100_ordered"], 'o-', label='resnet50_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_resnet50_100_dynamic"], 's-', label='resnet50_100_head-tail')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_resnet50_30_30_100_ordered"], 'o--', label='resnet50_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_resnet50_30_30_100_dynamic"], 's--', label='resnet50_30_30_100_head-tail')
plt.plot(length_of_ref, total_error_red_sticker_for_plot, '^-', label='Cotracker')
plt.xlim(1, 19)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_swin_100_ordered"], 'o-', label='swin_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_swin_100_dynamic"], 's-', label='swin_100_head-tail')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_swin_30_30_100_ordered"], 'o--', label='swin_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_swin_30_30_100_dynamic"], 's--', label='swin_30_30_100_head-tail')
plt.plot(length_of_ref, total_error_red_sticker_for_plot, '^-', label='Cotracker')
plt.xlim(1, 19)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_Convnext_100_ordered"], 'o-', label='Convnext_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_Convnext_100_dynamic"], 's-', label='Convnext_100_head-tail')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_Convnext_30_30_100_ordered"], 'o--', label='Convnext_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_Convnext_30_30_100_dynamic"], 's--', label='Convnext_30_30_100_head-tail')
plt.plot(length_of_ref, total_error_red_sticker_for_plot, '^-', label='Cotracker')
plt.xlim(1, 19)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_efficientnetv2_100_ordered"], 'o-', label='EfficientnetV2_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_efficientnetv2_100_dynamic"], 's-', label='EfficientnetV2_100_head-tail')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_efficientnetv2_30_30_100_ordered"], 'o--', label='EfficientnetV2_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_data["04941364_R_B_efficientnetv2_30_30_100_dynamic"], 's--', label='EfficientnetV2_30_30_100_head-tail')
plt.plot(length_of_ref, total_error_red_sticker_for_plot, '^-', label='Cotracker')
plt.xlim(1, 19)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)


plt.show()

#--------------------------------------------

plt.figure()
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_efficientnet_100_ordered"], 'o-', label='efficientnet_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_efficientnet_100_dynamic"], 's-', label='efficientnet_100_head-tail')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_efficientnet_30_30_100_ordered"], 'o--', label='efficientnet_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_efficientnet_30_30_100_dynamic"], 's--', label='efficientnet_30_30_100_head-tail')
plt.xlim(1, 20)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_resnet34_100_ordered"], 'o-', label='resnet34_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_resnet34_100_dynamic"], 's-', label='resnet34_100_head-tail')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_resnet34_30_30_100_ordered"], 'o--', label='resnet34_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_resnet34_30_30_100_dynamic"], 's--', label='resnet34_30_30_100_head-tail')
plt.xlim(1, 20)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_resnet50_100_ordered"], 'o-', label='resnet50_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_resnet50_100_dynamic"], 's-', label='resnet50_100_head-tail')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_resnet50_30_30_100_ordered"], 'o--', label='resnet50_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_resnet50_30_30_100_dynamic"], 's--', label='resnet50_30_30_100_head-tail')
plt.xlim(1, 20)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_swin_100_ordered"], 'o-', label='swin_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_swin_100_dynamic"], 's-', label='swin_100_head-tail')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_swin_30_30_100_ordered"], 'o--', label='swin_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_swin_30_30_100_dynamic"], 's--', label='swin_30_30_100_head-tail')
plt.xlim(1, 20)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

plt.figure()
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_Convnext_100_ordered"], 'o-', label='Convnext_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_Convnext_100_dynamic"], 's-', label='Convnext_100_head-tail')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_Convnext_30_30_100_ordered"], 'o--', label='Convnext_30_30_100_ordered')
plt.plot(length_of_ref, tracking_type_04941364_R_B_data["04941364_R_B_Convnext_30_30_100_dynamic"], 's--', label='Convnext_30_30_100_head-tail')
plt.xlim(1, 20)
plt.xlabel('Number of reference vectors', fontsize=44)
plt.ylabel('Euclidean distance (pixel)', fontsize=44)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(loc='best', fontsize=25)

# plt.show()

