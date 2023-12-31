import pickle
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from Model_Training.Models import LinearModelWithDropout2, LinearModelWithDropout, LinearModel
from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df
from plot_utils import solve_and_plot_with_nural_network, preprocess_greit_img
from pyeit import mesh
from pyeit.eit import protocol
from pyeit.mesh.shape import thorax
import os

from reconstruction_algorithims import solve_and_plot_jack, solve_and_plot_greit, solve_and_plot_bp
from utils import wait_for_start_of_measurement, add_normalizations

""" 0. build mesh """
n_el = 32  # nb of electrodes
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
# Dist_exc is the distance between the excitation and measurement electrodes (in number of electrodes)


mesh_obj = mesh.create(n_el, h0=0.1)

""" 1. problem setup """
anomaly = []
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
delta_perm = np.real(mesh_new.perm - mesh_obj.perm)

default_frame = None

list_for_default_frames_to_average = []


def get_averaged_frame(path, number_of_avgs, frequency=1000):
    """
    Gets the averaged frame from the given path. This is used as the default frame for the time difference.
    :param path:
    :param number_of_avgs:
    :return:
    """
    global default_frame
    global list_for_default_frames_to_average
    df = convert_multi_frequency_eit_to_df(path)
    df = df[df["frequency"] == frequency]
    v1 = df["amplitude"].to_numpy(dtype=np.float64)
    if default_frame is None:
        list_for_default_frames_to_average.append(v1)
        print(f"Added frame to list {path}")
        if len(list_for_default_frames_to_average) >= number_of_avgs:
            temp = np.mean(np.array(list_for_default_frames_to_average), axis=0)
            # list_for_default_frames_to_average = []
            temp = temp.reshape(-1)
            print("Default frame calculated")
            # convert to df with the same format as the other frames
            df = df[df["frequency"] == 1000]
            df["amplitude"] = temp
            default_frame = df


def plot_time_diff_eit_image(v1_path, debug_plots=False):
    """
    Plots the time difference between the two given eit frames.
    :param v1_path:
    :param debug_plots:
    :return:
    """
    global default_frame
    df_v1 = convert_multi_frequency_eit_to_df(v1_path)
    # find the most common frequency
    most_common_frequency = int(df_v1["frequency"].value_counts().idxmax())
    if default_frame is None:
        # df_v0 = convert_multi_frequency_eit_to_df(v0_path)
        get_averaged_frame(v1_path, number_of_avgs=30, frequency=most_common_frequency)
        return
    else:
        df_v0 = default_frame
    df_v1 = df_v1[df_v1["frequency"] == most_common_frequency]
    # df_v0 = df_v0[df_v0["frequency"] == 1000]
    v1 = df_v1["amplitude"].to_numpy(dtype=np.float64)
    v0 = df_v0["amplitude"].to_numpy(dtype=np.float64)
    # v0 = np.load(os.path.join(path, "v0.npy"))
    # save v0 as npy
    # np.save("v0.npy", v0)
    # calculate the voltage difference
    difference = (v1 - v0) / v0
    # normalize the voltage difference
    normalized_difference = difference
    if debug_plots:
        plt.plot(v0, label="v0")
        plt.plot(v1, label="v1")
        plt.legend()
        plt.title("Voltage")
        plt.show()
        plt.plot(normalized_difference)
        plt.title("Normalized Voltage difference")
        plt.show()
    if pca is not None:
        normalized_difference = pca.transform(normalized_difference.reshape(1, -1))
    # v0_traditional_algorithims = v0[protocol_obj.keep_ba]
    # v1_traditional_algorithims = v1[protocol_obj.keep_ba]
    # solve_and_plot_jack(v0_traditional_algorithims, v1_traditional_algorithims, mesh_obj, protocol_obj,
    #                     path1_for_name_only=v1_path, path2_for_name_only=v0_path)
    # img_greit = solve_and_plot_greit(v0_traditional_algorithims, v1_traditional_algorithims,
    #                                  mesh_obj, protocol_obj, path1_for_name_only=v1_path, path2_for_name_only=v0_path,
    #                                  plot=False)
    # normalize the image
    # img_greit = preprocess_greit_img(img_greit)
    # # plt.imshow(img_greit)
    # # plt.show()
    # # np clip between 0  and 255
    # cv2.imshow("GREIT", cv2.resize(img_greit, (0, 0), fx=4, fy=4))
    # cv2.waitKey(1)

    solve_and_plot_with_nural_network(model=model, model_input=normalized_difference, chow_center_of_mass=False,
                                      use_opencv_for_plotting=True)




def plot_eit_video(path):
    """
    Plots the eit video from the given path.
    There are new files in the folder every few seconds.
    Do the same as above continuously.
    :param path:
    :return:
    """
    eit_path = ""
    seen_files = []
    eit_path = wait_for_start_of_measurement(path)
    default_frame = None
    start_time = time.time()
    while True:
        for current_frame in os.listdir(os.getcwd()):
            # calculate the frame rate
            if time.time() - start_time > 1:
                print("FPS: ", len(seen_files) / (time.time() - start_time))
            if current_frame.endswith(".eit") and current_frame not in seen_files:
                print(current_frame)
                if default_frame is None:
                    default_frame = current_frame
                else:
                    plot_time_diff_eit_image(v1_path=os.path.join(eit_path, current_frame))
                    seen_files.append(current_frame)
                    # for file in seen_files:
                    #     os.remove(file)
                    # seen_files = []


path = "C:\\Users\\lgudjons\\Desktop\\eit_data"
VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64

# model_path = "Collected_Data/Combined_dataset/Models/LinearModelWithDropout2/TESTING_MORE_DATA_12_10/model_2023-10-12_11-55-44_epoche_232_of_300_best_model.pth"
#
# model_path = "Trainings_Data_EIT32/1_Freq_More_Orientations/Models/LinearModelWithDropout2/TEST_GOOD_SETTINGS/model_2023-12-21_17-29-03_79_80.pth"
# model_path = "Collected_Data/Even_Orientation_Dataset/Models/LinearModelWithDropout2/DEBUG/model_2023-11-16_13-44-26_112_200.pth"
# model_path = "Trainings_Data_EIT32/1_Freq_More_Orientations/Models/LinearModelWithDropout2/TESTING_19_12/model_2023-12-19_16-19-06_79_80.pth"
model_path = "Trainings_Data_EIT32/1_Freq_More_Orientations/Models/LinearModelWithDropout2/Test_06_12_2/model_2023-12-06_15-06-56_65_70.pth"
pca = None
pca_path = os.path.join(os.path.dirname(model_path), "pca.pkl")
if os.path.exists(pca_path):
    print("Loading the PCA")
    pca = pickle.load(open(pca_path, "rb"))
    print("PCA loaded")
    VOLTAGE_VECTOR_LENGTH = pca.n_components_

print("Loading the model")
model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
model.load_state_dict(torch.load(model_path))
model.eval()

model.eval()
plot_eit_video(path)
