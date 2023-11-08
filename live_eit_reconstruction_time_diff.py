import pickle
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from Model_Training.Models import LinearModelWithDropout2, LinearModelWithDropout
from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df
from plot_utils import solve_and_plot_with_nural_network, preprocess_greit_img
from pyeit import mesh
from pyeit.eit import protocol
from pyeit.mesh.shape import thorax
import os

from reconstruction_algorithims import solve_and_plot_jack, solve_and_plot_greit, solve_and_plot_bp
from utils import wait_for_start_of_measurement

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


def plot_time_diff_eit_image(v1_path, v0_path, debug_plots=False):
    global default_frame
    df_v1 = convert_multi_frequency_eit_to_df(v1_path)
    if default_frame is None:
        df_v0 = convert_multi_frequency_eit_to_df(v0_path)
    else:
        df_v0 = default_frame
    df_v1 = df_v1[df_v1["frequency"] == 1000]
    df_v0 = df_v0[df_v0["frequency"] == 1000]
    v1 = df_v1["amplitude"].to_numpy(dtype=np.float64)
    v0 = df_v0["amplitude"].to_numpy(dtype=np.float64)
    # v0 = np.load("v0.npy")
    # save v0 as npy
    np.save("v0.npy", v0)
    # calculate the voltage difference
    difference = (v1 - v0)
    # normalize the voltage difference
    difference = difference / v0
    # normalized_difference = difference - np.mean(difference)
    normalized_difference = difference
    if debug_plots:
        plt.plot(v0)
        plt.plot(v1)
        plt.title("Voltage")
        plt.show()
        plt.plot(normalized_difference)
        plt.title("Normalized Voltage difference")
        plt.show()
    if PCA:
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


def plot_eit_images_in_folder(path):
    old_path = os.getcwd()
    eit_path = ""
    for file_or_folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_or_folder)):
            os.chdir(os.path.join(path, file_or_folder))
            print(os.getcwd())
            os.chdir((os.path.join(os.getcwd(), "setup")))
            eit_path = os.getcwd()
            print(eit_path)
            break

    default_frame = None
    for current_frame in os.listdir(os.getcwd()):
        if current_frame.endswith(".eit"):
            if default_frame is None:
                default_frame = current_frame
            else:
                print(default_frame, current_frame)
                plot_time_diff_eit_image(os.path.join(eit_path, current_frame), os.path.join(eit_path, default_frame))
                # last_frame = current_frame
    # reset path
    os.chdir(old_path)


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
    while True:
        for current_frame in os.listdir(os.getcwd()):
            if current_frame.endswith(".eit") and current_frame not in seen_files:
                print(current_frame)
                if default_frame is None:
                    default_frame = current_frame
                else:
                    time.sleep(0.001)  # wait for file to be written
                    plot_time_diff_eit_image(v1_path=os.path.join(eit_path, current_frame),
                                             v0_path=os.path.join(eit_path, default_frame))
                    seen_files.append(current_frame)


path = "eit_data"
PCA = False
VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64
print("Loading the model")

model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)

# model_path = "Collected_Data/Combined_dataset/Models/LinearModelWithDropout2/TESTING_MORE_DATA_12_10/model_2023-10-12_11-55-44_epoche_232_of_300_best_model.pth"
#
#
model_path = "Training_Data/1_Freq_with_individual_v0s/Models/LinearModelWithDropout2/Run_25_10_dataset_individual_v0s/model_2023-10-27_14-25-23_148_150.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

pca_path = os.path.join(os.path.dirname(model_path), "pca.pkl")
if PCA:
    print("Loading the PCA")
    pca = pickle.load(open(pca_path, "rb"))
# get the pca.pkl in the same folder as the model

model.eval()
plot_eit_video(path)