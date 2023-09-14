import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from Model_Training.Models import LinearModelWithDropout, LinearModelWithDropout2
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df, convert_multi_frequency_eit_to_df
from plot_utils import solve_and_plot_cnn
from pyeit import mesh
from pyeit.eit import protocol
from pyeit.mesh.shape import thorax
import os

from reconstruction_algorithims import solve_and_plot_jack, solve_and_plot_greit
from utils import wait_for_start_of_measurement, get_relevant_voltages

""" 0. build mesh """
n_el = 32  # nb of electrodes
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
# Dist_exc is the distance between the excitation and measurement electrodes (in number of electrodes)

use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.1)

""" 1. problem setup """
anomaly = []
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
delta_perm = np.real(mesh_new.perm - mesh_obj.perm)

default_frame = None


def plot_time_diff_eit_image(v1_path, v0_path, frequency=1000):
    global default_frame
    df_v1 = convert_single_frequency_eit_file_to_df(v1_path)
    if default_frame is None:
        df_v0 = convert_single_frequency_eit_file_to_df(v0_path)
    else:
        df_v0 = default_frame
    v1 = get_relevant_voltages(df_v1, protocol_obj)
    v0 = get_relevant_voltages(df_v0, protocol_obj)
    # save v0 as npy
    np.save("v0.npy", v0)
    difference = (v1 - v0)
    # plt.plot(difference)
    difference = difference / v0
    difference = difference - np.mean(difference)
    # plt.plot(difference)
    # # plt.title("v1 - v0")
    # plt.title("Voltage difference")
    # # plt.legend(["v1 - v0", "(v1-v0)/v0"])
    # plt.show()
    img_name = v1_path.split('\\')[-1]
    save_path_cnn = f"{img_name}_cnn.png"
    save_path_jac = f"{img_name}_jac.png"
    v0_traditional_algorithims = v0[protocol_obj.keep_ba]
    v1_traditional_algorithims = v1[protocol_obj.keep_ba]
    # solve_and_plot_jack(v0, v1, mesh_obj, protocol_obj, path1_for_name_only=v1_path, path2_for_name_only=v0_path)
    # solve_and_plot_greit(v0_traditional_algorithims, v1_traditional_algorithims,
    #                      mesh_obj, protocol_obj, path1_for_name_only=v1_path, path2_for_name_only=v0_path)
    # solve_and_plot_bp(v0, v1, mesh_obj, protocol_obj, path1_for_name_only=path1, path2_for_name_only=path2)
    # solve_and_plot_cnn(model=model, voltage_difference=difference, chow_center_of_mass=True)
    solve_and_plot_cnn(model=model_pca, voltage_difference=difference_pca, chow_center_of_mass=True)


def plot_frequencies_diff_eit_image(path, f1, f2):
    """
    Plot the difference between two frequencies
    :param f1:
    :param f2:
    :return:
    """
    if not type(f1) == int or not type(f2) == int:
        raise Exception("f1 and f2 must be integers")
    df = convert_multi_frequency_eit_to_df(path)
    df1 = df[df["frequency"] == f1]
    df2 = df[df["frequency"] == f2]
    df1 = df.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    v0 = get_relevant_voltages(df1, protocol_obj)
    v1 = get_relevant_voltages(df2, protocol_obj)

    solve_and_plot_jack(v0, v1, mesh_obj, protocol_obj, path1_for_name_only=f"{f1}", path2_for_name_only=f"{f2}")


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
                if default_frame is None:
                    default_frame = current_frame
                else:
                    time.sleep(0.01)  # wait for file to be written
                    plot_time_diff_eit_image(v1_path=os.path.join(eit_path, current_frame),
                                             v0_path=os.path.join(eit_path, default_frame))
                    seen_files.append(current_frame)


path = "eit_data"

VOLTAGE_VECTOR_LENGTH = 1024
VOLTAGE_VECTOR_LENGTH_PCA = 128
OUT_SIZE = 64
print("Loading the model")
pca = pickle.load(open("pca.pkl", "rb"))

model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
model_pca = LinearModelWithDropout(input_size=VOLTAGE_VECTOR_LENGTH_PCA, output_size=OUT_SIZE ** 2)

model.load_state_dict(torch.load(
    "Collected_Data/Combined_dataset/Models/LinearModelDropout2/05_09_all_data_40mm_target_and_augmentation_more_noise/model_2023-09-05_15-34-02_epoche_120_of_200_best_model.pth"))

model_pca.load_state_dict(torch.load(
    "Collected_Data/Combined_dataset_multi_augmented/PCA_REDUCED128/Models/LinearModelWithDropout/Training_augmented_multifrequency_data_longer_training/model_2023-09-14_15-52-25_epoche_547_of_800_best_model.pth"))

# model.load_state_dict(torch.load(
#     "Own_Simulation_Dataset/Models/LinearModelDropout/Test_01_noise_regularization1e-6/model_2023-08-10_12-17-00_150_epochs.pth"))
model.eval()
model_pca.eval()
plot_eit_video(path)