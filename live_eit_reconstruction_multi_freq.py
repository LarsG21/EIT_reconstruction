import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from Model_Training.Models import LinearModelWithDropout
from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df
from plot_utils import solve_and_plot_cnn
from utils import wait_for_start_of_measurement


def plot_multi_frequency_eit_image(v1_path):
    global default_frame
    df = convert_multi_frequency_eit_to_df(v1_path)
    df_alternating = pd.DataFrame({"real": df["real"], "imaginary": df["imaginary"]}).stack().reset_index(drop=True)
    df_alternating = df_alternating.to_frame(name="amplitude")
    v1 = df_alternating["amplitude"].to_numpy(dtype=np.float64)
    # save v0 as npy
    PCA = True
    if PCA:
        v1 = pca.transform(v1.reshape(1, -1))
    solve_and_plot_cnn(model=model_pca, voltage_difference=v1, chow_center_of_mass=True)


def plot_eit_video(path):
    """
    Plots the eit video from the given path.
    There are new files in the folder every few seconds.
    Do the same as above continuously.
    :param path:
    :return:
    """
    seen_files = []
    eit_path = wait_for_start_of_measurement(path)
    while True:
        for current_frame in os.listdir(os.getcwd()):
            if current_frame.endswith(".eit") and current_frame not in seen_files:
                time.sleep(0.01)  # wait for file to be written
                plot_multi_frequency_eit_image(os.path.join(eit_path, current_frame))
                seen_files.append(current_frame)


path = "eit_data"

VOLTAGE_VECTOR_LENGTH = 1024
VOLTAGE_VECTOR_LENGTH_PCA = 128
OUT_SIZE = 64
print("Loading the model")

model_pca = LinearModelWithDropout(input_size=VOLTAGE_VECTOR_LENGTH_PCA, output_size=OUT_SIZE ** 2)

model_pca_path = "Collected_Data/Combined_dataset/Models/LinearModelWithDropout/TESTING/model_2023-09-14_18-04-34_epoche_217_of_400_best_model.pth"
# get the pca.okl in the same folder as the model
pca_path = os.path.join(os.path.dirname(model_pca_path), "pca.pkl")
pca = pickle.load(open(pca_path, "rb"))
model_pca.load_state_dict(torch.load(model_pca_path))

model_pca.eval()
plot_eit_video(path)
