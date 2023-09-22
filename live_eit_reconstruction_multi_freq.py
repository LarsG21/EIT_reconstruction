import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Model_Training.Models import LinearModelWithDropout
from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df
from plot_utils import solve_and_plot, solve_and_get_center
from utils import wait_for_start_of_measurement


def plot_multi_frequency_eit_image(v1_path, plot=False):
    global default_frame
    df = convert_multi_frequency_eit_to_df(v1_path)
    df_alternating = pd.DataFrame({"real": df["real"], "imaginary": df["imaginary"]}).stack().reset_index(drop=True)
    df_alternating = df_alternating.to_frame(name="amplitude")
    v1 = df_alternating["amplitude"].to_numpy(dtype=np.float64)
    # plt.plot(v1)
    # plt.show()
    # save v0 as npy
    PCA = True
    if PCA:
        v1 = pca.transform(v1.reshape(1, -1))
        if plot:
            plt.bar(x=range(len(v1.reshape(-1))), height=v1.reshape(-1))
            plt.title("PCA transformed voltage vector")
            plt.xlabel("PCA component")
            plt.ylabel("Intensity")
            plt.show()
    # solve_and_plot(model=model_pca, model_input=v1, chow_center_of_mass=False, use_opencv_for_plotting=True)
    img, center = solve_and_get_center(model=model_pca, model_input=v1)
    cv2.imshow("img", cv2.resize(img, (512, 512)))
    cv2.waitKey(1)
    return img, center


def plot_eit_video(path):
    """
    Plots the eit video from the given path.
    There are new files in the folder every few seconds.
    Do the same as above continuously.
    :param path:
    :return:
    """
    seen_files = []
    centers = []
    eit_path = wait_for_start_of_measurement(path)
    while True:
        for current_frame in os.listdir(os.getcwd()):
            empty_img = np.zeros([64, 64])
            if current_frame.endswith(".eit") and current_frame not in seen_files:
                time.sleep(0.01)  # wait for file to be written
                img, center = plot_multi_frequency_eit_image(os.path.join(eit_path, current_frame))
                centers.append(center)
                seen_files.append(current_frame)
                # last 10 centers
                for c in centers[-10:]:
                    # add circle to image to show center
                    cv2.circle(empty_img, (int(c[0]), int(c[1])), 1, (255, 255, 255), -1)
                    cv2.imshow("center", cv2.resize(empty_img, (512, 512)))



path = "eit_data"

VOLTAGE_VECTOR_LENGTH = 1024
VOLTAGE_VECTOR_LENGTH_PCA = 128
OUT_SIZE = 64
print("Loading the model")

model_pca = LinearModelWithDropout(input_size=VOLTAGE_VECTOR_LENGTH_PCA, output_size=OUT_SIZE ** 2)

model_pca_path = "Collectad_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies/Models/LinearModelWithDropout/run_9_1500_samples_more_negative_set_and_augmentation/model_2023-09-22_13-48-51_epoche_395_of_400_best_model.pth"
# get the pca.okl in the same folder as the model
pca_path = os.path.join(os.path.dirname(model_pca_path), "pca.pkl")
pca = pickle.load(open(pca_path, "rb"))
model_pca.load_state_dict(torch.load(model_pca_path))

model_pca.eval()
plot_eit_video(path)
