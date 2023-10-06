import logging
import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from Model_Training.Models import LinearModelWithDropout2, LinearModelWithDropout
from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df
from plot_utils import solve_and_plot
from utils import wait_for_start_of_measurement, preprocess_absolute_eit_frame, add_normalizations


def plot_multi_frequency_eit_image(v1_path, plot=False, save_video=False):
    global default_frame
    df = convert_multi_frequency_eit_to_df(v1_path)
    # Convert to an numpy array with alternating real and imag numbers
    v1 = preprocess_absolute_eit_frame(df)
    # Add normalizations
    v1 = add_normalizations(v1, NORMALIZE_MEDIAN=NORMALIZE, NORMALIZE_PER_ELECTRODE=False)
    # plt.plot(v1)
    # plt.show()
    PCA = True
    if PCA:
        v1 = pca.transform(v1.reshape(1, -1))
        if plot:
            plt.bar(x=range(len(v1.reshape(-1))), height=v1.reshape(-1))
            plt.title("PCA transformed voltage vector")
            plt.xlabel("PCA component")
            plt.ylabel("Intensity")
            plt.show()
    img = solve_and_plot(model=model_pca, model_input=v1, chow_center_of_mass=False,
                         use_opencv_for_plotting=True)

    # save the video to a folder
    if save_video:
        if not os.path.exists("eit_video"):
            os.mkdir("eit_video")
        img = img * 255
        # clip the values to 0-255
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img_path = os.path.join("eit_video", f"{time.time()}.png")
        # print(img_path)
        cv2.imwrite(img_path, cv2.resize(img, (512, 512)))

    # img, center = solve_and_get_center(model=model_pca, model_input=v1)
    # cv2.imshow("img", cv2.resize(img, (512, 512)))
    # cv2.waitKey(1)
    # return img, center
    # time.sleep(3)


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
                plot_multi_frequency_eit_image(os.path.join(eit_path, current_frame))
                # centers.append(center)
                seen_files.append(current_frame)
                # last 10 centers
                # for c in centers[-10:]:
                #     # add circle to image to show center
                #     cv2.circle(empty_img, (int(c[0]), int(c[1])), 1, (255, 255, 255), -1)
                #     cv2.imshow("center", cv2.resize(empty_img, (512, 512)))


def convert_pngs_in_folder_to_video(path):
    """
    Converts the pngs in the given folder to a mp4 video.
    :param path:
    :return:
    """
    img_array = []
    # sort the files by date
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(path, filename))
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    out = cv2.VideoWriter(os.path.join(path, "eit_video.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':

    ### Settings ###
    path = "test"
    VOLTAGE_VECTOR_LENGTH = 1024
    VOLTAGE_VECTOR_LENGTH_PCA = 128
    OUT_SIZE = 64
    # Normalize the data
    NORMALIZE = True

    print("Loading the model")
    ### Settings end ###

    model_pca = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH_PCA, output_size=OUT_SIZE ** 2)
    # model_pca_path = "Collectad_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies/Models/LinearModelWithDropout2/run_with_data_after_rebuild_of_setup3/model_2023-09-29_11-22-13_399_400.pth"

    model_pca_path = "Collected_Data_Variation_Experiments/High_Variation_multi/Models/LinearModelWithDropout2/Test_Run/model_2023-10-06_12-15-26_epoche_143_of_300_best_model.pth"

    # model_pca_path = "Collectad_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies/Models/LinearModelWithDropout2/Run_05_10_3629_samples_with_augmentation/model_2023-10-05_18-13-21_epoche_124_of_300_best_model.pth"
    # get the pca.okl in the same folder as the model
    pca_path = os.path.join(os.path.dirname(model_pca_path), "pca.pkl")
    pca = pickle.load(open(pca_path, "rb"))
    model_pca.load_state_dict(torch.load(model_pca_path))

    model_pca.eval()
    try:
        plot_eit_video(path)
    except RuntimeError as e:
        if str(e) == "mat1 and mat2 shapes cannot be multiplied (128x1 and 128x128)":
            logging.warning("comment out model plot utils line 160")
            logging.warning("Problem with Batch Norm Modles")

    # convert_pngs_in_folder_to_video("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\test\\3_freq_move_Target\\setup_1\eit_video")
