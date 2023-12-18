import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df
from plot_utils import solve_and_plot_with_nural_network, solve_and_get_center_with_nural_network
from utils import wait_for_start_of_measurement, preprocess_absolute_eit_frame, add_normalizations, \
    load_model_from_path


def plot_multi_frequency_eit_image(v1_path, debug_plot=False, save_video=False):
    """
    Plots the eit image from the given .eit frame file.
    :param v1_path:  The path to the .eit frame file
    :param debug_plot: Whether to plot additional debug plots
    :param save_video: Whether to save the video to a folder
    :return:
    """
    df = convert_multi_frequency_eit_to_df(v1_path)
    # Convert to an numpy array with alternating real and imag numbers
    v1 = preprocess_absolute_eit_frame(df)
    # Add normalizations
    # plt.plot(v1)
    # plt.show()
    images = {}
    for i, (title, model_temp, pca_temp, normalize_temp) in enumerate(
            zip(title_list, model_list, pca_list, NORMALIZE_LIST)):
        if normalize_temp:
            v1 = add_normalizations(v1, NORMALIZE_MEDIAN=True, NORMALIZE_PER_ELECTRODE=False)
        v1_pca = pca_temp.transform(v1.reshape(1, -1))
        if debug_plot:
            plt.bar(x=range(len(v1_pca.reshape(-1))), height=v1_pca.reshape(-1))
            plt.title("PCA transformed voltage vector")
            plt.xlabel("PCA component")
            plt.ylabel("Intensity")
            plt.show()
        # solve_and_get_center_with_nural_network(model=model_temp, model_input=v1_pca, debug=True)
        img = solve_and_plot_with_nural_network(model=model_temp, model_input=v1_pca, chow_center_of_mass=False,
                                                use_opencv_for_plotting=True
                                                , title=title,
                                                save_path=None)
        # save a screenshot if s is pressed
        images[title] = img
    if cv2.waitKey(1) & 0xFF == ord('s'):
        for title, img in images.items():
            plt.imshow(img)
            plt.tight_layout()
            plt.colorbar(fraction=0.046, pad=0.04)
            save_path = f"C:\\Users\\lgudjons\\Desktop\\{title}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_filename = f"{title}.pdf"
            save_filename = os.path.join(save_path, save_filename)
            if os.path.exists(save_filename):
                save_filename = os.path.join(save_path, f"{title}_{time.time()}.pdf")
            plt.imsave(save_filename, img)
            plt.show()
        print("saved")

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
    start_time = time.time()
    while True:
        for current_frame in os.listdir(os.getcwd()):
            if time.time() - start_time > 1:
                print("FPS: ", len(seen_files) / (time.time() - start_time))
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
    path = "C:\\Users\\lgudjons\\Desktop\\eit_data"
    VOLTAGE_VECTOR_LENGTH = 0
    OUT_SIZE = 64
    # Normalize the data
    NORMALIZE = True

    ### Settings end ###

    # model_pca_path = "Trainings_Data_EIT32/3_Freq_Even_orientation/Models/LinearModelWithDropout2/Test_Superposition_2/model_2023-12-13_13-37-55_69_70.pth"
    #
    # model_path_2 = "Trainings_Data_EIT32/3_Freq_Even_orientation/Models/LinearModelWithDropout2/Test_without_superposition/model_2023-12-13_14-17-56_69_70.pth"
    #
    # model_path_3 = "Trainings_Data_EIT32/3_Freq_Even_orientation_and_GREIT_data/Models/LinearModelWithDropout2/More_Superpositions/model_2023-12-14_14-46-52_99_100.pth"

    # model_pca_path = "Trainings_Data_EIT32/3_Freq_Even_orientation_and_GREIT_data/Models/LinearModelWithDropout2/No_Superpositions/model_2023-12-14_15-46-57_99_100.pth"
    model_path_2 = "Trainings_Data_EIT32/3_Freq_Even_orientation_and_GREIT_data/Models/LinearModelWithDropout2/More_Superpositions/model_2023-12-14_14-46-52_99_100.pth"
    # model_path_3 = "Trainings_Data_EIT32/3_Freq_Even_orientation_and_GREIT_data/Models/LinearModelWithDropout2/Model_16_12_many_augmentations_GPU_3/continued_model_2023-12-17_12-24-19_42_60.pth"
    model_paths = [
        # model_pca_path,
        model_path_2,
        # model_path_3
    ]
    model_list = []
    pca_list = []
    NORMALIZE_LIST = []
    for model_path in model_paths:
        model, pca, NORMALIZE = load_model_from_path(path=model_path, normalize=NORMALIZE)
        model_list.append(model)
        pca_list.append(pca)
        NORMALIZE_LIST.append(NORMALIZE)

    title_list = [
        # "Model without superposition",
        "Model with superposition",
        # "New Model GPU"
                  ]

    try:
        plot_eit_video(path)
    except RuntimeError as e:
        if str(e) == "mat1 and mat2 shapes cannot be multiplied (128x1 and 128x128)":
            logging.warning("comment out line 160 in model_plot_utils ")
            logging.warning("Problem with Batch Norm Modles")

    # convert_pngs_in_folder_to_video("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\test\\3_freq_move_Target\\setup_1\eit_video")
