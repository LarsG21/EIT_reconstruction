import os
import pickle
import time

import cv2
import pandas as pd
import torch
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

from pyeit import mesh
from pyeit.eit import protocol
from reconstruction_algorithims import solve_and_plot_greit
from utils import find_center_of_mass, add_normalizations, check_settings_of_model

from Model_Training.Models import LinearModelWithDropout2, LinearModel
from plot_utils import solve_and_get_center_with_nural_network, preprocess_greit_img
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr


pca = None

mesh_obj = mesh.create(32, h0=0.1)
n_el = 32  # nb of electrodes
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")


def get_position_error(img_reconstructed, target_image, show_plot=True):
    """
    Calculates the position error between the center of mass of the reconstructed image and the target position.
    :param img_reconstructed:
    :param target_image:
    :return:
    """
    center_of_mass = find_center_of_mass(img_reconstructed)
    target_position = find_center_of_mass(target_image)
    error_vect = center_of_mass - target_position
    distance_between_centers = np.linalg.norm(center_of_mass - target_position)

    # plot a plot with two dots at the center of mass and the target position
    if show_plot:
        if USE_OPENCV_FOR_PLOTTING:
            img_reconstructed = cv2.cvtColor(img_reconstructed.astype(np.float32), cv2.COLOR_GRAY2BGR)
            img_reconstructed = cv2.resize(img_reconstructed, (0, 0), fx=4, fy=4)
            center_of_mass = center_of_mass * 4
            target_position = target_position * 4
            cv2.circle(img_reconstructed, (int(center_of_mass[0]), int(center_of_mass[1])), 2, (0, 0, 255), 2)
            cv2.circle(img_reconstructed, (int(target_position[0]), int(target_position[1])), 2, (255, 0, 0), 2)
            cv2.imshow("Position Error", img_reconstructed)
        else:
            plt.imshow(img_reconstructed)
            plt.scatter(center_of_mass[0], center_of_mass[1], c="red")
            plt.scatter(target_position[0], target_position[1], c="blue")
            plt.show()
    # divide by radius of the tank to get the relative position error
    # error_vect = error_vect / (img_reconstructed.shape[0] / 2)
    # distance_between_centers = distance_between_centers / (img_reconstructed.shape[0] / 2)
    print("Distance between centers: ", distance_between_centers)
    print("Error vector: ", error_vect)
    return distance_between_centers, error_vect


def get_amplitude_response(img_reconstructed, target_image, show_plot=True):
    """
    Calculates the amplitude response of the reconstructed image.
    Amplitude response is defined as the ratio of the pixels in the intersection of the reconstructed image
    and the theoretical image to the pixels in the theoretical image.
    :param img_reconstructed:
    :param target_image:
    :return:
    """
    global USE_OPENCV_FOR_PLOTTING
    intersection = np.logical_and(img_reconstructed, target_image)
    intersection = intersection.astype(np.float32)
    # get reconstructed image at parts of the intersection as mask
    img_reconstructed_intersect = img_reconstructed * intersection
    # calculate the ratio of the pixels in the intersection to the pixels in the theoretical image
    amplitude_response = np.sum(img_reconstructed_intersect) / np.sum(target_image)
    if show_plot:
        if USE_OPENCV_FOR_PLOTTING:
            cv2.imshow("Amplitude Response", cv2.resize(img_reconstructed_intersect, (512, 512)))
        else:
            plt.imshow(img_reconstructed_intersect)
            plt.title(f"Amplitude response: {np.round(amplitude_response, 2)}")
            plt.show()
    return amplitude_response


def get_shape_deformation(img_reconstructed, show_plot=True):
    """
    Calculates the shape deformation of the reconstructed image.
    Shape deformation is defined as the fraction of the reconstructed one-fourth amplitude set which
    does not fit within a circle of an equal area.
    :param img_reconstructed: the reconstructed image
    :return:
    """
    global USE_OPENCV_FOR_PLOTTING
    RADIUS_TARGET_IN_MM = 40
    RADIUS_TANK_IN_MM = 190
    RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM
    # 1. calculate center of mass of reconstructed image
    center_of_mass = find_center_of_mass(img_reconstructed)
    img_ideal_circle = np.zeros_like(img_reconstructed)
    # 2. Put a circle with the same area as the anomaly in the center of mass
    radius = int((img_reconstructed.shape[0] * RELATIVE_RADIUS_TARGET / 2))
    cv2.circle(img_ideal_circle, center_of_mass, radius, 1, -1)
    # 3. calculate the difference of the two images
    img_reconstructed_masked = img_reconstructed.copy()
    img_reconstructed_masked[img_ideal_circle == 1] = 0
    img_reconstructed_masked.astype(np.float32)
    # SD measures the fraction of the reconstructed one-fourth amplitude set which
    # does not fit within a circle of an equal area
    shape_deformation = np.sum(img_reconstructed_masked) / np.sum(img_reconstructed)
    if show_plot:
        if USE_OPENCV_FOR_PLOTTING:
            cv2.imshow("Shape Deformation", cv2.resize(img_reconstructed_masked, (512, 512)))
        else:
            plt.imshow(img_reconstructed_masked)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"Shape deformation: {np.sum(img_reconstructed_masked) / np.sum(img_reconstructed)}")
            plt.show()

    return shape_deformation


def evaluate_reconstruction_model(ABSOLUTE_EIT, NORMALIZE, SHOW, df_test_set, v0=None, model=None, model_path=None,
                                  pca=None, regressor=None, debug=False):
    """

    :param ABSOLUTE_EIT: Whether to use absolute EIT or not
    :param NORMALIZE: Whether to normalize the input data or not
    :param SHOW: Whether to show the plots or not (for debugging)
    :param df_test_set: A dataframe containing the test dataset
    :param v0: v0 vector
    :param model: a pytorch model
    :param model_path: path to the model
    :param pca: Principal Component Analysis object
    :param regressor: a regressor for image reconstruction to use instead of the pytorch model
    :param debug:
    :return: a dataframe containing the evaluation results
    """
    positions = []  # position of the anomaly
    position_errors = []  # distance between the center of mass of the reconstructed image and the target position
    error_vectors = []  # vector from the center of mass of the reconstructed image to the target position
    amplitude_responses = []  # amplitude response of the reconstructed image
    shape_deformations = []  # shape deformation of the reconstructed image
    pearson_correlations = []  # cross correlation of the reconstructed image
    mean = df_test_set["images"].mean().flatten()
    ringings = []  # ringing of the reconstructed image
    print(f"Length of dataframe: {len(df_test_set)}")
    if regressor is not None:
        print(f"USING REGRESSOR: {regressor.__class__.__name__} for reconstruction")
    else:
        print(f"USING MODEL: {model.__class__.__name__} for reconstruction")
    for i, row in df_test_set.iterrows():
        raw_voltages = row["voltages"]
        target_image = row["images"]
        target_position = find_center_of_mass(target_image)
        positions.append(target_position)
        v1 = raw_voltages
        if not ABSOLUTE_EIT:
            # calculate the normalized voltage difference
            v1 = (v1 - v0) / v0
        v1 = add_normalizations(v1, NORMALIZE_MEDIAN=NORMALIZE, NORMALIZE_PER_ELECTRODE=False)
        if debug:
            plt.plot(v1)
            plt.title("v1 normalized")
            plt.show()
        if pca is not None:
            print("Transforming with PCA")
            v1 = pca.transform(v1.reshape(1, -1))
            if debug:
                v1_plot = v1.flatten()
                plt.bar(range(len(v1_plot)), v1_plot)
                plt.title("v1 pca")
                plt.show()
        if regressor is None and model is not None:
            img_reconstructed, _, img_non_thresh = solve_and_get_center_with_nural_network(model=model,
                                                                                           model_input=v1)
        else:
            v1 = v1.reshape(1, -1)
            new_flat_picture = regressor.predict(v1)
            if type(regressor) == LinearRegression:  # Subtract the mean for linear regression
                new_flat_picture = new_flat_picture - mean
            # set mode to 0
            mode = stats.mode(new_flat_picture)[0][0]
            new_flat_picture[new_flat_picture == mode] = 0
            img_non_thresh = new_flat_picture.reshape(OUT_SIZE, OUT_SIZE)
            # plt.imshow(img_non_thresh)
            # plt.title("Reconstructed image")
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.show()
            img_reconstructed = img_non_thresh.copy()
            img_reconstructed[img_non_thresh < 0.25] = 0
            # set smaller than 0.2 but bigger than 0 to 0
            # img_reconstructed[np.logical_and(img_reconstructed < 0.2, img_reconstructed > 0)] = 0
        ######################## Ringing #################################
        # Ringing is the sum of all negative values in the image devided by the sum of |all values| in the image
        ringing = - np.sum(img_non_thresh[img_non_thresh < 0]) / np.sum(np.abs(img_non_thresh))
        print(f"Ringing: {ringing}")
        ringings.append(ringing)

        ####################### Amplitude response #######################
        amplitude_response = get_amplitude_response(img_reconstructed, target_image, show_plot=SHOW)
        amplitude_responses.append(amplitude_response)
        print(f"Amplitude response: {amplitude_response}")
        ####################### Position error #######################
        if amplitude_response == 0:
            print("Amplitude response was 0, so the position error is set to NaN")
            distance_between_centers = np.NAN
            error_vect = np.NAN
        else:
            distance_between_centers, error_vect = get_position_error(img_reconstructed, target_image, show_plot=SHOW)
        position_errors.append(distance_between_centers)
        error_vectors.append(error_vect)
        ####################### Shape deformation #######################
        shape_deformation = get_shape_deformation(img_reconstructed, show_plot=SHOW)
        shape_deformations.append(shape_deformation)
        corr = pearsonr(img_reconstructed.flatten(), target_image.flatten())[0]
        pearson_correlations.append(corr)
        print(f"Pearson correlation: {corr}")
        if SHOW and USE_OPENCV_FOR_PLOTTING:
            cv2.waitKey(100)
    df = pd.DataFrame(
        data={"positions": positions, "position_error": position_errors, "error_vector": error_vectors,
              "amplitude_response": amplitude_responses, "shape_deformation": shape_deformations,
              "ringing": ringings, "pearson_correlation": pearson_correlations})
    path = "Results"
    if regressor is None:
        eval_df_name = f"evaluation_model_{model_path.split('/')[-1].split('.')[0]}.pkl"
    else:
        eval_df_name = f"evaluation_regressor_{regressor.__class__.__name__}.pkl"

    # eval_df_name = "TESTING.pickle"
    save_path = os.path.join(path, eval_df_name)
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
    df.to_pickle(save_path)
    print(f"saved dataframe to {save_path}")
    print("Use Plot_results_of_evaluation.py to evaluate the results")
    # print average of all metrics
    print(f"Average position error: {np.nanmean(position_errors)}")
    print(f"Average amplitude response: {np.nanmean(amplitude_responses)}")
    print(f"Average shape deformation: {np.nanmean(shape_deformations)}")
    print(f"Average ringing: {np.nanmean(ringings)}")
    print(f"Average pearson correlation: {np.nanmean(pearson_correlations)}")
    return df


### Setings ###
ABSOLUTE_EIT = True
OUT_SIZE = 64
VOLTAGE_VECTOR_LENGTH = 1024
NORMALIZE = True
USE_OPENCV_FOR_PLOTTING = True


### Setings ###

def main():
    global pca, NORMALIZE, ABSOLUTE_EIT, v0, VOLTAGE_VECTOR_LENGTH
    ####### Settings #######
    SHOW = True
    print("Loading the model")
    # Working Examples:
    # model_path = "../Collected_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies/Models/LinearModelWithDropout2/Run_12_10_with_normalization/model_2023-10-12_14-45-50_epoche_263_of_300_best_model.pth"
    # model_path = "../Trainings_Data_EIT32/1_Freq_More_Orientations/Models/LinearModelWithDropout2/TESTING/model_2023-11-27_17-24-18_99_100.pth"
    # model_path = "../Collected_Data/Combined_dataset/Models/LinearModelWithDropout2/TESTING_MORE_DATA_12_10/model_2023-10-12_11-55-44_epoche_232_of_300_best_model.pth"
    # model_path = "../Training_Data/3_Freq/Models/LinearModelWithDropout2/Run_16_12/model_2023-10-16_13-23-43_143_300.pth"
    # New Path
    # model_path = "../Training_Data/1_Freq_After_16_10/Models/LinearModelWithDropout2/Run_23_10_with_augment_more_negative_set/model_2023-10-23_15-02-47_149_150.pth"
    # model_path = "../Training_Data/1_Freq_with_individual_v0s/Models/LinearModelWithDropout2/Run_25_10_dataset_individual_v0s/model_2023-10-27_14-25-23_148_150.pth"
    # model_path = "../Training_Data/1_Freq_with_individual_v0s/Models/LinearModelWithDropout2/Run_06_11_with_blurr/model_2023-11-06_16-45-47_85_200.pth"
    # model_path = "../Training_Data/1_Freq_with_individual_v0s/Models/LinearModel/Few_Data_Test_5x_noise_aug_5x_rot_aug/model_2023-11-15_16-36-57_152_200.pth"
    model_path = "../Trainings_Data_EIT32/3_Freq_Even_orientation/Models/LinearModelWithDropout2/Test_Superposition_2/model_2023-12-13_13-37-55_69_70.pth"
    # load v0 from the same folder as the model
    # move up 4 directories up, then go to the v0.npy file
    # v0 = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(model_path)))),
    #                           "v0.npy"))

    # regressor_path = "../Results_Traditional_Models_TDEIT/LinearRegression/model.pkl"
    regressor_path = "../Trainings_Data_EIT32/3_Freq_Even_orientation_and_GREIT_data/Models/KNeighborsRegressor/KNeighborsRegressor.pkl"
    # regressor = None
    regressor = pickle.load(open(regressor_path, 'rb'))

    #### END Settings #######

    if regressor is None:
        pca_path = os.path.join(os.path.dirname(model_path), "pca.pkl")
    else:
        pca_path = os.path.join(os.path.dirname(regressor_path), "pca.pkl")

    # load pca if it exists
    if os.path.exists(pca_path):
        print("Loading PCA")
        pca = pickle.load(open(pca_path, "rb"))
        VOLTAGE_VECTOR_LENGTH = pca.n_components_
        input("Press Enter to continue...")

    # Choose the correct test set and set the voltage vector length
    if ABSOLUTE_EIT:
        # test_set_path = "../Test_Data/Test_Set_Circular_16_10_3_freq/combined.pkl"
        test_set_path = "../Test_Data_EIT32/3_Freq/Test_set_circular_24_11_3_freq_40mm_eit32_orientation25_2/combined.pkl"
        print(f"INFO: Setting Voltage_vector_length to {VOLTAGE_VECTOR_LENGTH}")
    else:
        # test_set_path = "../Test_Data/Test_Set_1_Freq_23_10_circular/combined.pkl"
        # test_set_path = "../Test_Data/Test_Set_Circular_single_freq/combined.pkl"
        test_set_path = "../Test_Data_EIT32/1_Freq/Test_set_circular_10_11_1_freq_40mm/combined.pkl"
        print(f"INFO: Setting Voltage_vector_length to {VOLTAGE_VECTOR_LENGTH}")

    if regressor is None:  # Use the nn model
        # check if the settings.txt file is in the same folder as the model
        norm, absolute = check_settings_of_model(model_path)
        if norm is not None and norm != NORMALIZE:
            print(f"INFO: Setting NORMALIZE to {norm} like in the settings.txt file")
            NORMALIZE = norm
        if absolute is not None and absolute != ABSOLUTE_EIT:
            print(f"INFO: Setting ABSOLUTE_EIT to {absolute} like in the settings.txt file")
            ABSOLUTE_EIT = absolute
        model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
        # model = LinearModel(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        input(f"Using regressor {regressor.__class__.__name__} for the reconstruction. \n"
              "Press Enter to continue...")
        model = None

    df_test_set = pd.read_pickle(test_set_path)
    if not ABSOLUTE_EIT:
        # load v0 from the same folder as the test set
        v0 = np.load(os.path.join(os.path.dirname(test_set_path), "v0.npy"))
    else:
        v0 = None

    input(f"ABSOLUTE_EIT: {ABSOLUTE_EIT} \nVOLTAGE_VECTOR_LENGTH: {VOLTAGE_VECTOR_LENGTH} \n"
          f"OUT_SIZE: {OUT_SIZE} \nNORMALIZE: {NORMALIZE} \nUSE_OPENCV_FOR_PLOTTING: {USE_OPENCV_FOR_PLOTTING} \n"
          f"Press Enter to continue...")
    evaluate_reconstruction_model(ABSOLUTE_EIT, NORMALIZE, SHOW, df_test_set, v0,
                                  model=model, model_path=model_path, pca=pca, regressor=regressor,
                                  debug=False)



if __name__ == '__main__':
    main()
