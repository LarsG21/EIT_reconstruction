import os
import pickle
import time

import cv2
import pandas as pd
import torch
import numpy as np

from pyeit import mesh
from pyeit.eit import protocol
from reconstruction_algorithims import solve_and_plot_greit
from utils import find_center_of_mass, add_normalizations, check_settings_of_model

from Model_Training.Models import LinearModelWithDropout2
from plot_utils import solve_and_get_center_with_nural_network, preprocess_greit_img
import matplotlib.pyplot as plt



pca = None

mesh_obj = mesh.create(32, h0=0.1)
n_el = 32  # nb of electrodes
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")


# TODO: THE COL TARGET POSITION IS SOMETIMES WRONG ! USE  target_position = find_center_of_mass(row["images"]) instead
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
            plt.title(f"Amplitude response: {amplitude_response}")
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
            plt.title(f"Shape deformation: {np.sum(img_reconstructed_masked) / np.sum(img_reconstructed)}")
            plt.show()

    return shape_deformation


### Setings ###
ABSOLUTE_EIT = False
VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64
NORMALIZE = True
USE_OPENCV_FOR_PLOTTING = True
USE_GREIT_FOR_RECONSTRUCTION = False

### Setings ###

def main():
    global pca, NORMALIZE, ABSOLUTE_EIT, v0, VOLTAGE_VECTOR_LENGTH
    input(f"ABSOLUTE_EIT: {ABSOLUTE_EIT} \nVOLTAGE_VECTOR_LENGTH: {VOLTAGE_VECTOR_LENGTH} \n"
          f"OUT_SIZE: {OUT_SIZE} \nNORMALIZE: {NORMALIZE} \nUSE_OPENCV_FOR_PLOTTING: {USE_OPENCV_FOR_PLOTTING} \n"
          f"Press Enter to continue...")
    ####### Settings #######
    SHOW = False
    print("Loading the model")
    model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
    # Working Examples:
    # model_path = "../Collected_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies/Models/LinearModelWithDropout2/Run_12_10_with_normalization/model_2023-10-12_14-45-50_epoche_263_of_300_best_model.pth"
    # model_path = "../Training_Data/1_Freq_After_16_10/Models/LinearModelWithDropout2/Run_23_10_with_augment/model_2023-10-23_13-50-11_122_150.pth"
    # model_path = "../Collected_Data/Combined_dataset/Models/LinearModelWithDropout2/TESTING_MORE_DATA_12_10/model_2023-10-12_11-55-44_epoche_232_of_300_best_model.pth"
    # model_path = "../Training_Data/3_Freq/Models/LinearModelWithDropout2/Run_16_12/model_2023-10-16_13-23-43_143_300.pth"
    # New Path
    # model_path = "../Training_Data/1_Freq_After_16_10/Models/LinearModelWithDropout2/Run_23_10_with_augment_more_negative_set/model_2023-10-23_15-02-47_149_150.pth"
    model_path = "../Training_Data/1_Freq_After_16_10/Models/LinearModelWithDropout2/Run_25_10/model_2023-10-27_13-23-19_112_150.pth"
    model.load_state_dict(torch.load(model_path))
    # load v0 from the same folder as the model
    # move up 4 directories up, then go to the v0.npy file
    # v0 = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(model_path)))),
    #                           "v0.npy"))

    norm, absolute = check_settings_of_model(model_path)
    # if norm is not None and norm != NORMALIZE:
    #     print(f"INFO: Setting NORMALIZE to {norm} like in the settings.txt file")
    #     NORMALIZE = norm
    # if absolute is not None and absolute != ABSOLUTE_EIT:
    #     print(f"INFO: Setting ABSOLUTE_EIT to {absolute} like in the settings.txt file")
    #     ABSOLUTE_EIT = absolute

    # Test set training_data_path
    if ABSOLUTE_EIT:
        test_set_path = "../Test_Data/Test_Set_Circular_16_10_3_freq/combined.pkl"
        VOLTAGE_VECTOR_LENGTH = 128
        print(f"INFO: Setting Voltage_vector_length to {VOLTAGE_VECTOR_LENGTH}")
    else:
        # test_set_path = "../Test_Data/Test_Set_1_Freq_23_10_circular/combined.pkl"
        test_set_path = "../Test_Data/Test_Set_Circular_single_freq/combined.pkl"
        # test_set_path = "../Training_Data/1_Freq/combined.pkl"
        VOLTAGE_VECTOR_LENGTH = 1024
        print(f"INFO: Setting Voltage_vector_length to {VOLTAGE_VECTOR_LENGTH}")
    input("Press Enter to continue...")
    df_test_set = pd.read_pickle(test_set_path)
    # load v0 from the same folder as the test set
    v0 = np.load(os.path.join(os.path.dirname(test_set_path), "v0.npy"))

    #### END Settings #######

    # load a regressor
    regressor_path = "../Results_Traditional_Models_TDEIT/LinearRegression/model.pkl"
    regressor = None
    # regressor = pickle.load(open(regressor_path, 'rb'))
    if regressor is not None:
        input(f"Using regressor {regressor.__class__.__name__} for the reconstruction. \n"
              "Press Enter to continue...")
    pca_path = os.path.join(os.path.dirname(regressor_path), "pca.pkl")
    if os.path.exists(pca_path):
        print("Loading PCA")
        pca = pickle.load(open(pca_path, "rb"))

    evaluate_reconstruction_model(ABSOLUTE_EIT, NORMALIZE, SHOW, df_test_set, model=model, model_path=model_path,
                                  pca=pca,
                                  regressor=regressor)


def evaluate_reconstruction_model(ABSOLUTE_EIT, NORMALIZE, SHOW, df, model=None, model_path=None, pca=None,
                                  regressor=None):
    """

    :param ABSOLUTE_EIT: Whether to use absolute EIT or not
    :param NORMALIZE: Whether to normalize the input data or not
    :param SHOW: Whether to show the plots or not (for debugging)
    :param df: A dataframe containing the test dataset
    :param model: a pytorch model
    :param model_path: path to the model
    :param pca: Principal Component Analysis object
    :param regressor: a regressor for image reconstruction to use instead of the pytorch model
    :return:
    """
    global v0
    positions = []  # position of the anomaly
    position_errors = []  # distance between the center of mass of the reconstructed image and the target position
    error_vectors = []  # vector from the center of mass of the reconstructed image to the target position
    amplitude_responses = []  # amplitude response of the reconstructed image
    shape_deformations = []  # shape deformation of the reconstructed image
    mean = df["images"].mean().flatten()
    print(f"Length of dataframe: {len(df)}")
    if regressor is not None:
        print(f"USING REGRESSOR: {regressor.__class__.__name__} for reconstruction")
    else:
        print(f"USING MODEL: {model.__class__.__name__} for reconstruction")
    for i, row in df.iterrows():
        raw_voltages = row["voltages"]
        target_image = row["images"]
        target_position = find_center_of_mass(target_image)
        positions.append(target_position)
        if not ABSOLUTE_EIT:
            v1 = raw_voltages
            # calculate the normalized voltage difference
            v1 = (v1 - v0) / v0
            # plt.plot(v1)
            # plt.show()
        else:
            v1 = add_normalizations(raw_voltages, NORMALIZE_MEDIAN=NORMALIZE, NORMALIZE_PER_ELECTRODE=False)
        if pca is not None:
            print("Transforming with PCA")
            v1 = pca.transform(v1.reshape(1, -1))
        if not USE_GREIT_FOR_RECONSTRUCTION:
            if regressor is None:
                img_reconstructed, _ = solve_and_get_center_with_nural_network(model=model, model_input=v1)
            else:
                v1 = v1.reshape(1, -1)
                new_flat_picture = regressor.predict(v1) - mean
                img_reconstructed = new_flat_picture.reshape(OUT_SIZE, OUT_SIZE)
                # img_reconstructed[img_reconstructed > 0.2] = 1
                img_reconstructed[img_reconstructed < 0.2] = 0
        ############################### For GREIT  EVALUATION ###############################
        else:
            v0_traditional_algorithims = v0[protocol_obj.keep_ba]
            v1_traditional_algorithims = v1[protocol_obj.keep_ba]
            img_greit = solve_and_plot_greit(v0_traditional_algorithims, v1_traditional_algorithims,
                                             mesh_obj, protocol_obj,
                                             plot=False)
            plt.imshow(img_greit)
            plt.title("GREIT original")
            plt.show()
            img_reconstructed = preprocess_greit_img(img_greit)
            plt.imshow(img_reconstructed)
            plt.title("GREIT preprocessed")
            plt.show()
            time.sleep(0.5)
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
        if SHOW and USE_OPENCV_FOR_PLOTTING:
            cv2.waitKey(300)
    df = pd.DataFrame(
        data={"positions": positions, "position_error": position_errors, "error_vector": error_vectors,
              "amplitude_response": amplitude_responses, "shape_deformation": shape_deformations})
    path = "C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results"
    if not USE_GREIT_FOR_RECONSTRUCTION:
        if regressor is None:
            eval_df_name = f"evaluation_model_{model_path.split('/')[-1].split('.')[0]}.pkl"
        else:
            eval_df_name = f"evaluation_regressor_{regressor.__class__.__name__}.pkl"
    else:
        eval_df_name = f"evaluation_GREIT.pkl"
    # eval_df_name = "TESTING.pickle"
    save_path = os.path.join(path, eval_df_name)
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
    df.to_pickle(save_path)
    print(f"saved dataframe to {save_path}")
    print("Use Plot_results_of_evaluation.py to evaluate the results")


if __name__ == '__main__':
    main()
