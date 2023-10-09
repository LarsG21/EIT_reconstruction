import os
import pickle

import cv2
import pandas as pd
import torch
import numpy as np
from utils import find_center_of_mass, add_normalizations

from Model_Training.Models import LinearModelWithDropout2
from plot_utils import solve_and_get_center_with_nural_network
import matplotlib.pyplot as plt

### Setings ###
MULTI_FREQUENCY = True
VOLTAGE_VECTOR_LENGTH = 128
OUT_SIZE = 64
NORMALIZE = True


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
    # if show_plot or distance_between_centers > 10:
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
    intersection = np.logical_and(img_reconstructed, target_image)
    intersection = intersection.astype(np.float32)
    # get reconstructed image at parts of the intersection as mask
    img_reconstructed_intersect = img_reconstructed * intersection
    # calculate the ratio of the pixels in the intersection to the pixels in the theoretical image
    amplitude_response = np.sum(img_reconstructed_intersect) / np.sum(target_image)
    if show_plot:
    # if show_plot or amplitude_response > 0.9:
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
    # if show_plot or shape_deformation > 0.4:
        # plt.imshow(img_reconstructed)
        # plt.show()
        plt.imshow(img_reconstructed_masked)
        plt.title(f"Shape deformation: {np.sum(img_reconstructed_masked) / np.sum(img_reconstructed)}")
        plt.show()

    return shape_deformation


def main():
    ####### Settings #######
    SHOW = False
    print("Loading the model")
    model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
    model_path = "../Collected_Data_Variation_Experiments/High_Variation_multi/Models/LinearModelWithDropout2/Test_Run/model_2023-10-06_12-15-26_epoche_143_of_300_best_model.pth"
    model.load_state_dict(torch.load(model_path))
    pca_path = os.path.join(os.path.dirname(model_path), "pca.pkl")
    pca = pickle.load(open(pca_path, "rb"))

    df = pd.read_pickle("../Collected_Data/Test_Set_Circular_06_10/combined.pkl")
    #### END Settings #######

    positions = []  # position of the anomaly
    position_errors = []  # distance between the center of mass of the reconstructed image and the target position
    error_vectors = []  # vector from the center of mass of the reconstructed image to the target position
    amplitude_responses = []  # amplitude response of the reconstructed image
    shape_deformations = []  # shape deformation of the reconstructed image
    print(f"Length of dataframe: {len(df)}")
    for i, row in df.iterrows():
        raw_voltages = row["voltages"]
        target_image = row["images"]
        target_position = find_center_of_mass(target_image)
        positions.append(target_position)
        if SHOW:
            plt.imshow(target_image)
            plt.show()
        v1 = add_normalizations(raw_voltages, NORMALIZE_MEDIAN=NORMALIZE, NORMALIZE_PER_ELECTRODE=False)
        v1 = pca.transform(v1.reshape(1, -1))
        img_reconstructed, center = solve_and_get_center_with_nural_network(model=model, model_input=v1)
        ####################### Position error #######################
        distance_between_centers, error_vect = get_position_error(img_reconstructed, target_image, show_plot=SHOW)
        position_errors.append(distance_between_centers)
        error_vectors.append(error_vect)
        ####################### Amplitude response #######################
        amplitude_response = get_amplitude_response(img_reconstructed, target_image, show_plot=SHOW)
        amplitude_responses.append(amplitude_response)
        ####################### Shape deformation #######################
        shape_deformation = get_shape_deformation(img_reconstructed, show_plot=SHOW)
        shape_deformations.append(shape_deformation)

    df = pd.DataFrame(
        data={"positions": positions, "position_error": position_errors, "error_vector": error_vectors,
              "amplitude_response": amplitude_responses, "shape_deformation": shape_deformations})
    path = "C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results"
    folder_name = model_path.split('/')[-1].split('.')[0]
    eval_df_name = f"TEST_evaluation_model_{model_path.split('/')[-1].split('.')[0]}.pkl"
    # eval_df_name = "TEST.pkl"
    save_path = os.path.join(path, folder_name, eval_df_name)
    if not os.path.exists(os.path.join(path, folder_name)):
        os.makedirs(os.path.join(path, folder_name))
    df.to_pickle(save_path)
    print(f"saved dataframe to {save_path}")
    print("Use Plot_results_of_evaluation.py to evaluate the results")

    # print(f"lenght of dataframe: {len(df)}")

    # number of images with amplitude response > 0.9
    # print(f"Number of images with amplitude response > 0.9: {np.sum(np.array(amplitude_responses) > 0.9)}")
    # print(f"Index of images with amplitude response > 0.9: {np.where(np.array(amplitude_responses) > 0.9)}")
    # # number of images with shape deformation > 0.4
    # print(f"Number of images with shape deformation > 0.4: {np.sum(np.array(shape_deformations) > 0.4)}")
    # print(f"Index of images with shape deformation > 0.4: {np.where(np.array(shape_deformations) > 0.4)}")
    # # number of images with position error > 10
    # print(f"Number of images with position error > 10: {np.sum(np.array(position_errors) > 10)}")
    # print(f"Index of images with position error > 10: {np.where(np.array(position_errors) > 10)}")


if __name__ == '__main__':
    main()
