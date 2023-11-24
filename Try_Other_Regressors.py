import copy
import os.path
import time
from datetime import datetime
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import numpy
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from PIL import Image

from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from Model_Training.EarlyStoppingHandler import EarlyStoppingHandler
from Model_Training.data_augmentation import add_noise_augmentation, add_rotation_augmentation, add_gaussian_blur
from Model_Training.dimensionality_reduction import perform_pca_on_input_data
from plot_utils import GridSearch_table_plot
from utils import add_normalizations

import pickle

from joblib import dump, load

LOSS_SCALE_FACTOR = 1000
# VOLTAGE_VECTOR_LENGTH = 6144
VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64

plt.rcParams.update({'font.size': 22})


def prepare_training_data(path, add_augmentation, normalize, pca_components=0, test_size=0.2):
    """
    Loads the training data from the given path and returns it as a tuple of numpy arrays
    :param path: path to the training data
    :param add_augmentation: if True, data augmentation is applied
    :param normalize: if True, the data is normalized
    :param pca_components: if > 0, PCA is applied to the data and the number of components is reduced to the given value
    :return: tuple of numpy arrays, split into train and test data
    """
    global VOLTAGE_VECTOR_LENGTH, pca
    if pca_components > 0:
        VOLTAGE_VECTOR_LENGTH = pca_components

    USE_DIFF_DIRECTLY = False
    if os.path.exists(os.path.join(path, "v1_array.npy")):
        voltage_data_np = np.load(os.path.join(path, "v1_array.npy"))
        print("INFO: Using v1 voltages and calculating voltage differences with one v0 as reference")
    else:
        try:
            voltage_data_np = np.load(os.path.join(path, "voltage_diff_array.npy"))
            print("INFO: Using voltage differences directly")
            USE_DIFF_DIRECTLY = True
        except FileNotFoundError:
            raise Exception("No voltage data found")
    image_data_np = np.load(os.path.join(path, "img_array.npy"))
    print(f"INFO: Loaded {len(image_data_np)} images and {len(voltage_data_np)} voltage vectors")
    # reduce the number of images
    # image_data_np = image_data_np[:200]
    # voltage_data_np = voltage_data_np[:200]
    # Highlight Step 1: In case of time difference EIT, we need to normalize the data with v0
    if not ABSOLUTE_EIT:
        if not USE_DIFF_DIRECTLY:
            print("INFO: Single frequency EIT data is used. Normalizing the data with v0")
            v0 = np.load(os.path.join(path, "v0.npy"))
            # v0 = np.load("../ScioSpec_EIT_Device/v0.npy")
            # normalize the voltage data
            voltage_data_np = (voltage_data_np - v0) / v0  # normalized voltage difference
            # Now the model should learn the difference between the voltages and v0 (default state)
        else:
            print("INFO: Single frequency EIT data is used. Using voltage differences directly")
    # Now the model should learn the difference between the voltages and v0 (default state)
    # Highlight Step 2: Preprocess the data (independent if it is absolute or difference EIT)
    voltage_data_np = add_normalizations(v1=voltage_data_np, NORMALIZE_MEDIAN=normalize,
                                         NORMALIZE_PER_ELECTRODE=False)
    print("Overall data shape: ", voltage_data_np.shape)
    trainX, testX, trainY, testY = train_test_split(voltage_data_np, image_data_np, test_size=test_size,
                                                    random_state=42)
    # Highlight Step 4.1: Augment the training data
    if add_augmentation:
        # augment the training data
        print("INFO: Adding noise augmentation")
        trainX, trainY = add_noise_augmentation(trainX, trainY,
                                                number_of_noise_augmentations, noise_level, device="cpu")
        print("INFO: Adding rotation augmentation")
        trainX, trainY = add_rotation_augmentation(trainX, trainY,
                                                   number_of_rotation_augmentations, device="cpu")

        trainY = add_gaussian_blur(trainY, device="cpu", nr_of_blurs=number_of_blur_augmentations)
    # Highlight Step4.2 Do PCA to reduce the number of input features
    if pca_components > 0:
        print("INFO: Performing PCA on input data")
        trainX, testX, _, pca = perform_pca_on_input_data(voltage_data_np, image_data_np, trainX,
                                                          testX, testX, f"{path}/Models",
                                                          "CPU",
                                                          debug=False,
                                                          train_images=image_data_np,
                                                          n_components=pca_components)
    # Highlight: flatten the images
    trainY = trainY.reshape(trainY.shape[0], -1)
    testY = testY.reshape(testY.shape[0], -1)
    print("TrainX shape: ", trainX.shape)
    print("TrainY shape: ", trainY.shape)
    print("TestX shape: ", testX.shape)
    print("TestY shape: ", testY.shape)
    return testX, testY, trainX, trainY, pca, voltage_data_np


testX, testY, trainX, trainY, pca = None, None, None, None, None


def train_regressor(model_name: str, regressor, path_to_training_data: str,
                    normalize=True, add_augmentation=False, results_folder="Results",
                    pca_components=0, ):
    global VOLTAGE_VECTOR_LENGTH, OUT_SIZE, testX, testY, trainX, trainY, pca

    if testX is None:
        print("INFO: Preparing training data")
        testX, testY, trainX, trainY, pca, _ = prepare_training_data(path_to_training_data, add_augmentation, normalize,
                                                                     pca_components=pca_components)
    else:
        print("INFO: Using cached training data")
    mean = trainY.mean()
    print("Mean: ", mean)
    regressor.fit(trainX, trainY - mean)
    print(f"Train score of {model_name}: ", regressor.score(trainX, trainY - mean))
    print(f"Test score of {model_name}: ", regressor.score(testX, testY - mean))
    print("OK")
    results_path = f"{results_folder}/{model_name}"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # save the model as pickle file
    pickle.dump(regressor, open(f"{results_path}/model.pkl", 'wb'))
    # save the pca as pickle file
    if pca_components > 0:
        pickle.dump(pca, open(f"{results_path}/pca.pkl", 'wb'))
    else:
        # if an pca dile already exists, delete it
        if os.path.exists(f"{results_path}/pca.pkl"):
            print("INFO: Deleting old pca.pkl file")
            os.remove(f"{results_path}/pca.pkl")

    new_flat_pictures = regressor.predict(testX) + mean
    # only use the first 10 pictures
    new_flat_pictures = new_flat_pictures[:12]
    testY_selection = testY[:12]
    plt.plot(testX[0])
    plt.title("Input data example")
    plt.show()
    for picture, testY_sample in zip(new_flat_pictures, testY_selection):
        plt.figure(figsize=[20, 10])
        plt.subplot(121)
        plt.imshow(testY_sample.reshape(OUT_SIZE, OUT_SIZE), cmap='viridis')
        plt.title("Target")
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(picture.reshape(OUT_SIZE, OUT_SIZE), cmap='viridis')
        plt.title(model_name)
        plt.colorbar()
        plt.savefig(f"{results_path}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        plt.show()


from sklearn.model_selection import GridSearchCV


def hyperparameter_tuning():
    global VOLTAGE_VECTOR_LENGTH, OUT_SIZE, testX, testY, trainX, trainY, pca, add_augmentations
    print("INFO: Preparing training data")
    testX, testY, trainX, trainY, pca, voltage_data_np = prepare_training_data(path, add_augmentation=
    add_augmentations, normalize=True,
                                                                               pca_components=pca_components,
                                                                               test_size=0.05)
    voltage_data_df = pd.DataFrame(voltage_data_np)

    # set random 20% of the data to be test data
    split_index = [-1 if random() < 0.2 else 0 for x in voltage_data_df.index]

    # Use the list to create PredefinedSplit
    pds = PredefinedSplit(test_fold=split_index)
    mean = trainY.mean()
    print("Mean: ", mean)
    regressor = KNeighborsRegressor()
    parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    reg = GridSearchCV(regressor, parameters, verbose=10, cv=pds
                       , scoring='neg_mean_squared_error')

    reg.fit(trainX, trainY - mean)
    GridSearch_table_plot(reg, "n_neighbors", negative=False, display_all_params=False)

    print(reg.best_params_)
    print(reg.best_score_)


if __name__ == "__main__":
    ABSOLUTE_EIT = True
    # path = "Own_Simulation_Dataset"
    # path = "../Collected_Data_Variation_Experiments/High_Variation_multi"
    # path = "../Collected_Data/Combined_dataset"
    # path = "Training_Data/1_Freq_with_individual_v0s"
    # path = "Training_Data/1_Freq"
    # path = "Training_Data/1_Freq_After_16_10"
    # path = "Training_Data/3_Freq"
    # path = "Trainings_Data_EIT32/1_Freq_More_Orientations"
    path = "Trainings_Data_EIT32/3_Freq"
    pca_components = 128  # 0 means no pca
    noise_level = 0.05
    number_of_noise_augmentations = 5
    number_of_rotation_augmentations = 0
    number_of_blur_augmentations = 5
    add_augmentations = True
    results_folder = "Results_Traditional_Models_AbsoluteEIT" if ABSOLUTE_EIT else "Results_Traditional_Models_TDEIT"
    # hyperparameter_tuning()
    regressors = [
        LinearRegression(),
        # Ridge(alpha=1),
        # Lasso(alpha=0.001, tol=0.01),
        KNeighborsRegressor(n_neighbors=4),
        # DecisionTreeRegressor(max_depth=80),
        # RandomForestRegressor(max_depth=40, n_estimators=20),
        # GradientBoostingRegressor(),
        # AdaBoostRegressor(),
        # BaggingRegressor(),
        # pickle.load(open("Results_Traditional_Models_AbsoluteEIT/LinearRegression/model.pkl", 'rb')),
    ]
    for regressor in regressors:
        model_name = regressor.__class__.__name__
        print("Training with regressor: ", regressor.__class__.__name__)
        start_time = time.time()
        train_regressor(model_name=model_name, regressor=regressor, path_to_training_data=path, normalize=False,
                        add_augmentation=add_augmentations, results_folder=results_folder,
                        pca_components=pca_components)
        print("Training took: ", time.time() - start_time)
        time.sleep(5)
