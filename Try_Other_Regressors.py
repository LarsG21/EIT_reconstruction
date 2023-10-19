import copy
import os.path
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import numpy
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from PIL import Image

from sklearn.model_selection import train_test_split
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
from Model_Training.data_augmentation import add_noise_augmentation, add_rotation_augmentation
from utils import add_normalizations

import pickle

LOSS_SCALE_FACTOR = 1000
# VOLTAGE_VECTOR_LENGTH = 6144
VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64


# How to use Cuda gtx 1070: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113


def trainings_loop(model_name: str, regressor, path_to_training_data: str,
                   normalize=True, add_augmentation=False, results_folder="Results", ):
    global VOLTAGE_VECTOR_LENGTH

    ######################################################################################
    if pca_components > 0:
        VOLTAGE_VECTOR_LENGTH = pca_components
    #################################
    path = path_to_training_data
    #################################

    voltage_data_np = np.load(os.path.join(path, "v1_array.npy"))
    image_data_np = np.load(os.path.join(path, "img_array.npy"))

    # reduce the number of images
    # image_data_np = image_data_np[:200]
    # voltage_data_np = voltage_data_np[:200]

    # Highlight Step 1: In case of time difference EIT, we need to normalize the data with v0
    if not ABSOLUTE_EIT:
        print("INFO: Single frequency EIT data is used. Normalizing the data with v0")
        v0 = np.load(os.path.join(path, "v0.npy"))
        # v0 = np.load("../ScioSpec_EIT_Device/v0.npy")
        # normalize the voltage data
        voltage_data_np = (voltage_data_np - v0) / v0  # normalized voltage difference
    # Now the model should learn the difference between the voltages and v0 (default state)

    # Highlight Step 2: Preprocess the data (independent if it is absolute or difference EIT)
    voltage_data_np = add_normalizations(v1=voltage_data_np, NORMALIZE_MEDIAN=normalize,
                                         NORMALIZE_PER_ELECTRODE=False)

    print("Overall data shape: ", voltage_data_np.shape)

    trainX, testX, trainY, testY = train_test_split(voltage_data_np, image_data_np, test_size=0.2, random_state=42)

    # Highlight Step 4.1: Augment the training data
    if add_augmentation:
        # augment the training data
        print("INFO: Adding noise augmentation")
        trainX, trainY = add_noise_augmentation(trainX, trainY,
                                                number_of_noise_augmentations, noise_level, device="cpu")
        print("INFO: Adding rotation augmentation")
        trainX, trainY = add_rotation_augmentation(trainX, trainY,
                                                   number_of_rotation_augmentations, device="cpu")

    # flatten the images
    trainY = trainY.reshape(trainY.shape[0], -1)
    testY = testY.reshape(testY.shape[0], -1)

    print("TrainX shape: ", trainX.shape)
    print("TrainY shape: ", trainY.shape)
    print("TestX shape: ", testX.shape)
    print("TestY shape: ", testY.shape)

    mean = trainY.mean()
    regressor.fit(trainX, trainY - mean)
    print(f"Train score of {model_name}: ", regressor.score(trainX, trainY - mean))
    print(f"Test score of {model_name}: ", regressor.score(testX, testY - mean))
    print("OK")
    results_path = f"{results_folder}/{model_name}"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    new_flat_pictures = regressor.predict(testX) + mean
    # only use the first 10 pictures
    new_flat_pictures = new_flat_pictures[:20]
    testY = testY[:20]
    for picture, testY_sample in zip(new_flat_pictures, testY):
        plt.figure(figsize=[20, 10])
        plt.subplot(121)
        plt.imshow(testY_sample.reshape(OUT_SIZE, OUT_SIZE), cmap='viridis')
        plt.subplot(122)
        plt.imshow(picture.reshape(OUT_SIZE, OUT_SIZE), cmap='viridis')
        plt.title(model_name)
        plt.savefig(f"{results_path}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        plt.show()

    # save the model as pickle file
    pickle.dump(regressor, open(f"{results_path}/model.pkl", 'wb'))


if __name__ == "__main__":
    ABSOLUTE_EIT = True
    path = "Training_Data/3_Freq"
    # path = "../Collected_Data_Variation_Experiments/High_Variation_multi"
    # path = "../Collected_Data/Combined_dataset"
    pca_components = 0
    noise_level = 0.05
    number_of_noise_augmentations = 1
    number_of_rotation_augmentations = 0
    add_augmentations = True
    results_folder = "Results_Traditional_Models_AbsoluteEIT"
    regressors = [
        LinearRegression(),
        Ridge(alpha=0.1),
        # Lasso(alpha=0.001, tol=0.01),
        KNeighborsRegressor(n_neighbors=10),
        # DecisionTreeRegressor(max_depth=40),
        # RandomForestRegressor(max_depth=40, n_estimators=20),
        # GradientBoostingRegressor(),
        # AdaBoostRegressor(),
        BaggingRegressor(),
    ]
    for regressor in regressors:
        model_name = regressor.__class__.__name__
        print("Training with regressor: ", regressor.__class__.__name__)
        start_time = time.time()
        trainings_loop(model_name=model_name, regressor=regressor, path_to_training_data=path,
                       normalize=True, add_augmentation=add_augmentations, results_folder=results_folder
                       )
        print("Training took: ", time.time() - start_time)
