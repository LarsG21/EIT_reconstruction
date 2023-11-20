import os

import numpy as np
import pandas as pd

from EarlyStoppingHandler import EarlyStoppingHandler
from Evaluation.Evaluate_Test_Set_Dataframe import evaluate_reconstruction_model
from Evaluation.Plot_results_of_evaluation import plot_evaluation_results
from Model_Training_with_pca_reduction import trainings_loop
import matplotlib.pyplot as plt
import tikzplotlib


def train_multiple_times_and_plot_losses():
    path = "../Training_Data/1_Freq_with_individual_v0s"
    num_epochs = 400
    learning_rate = 0.001
    pca_components = 0
    add_augmentation = True
    noise_level = 0.02
    number_of_noise_augmentations = 10
    number_of_rotation_augmentations = 0
    number_of_blur_augmentations = 0
    weight_decay = 1e-3  # Adjust this value as needed (L2 regularization)
    df_complete = pd.DataFrame()
    for i in range(1, 10):
        print(f"Run {i}")
        early_stopping_handler = EarlyStoppingHandler(patience=20)
        df_losses, model, pca = trainings_loop(model_name=f"TESTING_{i}", path_to_training_data=path,
                                          num_epochs=num_epochs, learning_rate=learning_rate,
                                          early_stopping_handler=early_stopping_handler,
                                          pca_components=pca_components, add_augmentation=add_augmentation,
                                          noise_level=noise_level,
                                          number_of_noise_augmentations=number_of_noise_augmentations,
                                          number_of_rotation_augmentations=number_of_rotation_augmentations,
                                          number_of_blur_augmentations=number_of_blur_augmentations,
                                          weight_decay=weight_decay, normalize=False,
                                          )
        print(df_losses)
        # rename the columns
        df_losses = df_losses.rename(columns={"loss": f"loss_{i}", "val_loss": f"val_loss_{i}"})
        # add the dataframe to the side of the complete dataframe as new columns
        df_complete = pd.concat([df_complete, df_losses], axis=1)
        print(df_complete.shape)

    # save the complete dataframe
    df_complete.to_pickle("df_complete_no_normalization.pkl")
    # df_complete = pd.read_pickle("df_complete_no_normalization.pkl")
    average_length_col_without_nan = int(df_complete.count(axis=0).median())
    df_complete = df_complete.iloc[:300, :]
    # get the mean of the complete dataframe
    # select the columns with the loss values
    df_complete_train_loss = df_complete.filter(regex="loss")
    df_complete_train_loss["mean"] = df_complete_train_loss.mean(axis=1)
    df_complete_train_loss["std"] = df_complete_train_loss.std(axis=1)
    print(df_complete_train_loss)
    # select the columns with the val_loss values
    df_complete_val_loss = df_complete.filter(regex="val_loss")
    df_complete_val_loss["mean"] = df_complete_val_loss.mean(axis=1)
    df_complete_val_loss["std"] = df_complete_val_loss.std(axis=1)
    print(df_complete_val_loss)

    # plot the mean and the std
    plt.plot(df_complete_train_loss["mean"])
    plt.fill_between(df_complete_train_loss.index, df_complete_train_loss["mean"] - df_complete_train_loss["std"],
                     df_complete_train_loss["mean"] + df_complete_train_loss["std"], alpha=0.2)

    plt.plot(df_complete_val_loss["mean"])
    plt.fill_between(df_complete_val_loss.index, df_complete_val_loss["mean"] - df_complete_val_loss["std"],
                     df_complete_val_loss["mean"] + df_complete_val_loss["std"], alpha=0.2)
    plt.legend(["train_loss", "std_train_loss", "val_loss", "std_val_loss"])
    # save as tikz
    # tikzplotlib.save("losses.tex")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.show()


def plot_for_different_epochs():
    model_name = "TEST"
    # path = "../Training_Data/1_Freq_with_individual_v0s"
    # path = "../Trainings_Data_EIT32/3_Freq"
    # path = "../Collected_Data_Variation_Experiments/High_Variation_multi"
    # path = "../Collected_Data/Combined_dataset"
    # path = "../Collected_Data/Training_set_circular_08_11_3_freq_40mm"
    # path = "../Own_Simulation_Dataset"
    path = "../Trainings_Data_EIT32/1_Freq"
    # path = "../Collected_Data/Even_Orientation_Dataset"
    ABSOLUTE_EIT = False
    learning_rate = 0.001
    pca_components = 128  # 0 for no PCA
    add_augmentation = False
    noise_level = 0.02
    number_of_noise_augmentations = 0
    number_of_rotation_augmentations = 0
    number_of_blur_augmentations = 5
    weight_decay = 1e-2  # Adjust this value as needed (L2 regularization)
    USE_N_SAMPLES_FOR_TRAIN = 0  # 0 for all data

    num_epochs_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for num in num_epochs_list:
        print(f"####################Training for {num} epochs###########################")
        early_stopping_handler = EarlyStoppingHandler(patience=30)
        df, model, pca = trainings_loop(model_name=model_name, path_to_training_data=path,
                                        num_epochs=num, learning_rate=learning_rate, early_stopping_handler=early_stopping_handler,
                                        pca_components=pca_components, add_augmentation=add_augmentation, noise_level=noise_level,
                                        number_of_noise_augmentations=number_of_noise_augmentations,
                                        number_of_rotation_augmentations=number_of_rotation_augmentations,
                                        weight_decay=weight_decay, normalize=False, electrode_level_normalization=False,
                                        )

        if ABSOLUTE_EIT:
            test_set_path = "../Test_Data/Test_Set_Circular_16_10_3_freq/combined.pkl"
        else:

            test_set_path = "../Test_Data_EIT32/1_Freq/Test_set_circular_10_11_1_freq_40mm/combined.pkl"

        df_test_set = pd.read_pickle(test_set_path)
        # load v0 from the same folder as the test set
        v0 = np.load(os.path.join(os.path.dirname(test_set_path), "v0.npy"))

        df_evaluate_results = evaluate_reconstruction_model(ABSOLUTE_EIT=ABSOLUTE_EIT, NORMALIZE=False, SHOW=False, df_test_set=df_test_set,
                                                            v0=v0, model=model, model_path=f"/{model_name}.pkl", pca=pca, regressor=None)
        plt.title(f"Training for {num} epochs")
        plt.show()
        plot_evaluation_results(df_evaluate_results)



if __name__ == '__main__':
    plot_for_different_epochs()

