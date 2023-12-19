import os

import numpy as np
import pandas as pd

from EarlyStoppingHandler import EarlyStoppingHandler
from Evaluation.Evaluate_Test_Set_Dataframe import evaluate_reconstruction_model
from Evaluation.Plot_results_of_evaluation import plot_evaluation_results
from Model_Training_with_pca_reduction import trainings_loop
import matplotlib.pyplot as plt

from tiktzplot_utils import genterate_linepot_with_std
import seaborn as sns


def train_multiple_times_and_plot_losses():
    path = "../Trainings_Data_EIT32/1_Freq_More_Orientations"
    ABSOLUTE_EIT = False
    num_epochs = 60
    learning_rate = 0.001
    pca_components = 128
    add_augmentation = True
    noise_level = 0.02
    number_of_noise_augmentations = 4
    number_of_rotation_augmentations = 0
    number_of_blur_augmentations = 5
    weight_decay = 1e-3  # Adjust this value as needed (L2 regularization)
    df_complete = pd.DataFrame()
    for i in range(1, 10):
        print(f"Run {i}")
        early_stopping_handler = EarlyStoppingHandler(patience=20)
        df_losses, model, pca, model_path = trainings_loop(model_name=f"TESTING_{i}", path_to_training_data=path,
                                               num_epochs=num_epochs, learning_rate=learning_rate,
                                               early_stopping_handler=early_stopping_handler,
                                               pca_components=pca_components, add_augmentation=add_augmentation,
                                               noise_level=noise_level,
                                               number_of_noise_augmentations=number_of_noise_augmentations,
                                               number_of_rotation_augmentations=number_of_rotation_augmentations,
                                               number_of_blur_augmentations=number_of_blur_augmentations,
                                               weight_decay=weight_decay, normalize=False, absolute_eit=ABSOLUTE_EIT
                                               )
        print(df_losses)
        # rename the columns
        df_losses = df_losses.rename(columns={"loss": f"loss_{i}", "val_loss": f"val_loss_{i}"})
        # add the dataframe to the side of the complete dataframe as new columns
        df_complete = pd.concat([df_complete, df_losses], axis=1)
        print(df_complete.shape)

    # save the complete dataframe
    df_complete.to_pickle("df_complete_no_normalization.pkl")

    # load instead of training
    # df_complete = pd.read_pickle("df_complete_no_normalization.pkl")
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
    plt.title("Loss plot training and validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # save points in this format: {(0,0.224)...(103,203.943)}
    colors = ["blue", "orange"]
    labels = ["Training Loss", "Std Training Loss", "Validation Loss", "Std Validation Loss"]
    file_name = "loss_plot.txt"
    df_list = [df_complete_train_loss, df_complete_val_loss]
    genterate_linepot_with_std(file_name, df_list, colors, labels)

    plt.show()


def plot_for_different_hyperparameters():
    model_name = "TESTING"
    # path = "../Training_Data/1_Freq_with_individual_v0s"
    # path = "../Trainings_Data_EIT32/3_Freq"
    # path = "../Collected_Data_Variation_Experiments/High_Variation_multi"
    # path = "../Collected_Data/Combined_dataset"
    # path = "../Collected_Data/Training_set_circular_08_11_3_freq_40mm"
    # path = "../Own_Simulation_Dataset"
    # path = "../Trainings_Data_EIT32/1_Freq_More_Orientations"
    path = "../Trainings_Data_EIT32/3_Freq_Even_orientation_and_GREIT_data"
    ABSOLUTE_EIT = True
    learning_rate = 0.0001
    pca_components = 0  # 0 for no PCA
    add_augmentation = True
    noise_level = 0.02
    epochs = 70
    number_of_noise_augmentations = 1
    number_of_rotation_augmentations = 0
    number_of_superpos_augmentations = 2
    number_of_targets_in_superposition_samples = 2  # 2 equals 3 targets in total
    number_of_blur_augmentations = 4
    weight_decay = 1e-6  # Adjust this value as needed (L2 regularization)
    USE_N_SAMPLES_FOR_TRAIN = 0  # 0 for all data
    df_eval = pd.DataFrame()
    experiment_name = "Differnt_PCAS4"

    for i in range(4):
        print(f"###################### Run {i} #########################")
        amplitude_responses = []
        shape_deformations = []
        ringings = []
        position_errors = []
        pearson_correlations = []

        # wheight_decay_list = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
        # learning_rate_list = [0.00001, 0.0001, 0.001, 0.01]
        # dropout_pobs = [0.05, 0.1, 0.15]
        # number_of_blur_augmentations_list = [0, 2, 4, 6, 8]
        number_of_pca_components_list = [1024, 512, 256, 128, 64]
        for nr_pca in number_of_pca_components_list:
            model_name = f"TESTING_{epochs}_epochs_{str(nr_pca).replace('.', '_')}_wd"
            print(
                f"####################Training with {nr_pca} PCA Components for {epochs} num epochs ###########################")
            early_stopping_handler = EarlyStoppingHandler(patience=30)
            df, model, pca, model_path = trainings_loop(model_name=model_name, path_to_training_data=path,
                                                        num_epochs=epochs, learning_rate=learning_rate,
                                                        early_stopping_handler=early_stopping_handler,
                                                        pca_components=nr_pca, add_augmentation=add_augmentation,
                                                        noise_level=noise_level,
                                                        number_of_noise_augmentations=number_of_noise_augmentations,
                                                        number_of_rotation_augmentations=number_of_rotation_augmentations,
                                                        number_of_blur_augmentations=number_of_blur_augmentations,
                                                        weight_decay=weight_decay, normalize=False,
                                                        absolute_eit=ABSOLUTE_EIT,
                                                        number_of_superpos_augmentations=number_of_superpos_augmentations,
                                                        number_of_targets_in_superposition_samples=number_of_targets_in_superposition_samples,
                                                        )

            if ABSOLUTE_EIT:
                test_set_path = "../Test_Data_EIT32/3_Freq/Test_set_circular_24_11_3_freq_40mm_eit32_orientation25_2/combined.pkl"
                v0 = None
            else:

                test_set_path = "../Test_Data_EIT32/1_Freq/Test_set_circular_10_11_1_freq_40mm/combined.pkl"
                v0 = np.load(os.path.join(os.path.dirname(test_set_path), "v0.npy"))

            df_test_set = pd.read_pickle(test_set_path)
            # load v0 from the same folder as the test set

            df_evaluate_results = evaluate_reconstruction_model(ABSOLUTE_EIT=ABSOLUTE_EIT, NORMALIZE=False,
                                                                SHOW=False, df_test_set=df_test_set,
                                                                v0=v0, model=model, model_path=f"/{model_name}.pkl",
                                                                pca=pca, regressor=None)
            ar = df_evaluate_results["amplitude_response"].mean()
            sd = df_evaluate_results["shape_deformation"].mean()
            ringing = df_evaluate_results["ringing"].mean()
            pe = df_evaluate_results["position_error"].mean()
            pc = df_evaluate_results["pearson_correlation"].mean()
            amplitude_responses.append(ar)
            shape_deformations.append(sd)
            ringings.append(ringing)
            position_errors.append(pe)
            pearson_correlations.append(pc)
            if len(df_eval) == 0:
                df_eval = pd.DataFrame(data={"x": nr_pca, "ar": ar, "sd": sd, "ringing": ringing, "pe": pe, "pc": pc},
                                       index=[0])
            else:
                df_eval = pd.concat([df_eval, pd.DataFrame(data={"x": nr_pca, "ar": ar, "sd": sd, "ringing": ringing,
                                                                 "pe": pe, "pc": pc}, index=[0])])

            plt.title(f"Training for {epochs} epochs")
            plt.show()
            # plot_evaluation_results(df_evaluate_results)
            # plot averages for each hyperparameter in a new plot with log scale x axis

        def plot_metrics(wheight_decay_list, metrics, metric_names):
            num_metrics = len(metrics)
            for i in range(num_metrics):
                plt.plot(wheight_decay_list, metrics[i])
                plt.title(f"Average {metric_names[i]}")
                plt.xlabel("Weight decay")
                plt.xscale('log')  # Set logarithmic scale on x-axis
                plt.ylabel(metric_names[i])
                plt.show()

        metric_names = ['Amplitude response', 'Shape deformation'
                                              'Ringing', 'Position error', 'Pearson correlation']
        try:
            plot_metrics(number_of_pca_components_list, [amplitude_responses, shape_deformations,
                                                         ringings, position_errors, pearson_correlations], metric_names)
            print("OK")
        except IndexError:
            print("Index error")
        print(df_eval)
    df_eval.to_pickle(f"df_eval_New_{experiment_name}.pkl")

    # load instead of training
    # df_eval = pd.read_pickle(f"df_eval_New_{experiment_name}.pkl")
    df_eval = df_eval.reset_index(drop=True)
    # replace 0 with 1024 where col "x" == 0
    # df_eval.loc[df_eval["x"] == 0, "x"] = 1024


    # plot lineplot with std over lr for each metric with seaborn

    def plot_metrics_with_std(df_eval, metric_names):
        for metric in metric_names:
            plt.figure()
            sns.lineplot(data=df_eval, x="x", y=metric)
            plt.xlabel("Nr PCA components")
            # plt.xscale('log')
            plt.ylabel(metric)
            plt.show()
            # delete figure
            plt.clf()

    metric_names_df = [
        "ar",
        "sd",
        "ringing",
        "pe",
        "pc"
    ]

    plot_metrics_with_std(df_eval, metric_names_df)


if __name__ == '__main__':
    # train_multiple_times_and_plot_losses()
    plot_for_different_hyperparameters()
