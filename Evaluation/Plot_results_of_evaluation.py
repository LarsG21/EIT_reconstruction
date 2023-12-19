import json
import os
import pickle

import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt

from Evaluation.Evaluate_Test_Set_Dataframe import evaluate_reconstruction_model
from Evaluation.eval_plots import plot_shape_deformation, plot_position_error, plot_amplitude_response, plot_ringing
import tikzplotlib

from Model_Training.Models import LinearModel, LinearModelWithDropout2
from tiktzplot_utils import generate_boxplot
from utils import load_model_from_path


def plot_evaluation_results(df, open_plots_over_space=True, save_path=""):
    remove_outliers = True

    def pretty(d):
        print("{")
        for key, value in d.items():
            print(str(key) + "=" + str(value) + ",")
        print("}")

    # remove outliers from df in amplitude_response and position_error > or < N std
    N = 3
    border_amplitude_response = N * df["amplitude_response"].std()
    border_position_error = N * df["position_error"].std()
    border_shape_deformation = N * df["shape_deformation"].std()
    print(border_amplitude_response)
    print(border_position_error)
    print(border_shape_deformation)
    print("Number of samples", len(df))

    # remove constant offset from position_error

    errors = df["error_vector"].apply(lambda x: np.array(x))

    # find mean error_vector
    mean = errors.mean()
    print(mean)
    # subtract mean from error_vector
    df["error_vector"] = df["error_vector"].apply(lambda x: np.array(x) - mean)

    print("Number of samples", len(df))
    if open_plots_over_space:
        try:
            if remove_outliers:
                df_ar = df[
                    np.abs(df["amplitude_response"] - df["amplitude_response"].mean()) <= border_amplitude_response]
                print("Number of samples AR", len(df_ar))
            else:
                df_ar = df
            plot_amplitude_response(df_ar,
                                    save_path="Results/amplitude_response.png"
                                    )
            if remove_outliers:
                df_pe = df[np.abs(df["position_error"] - df["position_error"].mean()) <= border_position_error]
                print("Number of samples PE", len(df_pe))
            else:
                df_pe = df

            plot_position_error(df_pe,
                                save_path="Results/position_error.png"
                                )

            if remove_outliers:
                df_sd = df[np.abs(df["shape_deformation"] - df["shape_deformation"].mean()) <= border_shape_deformation]
                print("Number of samples SD", len(df_sd))
            else:
                df_sd = df
            plot_shape_deformation(df_sd,
                                   save_path="Results/shape_deformation.png"
                                   )

            plot_ringing(df,
                         save_path="Results/ringing.png"
                         )
        except ValueError as e:
            print(e)
            print("Error while plotting over space")
            return None

    # convert col error_vector to np.array

    # scatter plot of error_vector with number at each point represented as a color
    # plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]), c=df["position_error"])
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]))

    plt.xlabel("x error [px]")
    plt.ylabel("y error [px]")
    plt.title("Error vector")
    # save plot
    # plt.savefig("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\error_vector.png")
    plt.show()

    if remove_outliers:
        df = df[np.abs(df["amplitude_response"] - df["amplitude_response"].mean()) <= border_amplitude_response]
        df = df[np.abs(df["position_error"] - df["position_error"].mean()) <= border_position_error]
        df = df[np.abs(df["shape_deformation"] - df["shape_deformation"].mean()) <= border_shape_deformation]

    # in one plot
    plt.boxplot([df["amplitude_response"], df["shape_deformation"], df["ringing"]], labels=["amplitude_response",
                                                                                            "shape_deformation",
                                                                                            "ringing"])
    # get median of all metrics
    median_ar = df["amplitude_response"].median()
    median_sd = df["shape_deformation"].median()
    median_ringing = df["ringing"].median()
    median_pe = df["position_error"].median()
    # get upper and lower quartile
    q1_ar = df["amplitude_response"].quantile(0.25)
    q3_ar = df["amplitude_response"].quantile(0.75)
    q1_sd = df["shape_deformation"].quantile(0.25)
    q3_sd = df["shape_deformation"].quantile(0.75)
    q1_ringing = df["ringing"].quantile(0.25)
    q3_ringing = df["ringing"].quantile(0.75)
    q1_pe = df["position_error"].quantile(0.25)
    q3_pe = df["position_error"].quantile(0.75)
    # get whiskers
    whis_ar = [max(0, q1_ar - 1.5 * (q3_ar - q1_ar)), q3_ar + 1.5 * (q3_ar - q1_ar)]
    whis_sd = [max(0, q1_sd - 1.5 * (q3_sd - q1_sd)), q3_sd + 1.5 * (q3_sd - q1_sd)]
    whis_ringing = [max(0, q1_ringing - 1.5 * (q3_ringing - q1_ringing)), q3_ringing + 1.5 * (q3_ringing - q1_ringing)]
    whis_pe = [max(0, q1_pe - 1.5 * (q3_pe - q1_pe)), q3_pe + 1.5 * (q3_pe - q1_pe)]

    # combine all in dict
    boxplot_dict_ar = {
        "lower whisker": whis_ar[0],
        "lower quartile": q1_ar,
        "median": median_ar,
        "upper quartile": q3_ar,
        "upper whisker": whis_ar[1]
    }
    boxplot_dict_sd = {
        "lower whisker": whis_sd[0],
        "lower quartile": q1_sd,
        "median": median_sd,
        "upper quartile": q3_sd,
        "upper whisker": whis_sd[1]
    }
    boxplot_dict_ringing = {
        "lower whisker": whis_ringing[0],
        "lower quartile": q1_ringing,
        "median": median_ringing,
        "upper quartile": q3_ringing,
        "upper whisker": whis_ringing[1]
    }

    boxplot_dict_pe = {
        "lower whisker": whis_pe[0],
        "lower quartile": q1_pe,
        "median": median_pe,
        "upper quartile": q3_pe,
        "upper whisker": whis_pe[1]
    }

    boxplot_dict_list = [boxplot_dict_ar, boxplot_dict_sd, boxplot_dict_ringing]
    labels = ["AR", "SD", "Ringing"]
    colors = ["cyan", "orange", "green"]
    generate_boxplot("boxplot.tex", boxplot_dict_list, labels, colors)

    boxplot_dict_list = [boxplot_dict_pe]
    labels = ["PE"]
    colors = ["cyan"]
    generate_boxplot("boxplot_pe.tex", boxplot_dict_list, labels, colors)

    # print them all out

    print(f"Boxplot dict amplitude response: \n {pretty(boxplot_dict_ar)}")
    print(f"Boxplot dict shape deformation: \n {pretty(boxplot_dict_sd)}")
    print(f"Boxplot dict ringing: \n {pretty(boxplot_dict_ringing)}")

    plt.title("Evaluation metrics")
    plt.ylabel("Relative Metric")
    tikzplotlib.save("Results/boxplot_all.tikz")
    plt.savefig("Results/boxplot_all.png")
    if save_path != "":
        plt.savefig(os.path.join(save_path, "boxplot_all.png"))
    plt.show()

    plt.boxplot([df["position_error"]], labels=["position_error"])
    plt.title("Boxplot of position_error")
    plt.ylabel("Error [px]")
    tikzplotlib.save("Results/boxplot_position_error.tikz")
    plt.savefig("Results/boxplot_position_error.png")
    if save_path != "":
        plt.savefig(os.path.join(save_path, "boxplot_position_error.png"))
    plt.show()
    # if df contains pearson_correlation
    if "pearson_correlation" in df.columns:
        plt.boxplot([df["pearson_correlation"]], labels=["pearson_correlation"])
        plt.title("Boxplot of pearson_correlation")
        plt.ylabel("Pearson Correlation")
        tikzplotlib.save("Results/boxplot_pearson_correlation.tikz")
        plt.savefig("Results/boxplot_pearson_correlation.png")
        if save_path != "":
            plt.savefig(os.path.join(save_path, "boxplot_pearson_correlation.png"))
        plt.show()


    boxplot_dict_pe = {
        "lower whisker": whis_pe[0],
        "lower quartile": q1_pe,
        "median": median_pe,
        "upper quartile": q3_pe,
        "upper whisker": whis_pe[1]
    }
    print(f"Boxplot dict position error: \n {pretty(boxplot_dict_pe)}")

    # save results dict in json file
    # with open("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\results_dict.json", "w") as f:
    #     json.dump(results_dict, f)


def main():
    ABSOLUTE_EIT = False
    normalize = False
    model_path = "../Trainings_Data_EIT32/1_Freq_More_Orientations/Models/LinearModelWithDropout2/THESIS_BES_MODEL/model_2023-12-18_12-41-06_59_60.pth"

    if ABSOLUTE_EIT:
        test_set_path = "../Test_Data_EIT32/3_Freq/Test_set_circular_24_11_3_freq_40mm_eit32_orientation25_2/combined.pkl"
    else:
        # test_set_path = "../Test_Data/Test_Set_1_Freq_23_10_circular/combined.pkl.pkl"
        # test_set_path = "../Test_Data/Test_Set_Circular_single_freq/combined.pkl.pkl"
        test_set_path = "../Test_Data_EIT32/1_Freq/Test_set_circular_10_11_1_freq_40mm/combined.pkl"

    # load v0 from the same folder as the test set
    if not ABSOLUTE_EIT:
        v0 = np.load(os.path.join(os.path.dirname(test_set_path), "v0.npy"))
    else:
        v0 = None
    df_test_set = pd.read_pickle(test_set_path)

    model, pca, NORMALIZE = load_model_from_path(path=model_path, normalize=normalize, absoulte_eit=ABSOLUTE_EIT)

    df_evaluate_results = evaluate_reconstruction_model(ABSOLUTE_EIT=ABSOLUTE_EIT, NORMALIZE=normalize, SHOW=True,
                                                        df_test_set=df_test_set,
                                                        v0=v0, model=model, model_path=model_path, pca=pca,
                                                        regressor=None,
                                                        )

    plot_evaluation_results(df_evaluate_results)


if __name__ == '__main__':
    # # Path to a pickle file containing the evaluation results created by Evaluate_Test_Set_Dataframe.py
    # df = pd.read_pickle(
    #     "Results/evaluation_model_model_2023-11-15_12-54-12_99_100.pkl")
    df = pd.read_pickle(
        "Results\evaluation_regressor_KNeighborsRegressor.pkl")
    plot_evaluation_results(df, open_plots_over_space=True)
    # main()
