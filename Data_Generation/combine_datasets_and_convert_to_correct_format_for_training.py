import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Data_Generation.utils import look_at_dataset
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df
from pyeit import mesh
from pyeit.eit import protocol
from reconstruction_algorithims import solve_and_plot_jack


def get_relevant_voltages(df, protocol_obj):
    """
    This function takes a dataframe and returns the voltages that are relevant for the reconstruction
    Removes voltages of the injection electrodes.
    :param df:
    :return:
    """
    if type(df) == np.ndarray:
        df = pd.DataFrame(df, columns=["amplitude"])
    keep_mask = protocol_obj.keep_ba
    df_keep_mask = pd.DataFrame(keep_mask, columns=["keep"])
    df = pd.concat([df, df_keep_mask], axis=1)
    df1 = df[df["keep"] == True].drop("keep", axis=1)
    v = df1["amplitude"].to_numpy(dtype=np.float64)
    return v


def reconstruct_multiple_voltages(voltage_array, v0, img_array=None):
    """
    Reconstructs the image for multiple voltages
    :param voltage_array:
    :param v0:
    :return:
    """
    mesh_obj = mesh.create(32, h0=0.1)
    if img_array is None:
        for v1 in voltage_array:
            solve_and_plot_jack(v0=v0, v1=v1, mesh_obj=mesh_obj, protocol_obj=protocol_obj)
    else:
        for v1, img in zip(voltage_array, img_array):
            solve_and_plot_jack(v0=v0, v1=v1, mesh_obj=mesh_obj, protocol_obj=protocol_obj)
            plt.imshow(img)
            plt.show()


def convert_df_to_separate_npy_files(df, save_path):
    """
    Converts the dataframe to separate npy files. These can be used in Model Training.
    :param df: Dataframe with images and voltages
    :param save_path: Path to save the npy files
    :return:
    """
    # remove all rows from df where len(voltages) != median of len(voltages)
    # remove accidentally added rows
    df = df[df["voltages"].apply(lambda x: len(x)) == df["voltages"].apply(lambda x: len(x)).median()]

    AVERGAGE_V0_FRAME = False
    img_array = df["images"].to_list()
    img_array = np.array(img_array)
    voltages_df = df["voltages"]
    # v0_df = convert_single_frequency_eit_file_to_df(path_vo)
    # v0 = get_relevant_voltages(v0_df, protocol_obj=protocol_obj)
    # v0 = v0_df["amplitude"].to_numpy(dtype=np.float64)
    # if AVERGAGE_V0_FRAME:
    #     plt.plot(v0)


    # apply get_relevant_voltages to all voltages
    # voltage_array = np.array(
    #     [get_relevant_voltages(v, protocol_obj=protocol_obj) for v in voltages_df])
    voltage_array = np.array(voltages_df.to_list())
    # save voltages to npy
    np.save(os.path.join(save_path, "v1_array.npy"), voltage_array)
    if AVERGAGE_V0_FRAME:
        # v0 = average over all v1
        v0_mean = np.mean(voltage_array, axis=0)
        v0 = v0_mean
        plt.plot(v0_mean)
        plt.legend(["v0", "v0_mean"])
        plt.show()
        np.save(os.path.join(save_path, "v0_05_09.npy"),
                v0)  # Dont save anymore ! Use averaged V0 from Negative_Sample_set
    # save images to npy
    np.save(os.path.join(save_path, "img_array.npy"), img_array)
    return voltage_array, img_array


def combine_multiple_pickles(path):
    """
    Combines multiple pickles to one pickle.
    :param path:
    :return:
    """
    complete_df = None
    negative_samples = 0
    positive_samples = 0
    for file in os.listdir(path):
        if file.endswith(".pkl") and file != "combined.pkl" and file != "pca.pkl":
            df_new = pd.read_pickle(os.path.join(path, file))
            print(f"length of {file}: {len(df_new)}")
            # if length of df_new is > 300, take only 300 samples
            # if len(df_new) > 300:
            #     df_new = df_new.sample(300)
            if "negative" in file:
                negative_samples += len(df_new)
            else:
                positive_samples += len(df_new)

            if complete_df is None:
                complete_df = df_new
            else:
                complete_df = pd.concat([complete_df, df_new])
    complete_df.to_pickle(os.path.join(path, "combined.pkl"))
    print(f"Length of combined df: {len(complete_df)}")
    print(f"Percentage of negative samples: {negative_samples / (negative_samples + positive_samples)}")
    print(f"Percentage of positive samples: {positive_samples / (negative_samples + positive_samples)}")
    return complete_df


def get_infos_about_eit_dataframe(df, complex_values=True):
    """
    This function prints some infos about the dataframe like number of samples,
    used frequencies, number of electrodes etc.
    :param df:
    :param complex_values: If True, the number of measurements is doubled
    :return:
    """
    number_electrodes = 32
    number_measurements = number_electrodes ** 2
    if complex_values:
        number_measurements *= 2

    print(f"Number of samples: {len(df)}")
    print(f"Length of one Voltage Vector:{df['voltages'].reset_index().iloc[0]['voltages'].shape[0]}")
    print(
        f"Number of used frequencies: {df['voltages'].reset_index().iloc[0]['voltages'].shape[0] / number_measurements}")
    print(f"Number of electrodes: {number_electrodes}")

if __name__ == '__main__':
    # read df from pickle
    # df = pd.read_pickle("Data_measured2023-08-23 16_04_17.pkl")
    protocol_obj = protocol.create(32, dist_exc=1, step_meas=1, parser_meas="std")

    # path = "../Collected_Data/Combined_dataset"
    path = "../Training_Data/1_Freq_After_16_10"
    # path = "../Collected_Data/Data_23_10_40mm"
    # path = "../Collected_Data/Data_16_10_single_freq_40mm"
    # path = "../Training_Data/3_Freq"

    df = combine_multiple_pickles(path=path)
    img_array = df["images"].to_list()
    img_array = np.array(img_array)
    voltages_df = df["voltages"]
    path_vo = "../eit_experiments/Move_Bottle_2/setup/setup_00001.eit"

    v0 = np.load("../Collected_Data/Combined_dataset/v0.npy")
    # shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    voltage_array, img_array = convert_df_to_separate_npy_files(df,
                                                                save_path=path, )
    look_at_dataset(img_array=img_array, v1_array=voltage_array,
                    # v0=v0,
                    )
    # reconstruct_multiple_voltages(voltage_array=voltage_array, v0=v0, img_array=img_array)

    # v0 = np.load("../Collected_Data/V0_SAMPLES_13_10_2023/v0_13_10.npy")
    # plt.plot(v0)
    # plt.show()
