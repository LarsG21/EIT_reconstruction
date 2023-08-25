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
    protocol_obj = protocol.create(32, dist_exc=8, step_meas=1, parser_meas="std")
    if img_array is None:
        for v1 in voltage_array:
            solve_and_plot_jack(v0=v0, v1=v1, mesh_obj=mesh_obj, protocol_obj=protocol_obj)
    else:
        for v1, img in zip(voltage_array, img_array):
            solve_and_plot_jack(v0=v0, v1=v1, mesh_obj=mesh_obj, protocol_obj=protocol_obj)
            plt.imshow(img)
            plt.show()


def convert_df_to_separate_npy_files(df, save_path, path_vo="v0.eit"):
    """
    Converts the dataframe to separate npy files. These can be used in Model Training.
    :param df: Dataframe with images and voltages
    :param save_path: Path to save the npy files
    :return:
    """
    img_array = df["images"].to_list()
    img_array = np.array(img_array)
    voltages_df = df["voltages"]
    v0_df = convert_single_frequency_eit_file_to_df(path_vo)
    v0 = get_relevant_voltages(v0_df, protocol_obj=protocol.create(32, dist_exc=8, step_meas=1, parser_meas="std"))

    # save v0 to npy
    np.save(os.path.join(save_path, "v0.npy"), v0)

    # apply get_relevant_voltages to all voltages
    voltage_array = np.array(
        [get_relevant_voltages(v, protocol_obj=protocol.create(32, dist_exc=8, step_meas=1, parser_meas="std")) for v in
         voltages_df])
    # save voltages to npy
    np.save(os.path.join(save_path, "v1_array.npy"), voltage_array)
    # save images to npy
    np.save(os.path.join(save_path, "img_array.npy"), img_array)
    return v0, voltage_array, img_array


def combine_multiple_pickles(path):
    """
    Combines multiple pickles to one pickle.
    :param path:
    :return:
    """
    complete_df = None
    for file in os.listdir(path):
        if file.endswith(".pkl") and file != "combined.pkl":
            df_new = pd.read_pickle(os.path.join(path, file))
            if complete_df is None:
                complete_df = df_new
            else:
                complete_df = pd.concat([complete_df, df_new])
    complete_df.to_pickle(os.path.join(path, "combined.pkl"))
    print(f"Length of combined df: {len(complete_df)}")
    return complete_df

if __name__ == '__main__':
    # read df from pickle
    # df = pd.read_pickle("Data_measured2023-08-23 16_04_17.pkl")
    path = "../Collected_Data/Data_25_08"
    df = combine_multiple_pickles(path=path)
    img_array = df["images"].to_list()
    img_array = np.array(img_array)
    voltages_df = df["voltages"]
    path_vo = "v0.eit"

    v0, voltage_array, img_array = convert_df_to_separate_npy_files(df,
                                                                    save_path=path,
                                                                    path_vo=path_vo)

    look_at_dataset(img_array=img_array, v1_array=voltage_array, v0=v0)
    # reconstruct_multiple_voltages(voltage_array=voltage_array, v0=v0, img_array=img_array)
