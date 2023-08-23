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


if __name__ == '__main__':
    # read df from pickle
    df = pd.read_pickle("Data_measured2023-08-23 15_05_04.pkl")
    img_array = df["images"].to_numpy()
    voltages_df = df["voltages"]
    path_vo = "v0.eit"
    v0_df = convert_single_frequency_eit_file_to_df(path_vo)
    v0 = get_relevant_voltages(v0_df, protocol_obj=protocol.create(32, dist_exc=8, step_meas=1, parser_meas="std"))

    # apply get_relevant_voltages to all voltages
    voltage_array = np.array(
        [get_relevant_voltages(v, protocol_obj=protocol.create(32, dist_exc=8, step_meas=1, parser_meas="std")) for v in
         voltages_df])

    look_at_dataset(img_array=img_array, v1_array=voltage_array, v0=v0)
    # reconstruct_multiple_voltages(voltage_array=voltage_array, v0=v0, img_array=img_array)
