import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Data_Generation.utils import look_at_dataset
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df
from pyeit.eit import protocol


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

# read df from pickle
df = pd.read_pickle("Data_measured2023-08-17 15_47_59.pkl")

img_array = df["images"].to_numpy()
voltages_df = df["voltages"]

path_vo = "../eit_data/20230817 15.44.29/setup/setup_00062.eit"
v0_df = convert_single_frequency_eit_file_to_df(path_vo)
v0 = get_relevant_voltages(v0_df, protocol_obj=protocol.create(32, dist_exc=8, step_meas=1, parser_meas="std"))

# apply get_relevant_voltages to all voltages
voltage_array = np.array(
    [get_relevant_voltages(v, protocol_obj=protocol.create(32, dist_exc=8, step_meas=1, parser_meas="std")) for v in
     voltages_df])

look_at_dataset(img_array=img_array, v1_array=voltage_array, v0=v0)
