import os
import time

import numpy as np
import pandas as pd


def wait_for_start_of_measurement(path):
    """
    Waits for the first file to be written. Searches for the setup folder and returns the path to it.
    :param eit_path:
    :param path:
    :return:
    """
    eit_path = ""
    while len(os.listdir(path)) == 0:
        print("Waiting for files to be written")
        time.sleep(0.5)
    print("EIT capture started")
    time.sleep(1)
    for file_or_folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_or_folder)):
            os.chdir(os.path.join(path, file_or_folder))
            print(os.getcwd())
            os.chdir((os.path.join(os.getcwd(), "setup")))
            eit_path = os.getcwd()
            print(eit_path)
            break
    return eit_path


def get_relevant_voltages(df, protocol_obj):
    """
    This function takes a dataframe and returns the voltages that are relevant for the reconstruction
    Removes voltages of the injection electrodes.
    :param df:
    :return:
    """
    # TODO Find out how dist_exc work
    keep_mask = protocol_obj.keep_ba
    df_keep_mask = pd.DataFrame(keep_mask, columns=["keep"])
    df = pd.concat([df, df_keep_mask], axis=1)
    df1 = df[df["keep"] == True].drop("keep", axis=1)
    v = df1["amplitude"].to_numpy(dtype=np.float64)
    return v
