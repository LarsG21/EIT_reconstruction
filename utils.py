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
            os.chdir(os.path.join(path, file_or_folder))  # Move into folder with the name of the current date
            print(os.getcwd())
            for file_or_folder in os.listdir(os.getcwd()):  # Move into folder with the name of the setup
                if os.path.isdir(os.path.join(os.getcwd(), file_or_folder)):
                    os.chdir(os.path.join(os.getcwd(), file_or_folder))
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
    # TODO: Verify again, that this is correct !
    keep_mask = protocol_obj.keep_ba
    # pd.concat([df,pd.DataFrame(keep_mask)], axis=1)
    # keep only where measuring electrodes != injection_pos and != injection_neg
    # df = df[df["injection_pos"] != df["measuring_electrode"]]
    # df = df[df["injection_neg"] != df["measuring_electrode"]]
    v = df["amplitude"].to_numpy(dtype=np.float64)
    # v = v[keep_mask]
    return v


def find_center_of_mass(img):
    """
    Find center of mass of image (To detect position of anomaly)
    :param img:
    :return:
    """
    # all pixels > 0 are part of the anomaly and should be set to 1
    img_copy = img.copy()
    img_copy[img_copy > 0] = 1
    center_of_mass = np.array(np.where(img_copy == np.max(img_copy)))
    center_of_mass = np.mean(center_of_mass, axis=1)
    center_of_mass = center_of_mass.astype(int)
    center_of_mass = np.array((center_of_mass[1], center_of_mass[0]))

    return center_of_mass
