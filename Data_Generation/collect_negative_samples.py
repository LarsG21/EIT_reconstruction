import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df, convert_single_frequency_eit_file_to_df

img_size = 64
TIME_FORMAT = "%Y-%m-%d %H_%M_%S"

ABSOLUTE_EIT = True


# TODO: Import form utils instead
def preprocess_absolute_eit_frame(df):
    """
    Preprocesses the absolute eit frame to an array of alternating real and imaginary values.
    Use this methode to be consistent with preprocessing over all the project.
    :param df: Dataframe with the eit frame
    :return: The preprocessed eit frame as a numpy array
    """
    df_alternating = pd.DataFrame({"real": df["real"], "imaginary": df["imaginary"]}).stack().reset_index(drop=True)
    df_alternating = df_alternating.to_frame(name="amplitude")
    v1 = df_alternating["amplitude"].to_numpy(dtype=np.float64)  # add alternating imaginary and real values
    return v1

def collect_samples(eit_path: str, save_path: str):
    """
    Creates a df with negative samples from a folder with eit files.
    Caution: Only use eit frames with no object in the tank.
    :param eit_path: Path to the folder with the eit files
    :param save_path: Path to save the df
    :return:
    """

    """ 4. collect data """
    # get the newest file in the folder
    files = os.listdir(eit_path)
    paths = [os.path.join(eit_path, basename) for basename in files]
    # select only files that g´have ending .eit
    paths = [path for path in paths if path.endswith(".eit")]
    voltages = []
    images = []
    for file_path in paths:
        print(file_path)
        if ABSOLUTE_EIT:
            df = convert_multi_frequency_eit_to_df(file_path)
            v1 = preprocess_absolute_eit_frame(df)
        else:
            df = convert_single_frequency_eit_file_to_df(file_path)
            v1 = df["amplitude"].to_numpy(dtype=np.float64)
            # plt.plot(v1)
            # plt.show()

        img = np.zeros([img_size, img_size])

        voltages.append(v1)
        images.append(img)
    df = pd.DataFrame({"images": images, "voltages": voltages})
    save_path_data = os.path.join(save_path,
                                  f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
    df.to_pickle(save_path_data)
    print(f"Saved data to {save_path_data}")


if __name__ == '__main__':
    eit_path = "C:\\Users\\lgudjons\Desktop\\eit_data\\20231124 15.50.42\\setup_1"
    save_path = "../Collected_Data/negatives_3_freq_orientation4"
    absolute_eit = input("Absolute EIT? (y/n)")
    if absolute_eit == "y":
        ABSOLUTE_EIT = True
    else:
        ABSOLUTE_EIT = False

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    collect_samples(eit_path, save_path)
