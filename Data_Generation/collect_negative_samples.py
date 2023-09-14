import datetime
import os

import numpy as np
import pandas as pd

from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df

img_size = 64
TIME_FORMAT = "%Y-%m-%d %H_%M_%S"


def collect_samples(eit_path: str, save_path: str):
    img = np.zeros([img_size, img_size])

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
        df = convert_multi_frequency_eit_to_df(file_path)

        # combine both df to alternating real and imaginary values
        df_alternating = pd.DataFrame({"real": df["real"], "imaginary": df["imaginary"]}).stack().reset_index(drop=True)
        # convert to numpy array
        alternating_values = df_alternating.to_numpy()
        v1 = alternating_values
        img = np.zeros([img_size, img_size])

        voltages.append(v1)
        images.append(img)
    df = pd.DataFrame({"images": images, "voltages": voltages})
    save_path_data = os.path.join(save_path,
                                  f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
    df.to_pickle(save_path_data)
    print(f"Saved data to {save_path_data}")


if __name__ == '__main__':
    eit_path = "../eit_data/20230914 16.33.17/setup_1"
    save_path = "../Collected_Data/Data_14_09_negative_multifreq"
    collect_samples(eit_path, save_path)
