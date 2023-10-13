import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def add_normalizations(v1, NORMALIZE_MEDIAN, NORMALIZE_PER_ELECTRODE=False):
    """
    Adds the normalizations to the eit frame.
    :param v1: The eit frame as a numpy array
    :param NORMALIZE_MEDIAN: (x - median) / median
    :param NORMALIZE_PER_ELECTRODE: Normalize the samples per max of electrode (CURRENTLY ONLY FOR 3 FREQUENCIES)
    :return: the preprocessed eit frame as a numpy array
    """
    if NORMALIZE_PER_ELECTRODE:
        normalized_samples = []
        if len(v1) == 6144:
            length = len(v1)
            freq1 = v1[0:int(length / 3)]
            freq2 = v1[int(length / 3):int(2 * length / 3)]
            freq3 = v1[int(2 * length / 3):length]

            # Divide all frequencies in 32 equal parts
            freq1_split = np.array_split(freq1, 32)
            freq2_split = np.array_split(freq2, 32)
            freq3_split = np.array_split(freq3, 32)

            # Normalize the individual parts by the max value
            freq1_split_normalized = [x / np.max(x) for x in freq1_split]
            freq2_split_normalized = [x / np.max(x) for x in freq2_split]
            freq3_split_normalized = [x / np.max(x) for x in freq3_split]

            # Flatten the normalized parts
            freq1_split_normalized = np.concatenate(freq1_split_normalized)
            freq2_split_normalized = np.concatenate(freq2_split_normalized)
            freq3_split_normalized = np.concatenate(freq3_split_normalized)

            # Concatenate the normalized parts
            sample_normalized = np.concatenate((freq1_split_normalized, freq2_split_normalized, freq3_split_normalized))

            # Append the normalized sample to the list
            normalized_samples.append(sample_normalized)
            return np.array(normalized_samples).flatten()
            # plt.plot(sample_normalized)
            # plt.show()
            # print("OK")
        else:
            for sample in v1:
                # Divide the sample into 3 equal parts
                length = len(sample)
                freq1 = sample[0:int(length / 3)]
                freq2 = sample[int(length / 3):int(2 * length / 3)]
                freq3 = sample[int(2 * length / 3):length]

                # Divide all frequencies in 32 equal parts
                freq1_split = np.array_split(freq1, 32)
                freq2_split = np.array_split(freq2, 32)
                freq3_split = np.array_split(freq3, 32)

                # Normalize the individual parts by the max value
                freq1_split_normalized = [x / np.max(x) for x in freq1_split]
                freq2_split_normalized = [x / np.max(x) for x in freq2_split]
                freq3_split_normalized = [x / np.max(x) for x in freq3_split]

                # Flatten the normalized parts
                freq1_split_normalized = np.concatenate(freq1_split_normalized)
                freq2_split_normalized = np.concatenate(freq2_split_normalized)
                freq3_split_normalized = np.concatenate(freq3_split_normalized)

                # Concatenate the normalized parts
                sample_normalized = np.concatenate(
                    (freq1_split_normalized, freq2_split_normalized, freq3_split_normalized))

                # Append the normalized sample to the list
                normalized_samples.append(sample_normalized)
                # plt.plot(sample_normalized)
                # plt.show()
                # print("OK")

            # Convert the list to a numpy array
            normalized_samples = np.array(normalized_samples)
            v1 = normalized_samples
            return v1
    else:
        median = np.median(v1)

        if NORMALIZE_MEDIAN:
            v1 = v1 - median
            v1 = v1 / median
        return v1


def check_settings_of_model(model_path):
    """
    Checks the settings.txt file of the model and returns the value of normalize.
    :param model_path:
    :return:
    """
    settings_path = os.path.join(os.path.dirname(model_path), "settings.txt")
    if os.path.exists(settings_path):
        print("Loading settings")
        # search for line with "normalize: " and see if it is True or False
        with open(settings_path, "r") as f:
            for line in f.readlines():
                if "normalize: " in line:
                    if "True" in line:
                        normalize = True
                    else:
                        normalize = False
                    break
                else:
                    normalize = None

    return normalize
