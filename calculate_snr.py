from datetime import datetime

import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit
import seaborn as sns

from utils import find_center_of_mass

RADIUS_TANK = 200  # mm

def calculate_voltage_differences(positive_samples, negative_samples):
    """
    Calculates the voltage differences between positive and negative samples.
    The voltage differences are the relevant EIT signals.
    V_Diff = V1 - V0. V0 can be calculated as mean(V0).
    :param positive_samples: NumPy array of positive samples
    :param negative_samples: NumPy array of negative samples
    :return:
    """
    v_diff_pos = (positive_samples - negative_samples.mean(axis=0)) / negative_samples.mean(axis=0).mean()
    v_diff_neg = (negative_samples - negative_samples.mean(axis=0)) / negative_samples.mean(axis=0).mean()
    return v_diff_pos, v_diff_neg


def calculate_snr(amplitude_signal, amplitude_noise):
    """
    Calculates the SNR of a given signal and noise
    :param amplitude_signal:
    :param amplitude_noise:
    :return:
    """
    snr = amplitude_signal.mean() / amplitude_noise.mean()
    snr_db = 20 * np.log10(snr)
    return snr, snr_db


def get_snr_of_sample(index, voltages_pos_np, voltages_neg_np):
    """
    Calculates the SNR of a selected sample
    :param index: index of sample
    :param voltages_pos_np:
    :param voltages_neg_np:
    :return:
    """
    # v_diff_pos = voltages_pos_np[index] - voltages_neg_np.mean(axis=0)
    # v_diff_neg = voltages_neg_np - voltages_neg_np.mean(axis=0)

    v_diff_pos, v_diff_neg = calculate_voltage_differences(voltages_pos_np[index], voltages_neg_np)

    amplitude_signal = np.mean(np.abs(v_diff_pos), axis=0)
    amplitude_noise = np.mean(np.abs(v_diff_neg), axis=0)
    snr = amplitude_signal / amplitude_noise.mean()
    snr_db = 20 * np.log10(snr)
    return round(snr_db, 2)


def calculate_snr_in_dataset(path_to_positives, path_to_negatives):
    """
    Calculates the SNR of the whole dataset.
    :return:
    """
    df_negatives = pd.read_pickle(path_to_negatives)
    df_positives = pd.read_pickle(path_to_positives)
    # remove offset in both datasets
    df_negatives["voltages"] = df_negatives["voltages"].apply(lambda x: x - x.mean(axis=0))
    df_positives["voltages"] = df_positives["voltages"].apply(lambda x: x - x.mean(axis=0))
    print(f"Samples positive: {len(df_positives)}")
    print(f"Samples negative: {len(df_negatives)}")

    # Extract data and convert to NumPy arrays
    voltages_pos_np = np.array(df_positives["voltages"].to_list())
    voltages_neg_np = np.array(df_negatives["voltages"].to_list())
    images_pos = df_positives["images"].to_list()

    # Calculate voltage differences
    v_diff_pos, v_diff_neg = calculate_voltage_differences(voltages_pos_np, voltages_neg_np)

    # calculate rms of voltage differences
    rms_signal = np.sqrt(np.mean(v_diff_pos ** 2, axis=0))
    rms_noise = np.sqrt(np.mean(v_diff_neg ** 2, axis=0))

    # Calculate SNR
    snr, snr_db = calculate_snr(rms_signal, rms_noise)

    # Print results
    print(f"SNR: {snr}")
    print(f"SNR (dB): {snr_db}")

    # Plot samples of v_diff
    for i in range(10):
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(v_diff_pos[i], label="signal")
        axs[0].plot(v_diff_neg[i], label="noise")
        axs[0].legend()
        axs[0].set_title(f"SNR: {get_snr_of_sample(i, voltages_pos_np, voltages_neg_np)} dB")
        axs[1].imshow(images_pos[i])
        axs[1].set_title(f"Corresponding Target Position")
        plt.savefig(f"snr_{i}.png")
        plt.show()

    # Define the exponential function


def exponential_function(t, a, b):
    return a * np.exp(b * t)


def evaluate_snr_over_radius(path_to_positives, path_to_negatives):
    """
    Calculates the SNR of the whole dataset.
    :return:
    """
    df_negatives = pd.read_pickle(path_to_negatives)
    df_positives = pd.read_pickle(path_to_positives)
    # if df_positives has no col timestamp, add it
    if "timestamp" not in df_positives.columns:
        df_positives["timestamp"] = datetime.now()

    print(f"Samples positive: {len(df_positives)}")
    print(f"Samples negative: {len(df_negatives)}")

    # Extract data and convert to NumPy arrays
    voltages_pos_np = np.array(df_positives["voltages"].to_list())
    voltages_neg_np = np.array(df_negatives["voltages"].to_list())
    images_pos = df_positives["images"].to_list()

    df = pd.DataFrame(columns=["radius", "snr"])

    for i, voltage in enumerate(voltages_pos_np):
        # Calculate voltage differences
        img = images_pos[i]
        target_position = find_center_of_mass(img)
        # calculate radius from the center of the image
        radius = np.sqrt((target_position[0] - img.shape[0] / 2) ** 2 + (target_position[1] - img.shape[1] / 2) ** 2)
        # draw a line from the center to the target
        # plt.imshow(img)
        # plt.scatter(target_position[0], target_position[1], c="red")
        # plt.title(f"Radius: {radius}")
        # plt.show()
        # print(i)
        snr = get_snr_of_sample(i, voltages_pos_np, voltages_neg_np)
        df = pd.concat([df, pd.DataFrame({"radius": radius, "snr": snr, "index": i,
                                          "timestamp": df_positives["timestamp"].iat[i]}, index=[i])])

    # Convert Pixels to mm
    df["radius"] = df["radius"].apply(lambda x: x / (img.shape[0]) * RADIUS_TANK)
    # # plot snr over radius
    # plt.scatter(df["radius"], df["snr"])
    # plt.title("SNR over Radius")
    # plt.xlabel("Radius (mm)")
    # plt.ylabel("SNR")
    # plt.show()
    # calculate snr in db
    df["snr_db"] = 20 * np.log10(df["snr"])
    # plot snr in db over radius
    # plt.scatter(df["radius"], df["snr_db"], c=df["timestamp"])
    # plt.colorbar()
    # # add color codeded timestamp
    # plt.title("SNR (dB) over Radius")
    # plt.xlabel("Radius (mm)")
    # plt.ylabel("SNR (dB)")
    # tikzplotlib.save("snr_over_radius.tex")
    # plt.show()
    # same with seaborn
    # convert timestamp to show only every hour
    df["timestamp"] = df["timestamp"].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
    sns.scatterplot(data=df, x="radius", y="snr_db", hue="timestamp")
    plt.title("SNR (dB) over Radius")
    plt.xlabel("Radius (mm)")
    plt.ylabel("SNR (dB)")
    plt.show()


if __name__ == "__main__":
    path_to_negatives = "Trainings_Data_EIT32/1_Freq/326_sampels_negative.pkl"
    path_to_positives = "Trainings_Data_EIT32/1_Freq_More_Orientations/combined.pkl"
    # path_to_negatives = "Collected_Data/Combined_dataset/data_244_sampels_negative_set.pkl"
    # path_to_positives = "Collected_Data/Combined_dataset/data_541_samples_25_08_2023_40mm.pkl"
    # path_to_negatives = "Trainings_Data_EIT32/1_Freq/326_sampels_negative.pkl"
    # path_to_positives = "Collected_Data/Train_set_15_11_40mm_eit32_Kartoffel/combined.pkl"
    # path_to_negatives = "Trainings_Data_EIT32/3_Freq/252_sampels_negative.pkl"
    # path_to_positives = "Trainings_Data_EIT32/3_Freq/810_samples_40mm.pkl"

    evaluate_snr_over_radius(path_to_positives, path_to_negatives)

    # calculate_snr_in_dataset(path_to_positives, path_to_negatives)
    # Amplitude_Signal = np.array([400])
    # Amplitude_Noise = np.array([10])
    # print(calculate_snr(Amplitude_Signal, Amplitude_Noise))
