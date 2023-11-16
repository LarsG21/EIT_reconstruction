import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

from utils import find_center_of_mass


def calculate_voltage_differences(positive_samples, negative_samples):
    """
    Calculates the voltage differences between positive and negative samples.
    The voltage differences are the relevant EIT signals.
    V_Diff = V1 - V0. V0 can be calculated as mean(V0).
    :param positive_samples: NumPy array of positive samples
    :param negative_samples: NumPy array of negative samples
    :return:
    """
    v_diff_pos = positive_samples - negative_samples.mean(axis=0)
    v_diff_neg = negative_samples - negative_samples.mean(axis=0)
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
    v_diff_pos = voltages_pos_np[index] - voltages_neg_np.mean(axis=0)
    amplitude_signal = np.mean(np.abs(v_diff_pos), axis=0)
    v_diff_neg = voltages_neg_np - voltages_neg_np.mean(axis=0)
    amplitude_noise = np.mean(np.abs(v_diff_neg), axis=0)
    snr = amplitude_signal.mean() / amplitude_noise.mean()
    return round(snr, 2)


def calculate_snr_in_dataset(path_to_positives, path_to_negatives):
    """
    Calculates the SNR of the whole dataset.
    :return:
    """
    df_negatives = pd.read_pickle(path_to_negatives)
    df_positives = pd.read_pickle(path_to_positives)
    print(f"Samples positive: {len(df_positives)}")
    print(f"Samples negative: {len(df_negatives)}")

    # Extract data and convert to NumPy arrays
    voltages_pos_np = np.array(df_positives["voltages"].to_list())
    voltages_neg_np = np.array(df_negatives["voltages"].to_list())
    images_pos = df_positives["images"].to_list()

    # Calculate voltage differences
    v_diff_pos, v_diff_neg = calculate_voltage_differences(voltages_pos_np, voltages_neg_np)

    # Calculate average amplitude
    amplitude_signal = np.mean(np.abs(v_diff_pos), axis=0)
    amplitude_noise = np.mean(np.abs(v_diff_neg), axis=0)

    # Calculate SNR
    snr, snr_db = calculate_snr(amplitude_signal, amplitude_noise)

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
        df = pd.concat([df, pd.DataFrame({"radius": radius, "snr": snr, "index": i}, index=[i])])

    # plot snr over radius
    plt.scatter(df["radius"], df["snr"])
    plt.title("SNR over Radius")
    plt.xlabel("Radius")
    plt.ylabel("SNR")
    plt.show()

    # calculate snr in db
    df["snr_db"] = 20 * np.log10(df["snr"])
    # plot snr in db over radius
    plt.scatter(df["radius"], df["snr_db"])
    plt.title("SNR (dB) over Radius")
    plt.xlabel("Radius")
    plt.ylabel("SNR (dB)")
    # fit exponential function
    # popt, pcov = curve_fit(exponential_function, df["radius"], df["snr_db"])
    # plt.plot(df["radius"], exponential_function(df["radius"], *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    path_to_negatives = "Trainings_Data_EIT32/1_Freq/326_sampels_negative.pkl"
    path_to_positives = "Trainings_Data_EIT32/1_Freq/Data_09_11_40mm_eit32_over_night/combined.pkl"

    # evaluate_snr_over_radius(path_to_positives, path_to_negatives)

    calculate_snr_in_dataset(path_to_positives, path_to_negatives)
