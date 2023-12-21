import cmath
import pickle
import time
import timeit

import numpy as np
import pandas as pd
import datetime

from matplotlib import pyplot as plt

WAITING_TIME_FILE_WRITE_SINGLE = 0.0001

WAITING_TIME_FILE_WRITE_MULTI = 0.0025
def read_eit_data_single_frequency(path):
    """
    Reads the data from the given path_multi and returns a dictionary with the following structure:
    Header
    {(injection_electrode1, injection_electrode2): [voltage1, voltage2, ...], ...}
    :param path: path_multi to the .eit file
    :return: dictionary in form of {(injection_electrode1, injection_electrode2): [voltage1, voltage2, ...], ...}
    """
    metadata = {}
    with open(path) as f:
        try:
            lines = f.readlines()
            metadata["Number_of_Metadata_Entries"] = int(lines[0].strip())
            metadata["Fiel_Version_Number"] = lines[1].strip()
            metadata["Dataset_Name"] = lines[2].strip()
            metadata["Timestamp"] = lines[3].strip()
            metadata["Minimum_frequency"] = float(lines[4].strip())  # Hz
            metadata["Maximum_frequency"] = float(lines[5].strip())  # Hz
            metadata["Frequency_scale"] = lines[6].strip()  # 0 = linear, 1 = logarithmic
            metadata["Number_of_frequencies"] = int(float(lines[7].strip()))
            metadata["Amplitude"] = float(lines[8].strip())  # A
            metadata["FPS"] = float(lines[9].strip())  # Frames per second
            metadata["Phase_correction_parameter"] = float(lines[10].strip())
        except IndexError:
            print("IndexError: ", path)
            print("Trying to read again...")
            time.sleep(0.01)
            read_eit_data_single_frequency(path)
        # find line with "Measurement channels"
        index_start_measurement_channels = 0
        for i, line in enumerate(lines):
            if line.startswith("MeasurementChannels"):
                measurement_channels_str = line.split(":")[1]
                measurement_channels = measurement_channels_str.split(",")
                # strip all and convert to int
                measurement_channels = [int(channel.strip()) for channel in measurement_channels]
                index_start_measurement_channels = i + 2
                break
        data_dict = {}
        current_key = None
        current_values = []
        # The file looks like this:
        # 1 2
        # V1_RE V1_IM V2_RE V2_IM ... VN_RE VN_IM   <-- frequency 1
        # ...
        # 3 4
        # V1_RE V1_IM V2_RE V2_IM ... VN_RE VN_IM   <-- frequency 1
        # ...
        for i, line in enumerate(lines[index_start_measurement_channels:]):
            elements = line.strip().split(" ")
            if len(elements) == 2:
                if current_key is not None:
                    data_dict[current_key] = current_values
                    current_values = []
                current_key = (int(elements[0]), int(elements[1]))
            else:
                elements = line.strip().split("\t")
                current_values.append([float(element) for element in elements])
        # add last key
        data_dict[current_key] = current_values

    return data_dict


def _read_eit_data_multi_frequency(path):
    """
    Reads the data from the given path_multi and returns a dictionary with the following structure:
    Header
    {(injection_electrode1, injection_electrode2): [voltage1, voltage2, ...], ...}
    :param path: path_multi to the .eit file
    :return: dictionary in form of {(injection_electrode1, injection_electrode2): [voltage1, voltage2, ...], ...}
    """
    metadata = {}
    with open(path) as f:
        try:
            lines = f.readlines()
            metadata["Number_of_Metadata_Entries"] = int(lines[0].strip())
            metadata["Fiel_Version_Number"] = lines[1].strip()
            metadata["Dataset_Name"] = lines[2].strip()
            metadata["Timestamp"] = lines[3].strip()
            metadata["Minimum_frequency"] = float(lines[4].strip())  # Hz
            metadata["Maximum_frequency"] = float(lines[5].strip())  # Hz
            metadata["Frequency_scale"] = lines[6].strip()  # 0 = linear, 1 = logarithmic
            metadata["Number_of_frequencies"] = int(float(lines[7].strip()))
            metadata["Amplitude"] = float(lines[8].strip())  # A
            metadata["FPS"] = float(lines[9].strip())  # Frames per second
            metadata["Phase_correction_parameter"] = float(lines[10].strip())
        except IndexError:
            print("IndexError: ", path)
            print("Trying to read again...")
            time.sleep(0.01)
            read_eit_data_single_frequency(path)
        # find line with "Measurement channels"
        index_start_measurement_channels = 0
        for i, line in enumerate(lines):
            if line.startswith("MeasurementChannels"):
                measurement_channels_str = line.split(":")[1]
                measurement_channels = measurement_channels_str.split(",")
                # strip all and convert to int
                measurement_channels = [int(channel.strip()) for channel in measurement_channels]
                index_start_measurement_channels = i + 2
                break
        data_dict = {}
        current_key = None
        current_values = []
        # The file looks like this:
        # 1 2
        # V1_RE V1_IM V2_RE V2_IM ... VN_RE VN_IM   <-- frequency 1
        # V1_RE V1_IM V2_RE V2_IM ... VN_RE VN_IM   <-- frequency 2
        # ...
        # 3 4
        # V1_RE V1_IM V2_RE V2_IM ... VN_RE VN_IM   <-- frequency 1
        # V1_RE V1_IM V2_RE V2_IM ... VN_RE VN_IM   <-- frequency 2
        # ...
        for i, line in enumerate(lines[index_start_measurement_channels:]):
            elements = line.strip().split(" ")
            if len(elements) == 2:
                if current_key is not None:
                    data_dict[current_key] = current_values
                    current_values = []
                current_key = (int(elements[0]), int(elements[1]))
                lines_with_data_for_frequencies = lines[index_start_measurement_channels + i + 1:
                                                        index_start_measurement_channels + i + 1 + metadata[
                                                            "Number_of_frequencies"]]
                # get all frequencies from min, max and number of frequencies
                # example 1000, 2000, 3 -> [1000, 1500, 2000]
                frequencies = np.linspace(metadata["Minimum_frequency"], metadata["Maximum_frequency"],
                                          metadata["Number_of_frequencies"])
                voltage_data_per_frequency = {}
                for j, line_with_data_for_frequency in enumerate(lines_with_data_for_frequencies):
                    voltage_data_per_frequency[frequencies[j]] = line_with_data_for_frequency.strip().split("\t")
                current_values.append(voltage_data_per_frequency)
        # add last key
        data_dict[current_key] = current_values

    return data_dict



def _convert_multi_frequency_voltage_dict_to_dataframe(voltage_dict):
    """
    Converts the voltages dictionary from multi-frequency to a DataFrame
    """
    col_names = ["frequency", "injection_pos", "injection_neg", "measuring_electrode", "real", "imaginary"]
    data = []

    for key, values in voltage_dict.items():
        for frequency, voltages in values[0].items():
            for i in range(0, len(voltages), 2):
                data.append([frequency, key[0], key[1], int((i + 1) / 2 + 1), voltages[i], voltages[i + 1]])

    df = pd.DataFrame(data, columns=col_names)
    df = df.convert_dtypes()
    # convert real and imaginary to float
    df["real"] = df["real"].astype(float)
    df["imaginary"] = df["imaginary"].astype(float)
    return df


def _convert_cols_to_complex(df):
    """
    Converts the real and imaginary columns to complex numbers with python complex
    """
    df["complex"] = df["real"] + 1j * df["imaginary"]
    df["amplitude"] = np.abs(df["complex"])
    df["phase"] = np.angle(df["complex"])

    return df


def convert_multi_frequency_eit_to_df(path):
    """
    Converts a multi frequency eit file to a dataframe
    :param path:
    :return:
    """
    time.sleep(WAITING_TIME_FILE_WRITE_MULTI)  # wait for file to be written
    dictionary = _read_eit_data_multi_frequency(path)
    df = _convert_multi_frequency_voltage_dict_to_dataframe(dictionary)
    df = _convert_cols_to_complex(df)
    df = df.sort_values(by=["measuring_electrode", "injection_pos"])
    # reindex
    df = df.reset_index(drop=True)
    return df


# Single frequency

def read_eit_data_single_frequency(path):
    """
    Reads the data from the given path_multi and returns a dictionary with the following structure:
    Header
    {(injection_electrode1, injection_electrode2): [voltage1, voltage2, ...], ...}
    :param path: path_multi to the .eit file
    :return: dictionary in form of {(injection_electrode1, injection_electrode2): [voltage1, voltage2, ...], ...}
    """
    metadata = {}
    with open(path) as f:
        try:
            lines = f.readlines()
            metadata["Number_of_Metadata_Entries"] = int(lines[0].strip())
            metadata["Fiel_Version_Number"] = lines[1].strip()
            metadata["Dataset_Name"] = lines[2].strip()
            metadata["Timestamp"] = lines[3].strip()
            metadata["Minimum_frequency"] = float(lines[4].strip())  # Hz
            metadata["Maximum_frequency"] = float(lines[5].strip())  # Hz
            metadata["Frequency_scale"] = lines[6].strip()  # 0 = linear, 1 = logarithmic
            metadata["Number_of_frequencies"] = int(float(lines[7].strip()))
            metadata["Amplitude"] = float(lines[8].strip())  # A
            metadata["FPS"] = float(lines[9].strip())  # Frames per second
            metadata["Phase_correction_parameter"] = float(lines[10].strip())
        except IndexError:
            print("IndexError: ", path)
            print("Trying to read again...")
            time.sleep(0.01)
            read_eit_data_single_frequency(path)
        # find line with "Measurement channels"
        index_start_measurement_channels = 0
        for i, line in enumerate(lines):
            if line.startswith("MeasurementChannels"):
                measurement_channels_str = line.split(":")[1]
                measurement_channels = measurement_channels_str.split(",")
                # strip all and convert to int
                measurement_channels = [int(channel.strip()) for channel in measurement_channels]
                index_start_measurement_channels = i + 2
                break
        data_dict = {}
        current_key = None
        current_values = []
        # The file looks like this:
        # 1 2
        # V1_RE V1_IM V2_RE V2_IM ... VN_RE VN_IM   <-- frequency 1
        # ...
        # 3 4
        # V1_RE V1_IM V2_RE V2_IM ... VN_RE VN_IM   <-- frequency 1
        # ...
        for i, line in enumerate(lines[index_start_measurement_channels:]):
            elements = line.strip().split(" ")
            if len(elements) == 2:
                if current_key is not None:
                    data_dict[current_key] = current_values
                    current_values = []
                current_key = (int(elements[0]), int(elements[1]))
            else:
                elements = line.strip().split("\t")
                current_values.append([float(element) for element in elements])
        # add last key
        data_dict[current_key] = current_values

    return data_dict


# Convert voltage dict from IM RE to amplitude and phase
def convert_voltage_dict_to_complex(voltage_dict):
    """
    Converts the voltage dictionary from IM RE to amplitude and phase
    """
    output_dict = {}
    for key, values in voltage_dict.items():
        # values art in the pattern [V1_RE, V1_IM, V2_RE, V2_IM, ..., VN_RE, VN_IM]
        # convert to [V1, V2, ..., VN]
        complex_values = []
        values = values[0]
        for i in range(0, len(values), 2):
            complex_values.append(complex(values[i], values[i + 1]))
            output_dict[key] = complex_values
    return output_dict

    #     # convert to amplitude and phase


def convert_complex_dict_to_amplitude_phase(complex_dict):
    """
    Converts the complex dictionary to amplitude and phase
    """
    output_dict = {}
    all_amplitudes = []
    all_phses = []
    for key, values in complex_dict.items():
        # values art in the pattern [V1, V2, ..., VN]
        # convert to [V1_amp, V1_phase, V2_amp, V2_phase, ..., VN_amp, VN_phase]
        amplitude_phase_values = []
        for value in values:
            amplitude = abs(value)
            phase = cmath.phase(value)
            all_amplitudes.append(amplitude)
            all_phses.append(phase)
            amplitude_phase_values.append((amplitude, phase))
        output_dict[key] = amplitude_phase_values

    return output_dict, all_amplitudes, all_phses


def convert_complex_dict_to_dataframe(data_dict):
    """
    Converts the dictionary to a dataframe
    """
    col_names = ["injection_pos", "injection_neg", "measuring_electrode", "amplitude", "phase"]
    df_rows = []

    for key, values in data_dict.items():
        inj_pos = key[0] - 1
        inj_neg = key[1] - 1
        for i, (amplitude, phase) in enumerate(values):
            df_rows.append((inj_pos, inj_neg, i, amplitude, phase))

    df = pd.DataFrame.from_records(df_rows, columns=col_names)
    return df


def convert_single_frequency_eit_file_to_df(path):
    """
    Converts a single frequency EIT file to a dataframe
    """
    time.sleep(WAITING_TIME_FILE_WRITE_SINGLE)  # wait a bit to avoid file access errors
    voltage_dict = read_eit_data_single_frequency(path)
    complex_dict = convert_voltage_dict_to_complex(voltage_dict)
    amplitude_phase, all_amplitudes, all_phases = convert_complex_dict_to_amplitude_phase(complex_dict)
    df = convert_complex_dict_to_dataframe(amplitude_phase)
    # sort by measuring electrode
    df = df.sort_values(by=["measuring_electrode", "injection_pos"])
    # reindex
    df = df.reset_index(drop=True)
    df = df.convert_dtypes()
    return df


def plot_nyquist(df, title="Nyquist plot"):
    """
    Plots the nyquist plot of the given data
    :param reals:
    :param imags:
    :param frequencies:
    :return:
    """
    frequencies = df["frequency"].unique()
    means = []
    phases = []
    reals = []
    imags = []
    for frequency in frequencies:
        df_frequency = df[df["frequency"] == frequency]
        means.append(df_frequency["amplitude"].mean())
        phases.append(df_frequency["phase"].mean())
        reals.append(df_frequency["real"].mean())
        imags.append(df_frequency["imaginary"].mean())

    plt.plot(reals, imags)
    plt.title(title)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    # write the frequency next to the point
    for i, frequency in enumerate(frequencies):
        # if i % 10 == 0:
        plt.text(reals[i], imags[i], f"{int(frequency / 1000)} kHz")
    plt.show()


def plot_bode(df):
    """
    Plots the bode plot of the given data
    :param df:
    :return:
    """
    frequencies = df["frequency"].unique()
    means = []
    phases = []
    reals = []
    imags = []
    for frequency in frequencies:
        df_frequency = df[df["frequency"] == frequency]
        means.append(df_frequency["amplitude"].mean())
        phases.append(df_frequency["phase"].mean())
        reals.append(df_frequency["real"].mean())
        imags.append(df_frequency["imaginary"].mean())
        # plt.plot(df_frequency["amplitude"])
        # plt.title(frequency)
        # plt.show()

    # add bode plot with amplitude and phase in one plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Amplitude")
    ax1.plot(frequencies, means, color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Phase")
    ax2.plot(frequencies, phases, color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    fig.tight_layout()
    plt.title("Bode plot")
    # log scale
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    plt.show()


if __name__ == '__main__':

    path_single = "../sample_eit_frames/setup_00001.eit"
    # time1 = timeit.timeit(lambda: convert_single_frequency_eit_file_to_df(path_single), number=10)
    # print("Time of single frequency conversion: ", time1)
    # #
    path_multi4 = "../eit_experiments/3_freq_move_Target/setup_1/setup_1_00002.eit"
    # time = timeit.timeit(lambda: convert_multi_frequency_eit_to_df(path_single), number=10)
    # print("Time of multi frequency conversion: ", time)

    # df_single = convert_single_frequency_eit_file_to_df(path_single)
    # number_of_runs = 20
    # time = timeit.timeit(lambda: convert_single_frequency_eit_file_to_df(path_single), number=number_of_runs)
    # print("Time of single frequency conversion: ", time/number_of_runs)

    df = convert_multi_frequency_eit_to_df(path_multi4)
    df_alternating = pd.DataFrame({"real": df["real"], "imaginary": df["imaginary"]}).stack().reset_index(drop=True)
    df_alternating = df_alternating.to_frame(name="amplitude")
    v1 = df_alternating["amplitude"].to_numpy(dtype=np.float64)
    # np.save("v0.npy", v1)

    # print("finished conversion")
    #
    # # # df_multi = convert_multi_frequency_eit_to_df(path_multi)
    # #
    # # # df = convert_multi_frequency_eit_to_df(path_single)
    # # print(df)
    # #
    # # plt.plot(df["amplitude"])
    # # plt.show()
    # #
    # # plot_bode(df)
    #
    # # add nyquist plot
    # # split df in all measuring_electrodes
    # for measuring_electrode in df["measuring_electrode"].unique():
    #     print("Nyquist", measuring_electrode)
    #     df_measuring_electrode = df[df["measuring_electrode"] == measuring_electrode]
    #     plot_nyquist(df_measuring_electrode, title=f"Nyquist plot {measuring_electrode}")
