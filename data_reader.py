import cmath
import pickle
import time
import timeit

import numpy as np
import pandas as pd
import datetime



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
                                                        index_start_measurement_channels + i + 1 + metadata["Number_of_frequencies"]]
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
    Converts the voltages dictionary from multi frequency to a dataframe
    """
    col_names = ["frequency", "injection_pos", "injection_neg", "measuring_electrode", "real", "imaginary"]
    df = pd.DataFrame(columns=col_names)
    for key, values in voltage_dict.items():
        for frequency, voltages in values[0].items():
            for i in range(0, len(voltages), 2):
                df_new_row = pd.DataFrame([[frequency, key[0], key[1], int((i+1)/2 + 1), voltages[i], voltages[i+1]]], columns=col_names)
                df = pd.concat([df, df_new_row], ignore_index=True)
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
    dictionary = _read_eit_data_multi_frequency(path)
    df = _convert_multi_frequency_voltage_dict_to_dataframe(dictionary)
    df = _convert_cols_to_complex(df)
    df = df.sort_values(by=["measuring_electrode", "injection_pos"])
    # reindex
    df = df.reset_index(drop=True)
    return df



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
            complex_values.append(complex(values[i], values[i+1]))
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

def convert_complex_dict_to_dataframe(dict):
    """
    Converts the dictionary to a dataframe
    """
    col_names = ["injection_pos", "injection_neg", "measuring_electrode", "amplitude", "phase"]
    df = pd.DataFrame(columns=col_names)
    for key, values in dict.items():
        for i, value in enumerate(values):
            new_row = pd.DataFrame({"injection_pos":key[0]-1, "injection_neg":key[1]-1, "measuring_electrode": i, "amplitude": value[0], "phase": value[1]},index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
    return df

def convert_single_frequency_eit_file_to_df(path):
    """
    Converts a single frequency EIT file to a dataframe
    """
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


if __name__ == '__main__':

    path_single = "sample_eit_frames/setup_00001.eit"
    # time1 = timeit.timeit(lambda: convert_single_frequency_eit_file_to_df(path_single), number=10)
    # print("Time of single frequency conversion: ", time1)
    # #
    path_multi = "sample_eit_frames/setup_1_00001.eit"
    # time = timeit.timeit(lambda: convert_multi_frequency_eit_to_df(path_single), number=10)
    # print("Time of multi frequency conversion: ", time)



    df_single = convert_single_frequency_eit_file_to_df(path_single)
    df_multi = convert_multi_frequency_eit_to_df(path_multi)

    df = convert_multi_frequency_eit_to_df(path_single)

    print(df)


# print(dict)
# print(out)
# df_complex = convert_cols_to_complex(out)
# print(df_complex)
# read_protocol = pickle.load(open("protocol.pickle", "rb"))
# keep_mask = read_protocol.keep_ba
# # reverse keep_mask order
# print(keep_mask)
# frame2 = pd.DataFrame(keep_mask, columns=["keep"])
# df_with_keep_mask = pd.concat([df, frame2], axis=1)
# print(df_with_keep_mask)
#
# # keep only the rows with keep=True
#
# df_only_keep = df_with_keep_mask[df_with_keep_mask["keep"] == True].drop("keep", axis=1)
# print(df_only_keep)
#
# # get col amplitude as list and save as pickle
# amplitudes = df_only_keep["amplitude"].tolist()
# # convert to numpy array
# amplitudes = np.array(amplitudes)
# pickle.dump(amplitudes, open("v1.pickle", "wb"))




