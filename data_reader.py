import cmath
import pickle

import numpy as np
import pandas as pd
import datetime

path = "setup_00007.eit"


class SingleEITInjectionVoltages:
    def __init__(self, electrode1:int, electrode2:int, amplitudes:list, phases:list):
        self.injection_electrode1 = electrode1
        self.injection_electrode2 = electrode2
        self.amplitudes = amplitudes
        self.phases = phases
    def __str__(self):
        return f"SingleEITInjectionVoltages({self.injection_electrode1}, {self.injection_electrode2}, {self.amplitudes}, {self.phases})"


class EITFrame:
    def __init__(self, injection_voltages:list):
        self.injection_voltages = injection_voltages
    def __str__(self):
        return f"EITFrame({self.injection_voltages})"


def read_eit_data(path):
    """
    Reads the data from the given path and returns a dictionary with the following structure:
    Header
    {(injection_electrode1, injection_electrode2): [voltage1, voltage2, ...], ...}
    :param path: path to the .eit file
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
        metadata["Number_of_frequencies"] = float(lines[7].strip())
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
    voltage_dict = read_eit_data(path)
    complex_dict = convert_voltage_dict_to_complex(voltage_dict)
    amplitude_phase, all_amplitudes, all_phases = convert_complex_dict_to_amplitude_phase(complex_dict)
    df = convert_complex_dict_to_dataframe(amplitude_phase)
    # sort by measuring electrode
    df = df.convert_dtypes()
    df = df.sort_values(by=["measuring_electrode", "injection_pos"])
    # reindex
    df = df.reset_index(drop=True)
    df = df.convert_dtypes()
    return df


path = "setup_1_00003.eit"
df = convert_single_frequency_eit_file_to_df(path)
#
print(df)
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




