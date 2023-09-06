import time

import numpy as np
import pandas as pd
import torch

from Data_Generation.utils import generate_random_anomaly_list, wait_for_n_secs_with_print, get_newest_file, \
    wait_for_start_of_measurement
from Evaluation.evaluation_metrics import evaluate_position_error, calculate_amplitude_response
from G_Code_Device.GCodeDevice import list_serial_devices, GCodeDevice
from Model_Training.Models import LinearModelWithDropout2
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df
from plot_utils import solve_and_plot_cnn
from pyeit.eit import protocol

n_el = 32  # nb of electrodes
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
keep_mask = protocol_obj.keep_ba
df_keep_mask = pd.DataFrame(keep_mask, columns=["keep"])
# Dist_exc is the distance between the excitation and measurement electrodes (in number of electrodes)

RADIUS_TARGET_IN_MM = 40
RADIUS_TANK_IN_MM = 190
RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM

img_size = 64


def collect_one_sample(gcode_device: GCodeDevice, eit_path: str, last_position: np.ndarray):
    anomaly_list = generate_random_anomaly_list(max_number_of_anomalies=1, min_radius=RELATIVE_RADIUS_TARGET,
                                                max_radius=RELATIVE_RADIUS_TARGET, min_perm=1000,
                                                max_perm=1000, outer_circle_radius=1 - RELATIVE_RADIUS_TARGET)

    anomaly = anomaly_list[0]
    center = np.array((anomaly.center[0], anomaly.center[1]))

    # convert center from [-1, 1] to [0, max_moving_space]
    center_for_moving = (center + 1) * gcode_device.maximal_limits[0] / 2
    # invert x axis
    center_for_moving[0] = gcode_device.maximal_limits[0] - center_for_moving[0]
    center_for_moving = center_for_moving.astype(int)
    gcode_device.move_to(x=center_for_moving[0], y=0, z=center_for_moving[1])
    move_time = gcode_device.calculate_moving_time(last_position,
                                                   center_for_moving) + 4  # 4 seconds for safety and measurement
    wait_for_n_secs_with_print(move_time)

    """ 4. collect data """
    # get the newest file in the folder
    file_path = get_newest_file(eit_path)
    print(file_path)
    df_1 = convert_single_frequency_eit_file_to_df(file_path)
    v1 = df_1["amplitude"].to_numpy(dtype=np.float64)
    # v1 = v1[keep_mask]
    difference = (v1 - v0) / v0
    img_reconstructed = solve_and_plot_cnn(model=model, voltage_difference=difference, chow_center_of_mass=False)
    return img_reconstructed, v1, center_for_moving


def compare_multiple_positions(gcode_device: GCodeDevice, number_of_samples: int, eit_data_path: str):
    """
    Collects a number of samples.
    :param gcode_device:
    :param number_of_samples:
    :return:
    """
    last_centers = [np.array([gcode_device.maximal_limits[0] / 2, gcode_device.maximal_limits[2] / 2])]
    eit_path = wait_for_start_of_measurement(
        eit_data_path)  # Wait for the start of the measurement and return the path to the data
    time.sleep(4)
    file_path = get_newest_file(eit_path)
    print(file_path)
    # save df to pickle
    time.sleep(1)
    position_errors = []
    amplitude_responses = []
    for i in range(number_of_samples):
        img_reconstructed, v1, center_for_moving = collect_one_sample(gcode_device=gcode_device, eit_path=eit_path,
                                                                      last_position=last_centers[-1])
        last_centers.append(center_for_moving)
        position_error = evaluate_position_error(center_for_moving, gcode_device, img_reconstructed,
                                                 relative_radius_target=RELATIVE_RADIUS_TARGET)
        position_errors.append(position_error)

        amplitude_response = calculate_amplitude_response(center_for_moving, gcode_device, img_reconstructed,
                                                          relative_radius_target=RELATIVE_RADIUS_TARGET)
        amplitude_responses.append(amplitude_response)

        # todo: create plots of amplitude response and position errors over space


def main():
    global model, v0
    VOLTAGE_VECTOR_LENGTH = 1024
    OUT_SIZE = 64
    print("Loading the model")
    model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
    model.load_state_dict(torch.load(
        "../Collected_Data/Combined_dataset/Models/LinearModelDropout2/05_09_all_data_40mm_target_and_augmentation_more_noise/model_2023-09-05_15-34-02_epoche_120_of_200_best_model.pth"))
    devices = list_serial_devices()
    ender = None
    for device in devices:
        if "USB-SERIAL CH340" in device.description:
            ender = GCodeDevice(device.device, movement_speed=6000,
                                home_on_init=False
                                )
            MAX_RADIUS = RADIUS_TANK_IN_MM - RADIUS_TARGET_IN_MM / 2 + 1
            ender.maximal_limits = [MAX_RADIUS, MAX_RADIUS, MAX_RADIUS]
            # calibration_procedure(ender)
            break
    if ender is None:
        raise Exception("No Ender 3 found")
    else:
        print("Ender 3 found")
    # v0_df = convert_single_frequency_eit_file_to_df("v0.eit")
    # v0 = v0_df["amplitude"].to_numpy(dtype=np.float64)
    v0 = np.load("v0.npy")
    # v0 = v0[keep_mask]

    compare_multiple_positions(gcode_device=ender, number_of_samples=4000,
                               eit_data_path="../eit_data", )


if __name__ == '__main__':
    main()
