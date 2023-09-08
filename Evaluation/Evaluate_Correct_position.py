import datetime
import os
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.interpolate import interpolate, griddata

from Data_Generation.collect_real_data import calibration_procedure
from Data_Generation.utils import generate_random_anomaly_list, wait_for_n_secs_with_print, get_newest_file, \
    wait_for_start_of_measurement
from Evaluation.evaluation_metrics import evaluate_position_error, calculate_amplitude_response
from G_Code_Device.GCodeDevice import list_serial_devices, GCodeDevice
from Model_Training.Models import LinearModelWithDropout2
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df
from plot_utils import solve_and_plot_cnn
from pyeit.eit import protocol
import plotly.express as px

RESOLUTION_PLOT = 80
nr_of_blurs = 1

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
    error_vectors = []
    amplitude_responses = []
    for i in range(number_of_samples):
        img_reconstructed, v1, center_for_moving = collect_one_sample(gcode_device=gcode_device, eit_path=eit_path,
                                                                      last_position=last_centers[-1])
        last_centers.append(center_for_moving)
        position_error, error_vector = evaluate_position_error(center_for_moving, gcode_device, img_reconstructed,
                                                               relative_radius_target=RELATIVE_RADIUS_TARGET)
        position_errors.append(position_error)
        error_vectors.append(error_vector)

        amplitude_response = calculate_amplitude_response(center_for_moving, gcode_device, img_reconstructed,
                                                          relative_radius_target=RELATIVE_RADIUS_TARGET)
        amplitude_responses.append(amplitude_response)
        # remove first element of last_centers
        if i == 0:
            last_centers = last_centers[1:]
        # create dataframe
        df = pd.DataFrame(
            data={"positions": last_centers, "position_error": position_errors, "error_vector": error_vectors,
                  "amplitude_response": amplitude_responses})
        path = "C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results"
        save_path = os.path.join(path, f"evaluation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")
        df.to_pickle(save_path)
        print("saved dataframe to pickle")
    return df

    # todo: create plots of amplitude response and position errors over space


def plot_amplitude_response(df: pd.DataFrame, save_path: str = None):
    df["x"] = [x[0] for x in df["positions"]]
    df["y"] = [x[1] for x in df["positions"]]
    # fig = px.scatter(df, x="x", y="y", color="amplitude_response")
    # fig.show()
    # interpolate between points
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["amplitude_response"].to_numpy()
    # plot scatter 2d with color coding
    # fig = px.scatter(df, x="x", y="y", color="amplitude_response")
    # # show
    # fig.show()
    # interpolate
    # Create a grid of points for interpolation
    xi, yi = np.linspace(x.min(), x.max(), RESOLUTION_PLOT), np.linspace(y.min(), y.max(), RESOLUTION_PLOT)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the data using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi = np.abs(zi)

    # Create a heatmap with color coding
    for i in range(nr_of_blurs):
        zi = ndimage.gaussian_filter(zi, sigma=1, radius=1)
    fig = px.imshow(zi, x=xi[0, :], y=yi[:, 0], color_continuous_scale='Viridis')
    fig.update_layout(title='Amplitude response over space', xaxis_title="x [mm]", yaxis_title="y [mm]")
    # text to colorbar
    fig.update_layout(coloraxis_colorbar=dict(
        title="Amplitude response",
        thicknessmode="pixels", thickness=50,
        lenmode="pixels",
        yanchor="top", y=1,
        ticks="outside", ticksuffix="",
        dtick=0.1
    ))
    if save_path is not None:
        fig.write_image(save_path)
    # Show the plot
    fig.show()
    # Create a new DataFrame for interpolated data
    interpolated_df = pd.DataFrame({'x': xi.flatten(), 'y': yi.flatten(), 'amplitude_response': zi.flatten()})
    # Create a scatter plot with interpolated data
    # fig = px.scatter(interpolated_df, x="x", y="y", color="amplitude_response")
    # fig.update_layout(title='Amplitude response over space', xaxis_title="x", yaxis_title="y")
    # # Show the plot
    # fig.show()


def plot_position_error(df: pd.DataFrame, save_path: str = None):
    df["x"] = [x[0] for x in df["positions"]]
    df["y"] = [x[1] for x in df["positions"]]
    # fig = px.scatter(df, x="x", y="y", color="position_error")
    # fig.show()
    # interpolate between points
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["position_error"].to_numpy()
    # fig = px.scatter(df, x="x", y="y", color="position_error")
    # # show
    # fig.show()
    # interpolate
    # Create a grid of points for interpolation
    xi, yi = np.linspace(x.min(), x.max(), RESOLUTION_PLOT), np.linspace(y.min(), y.max(), RESOLUTION_PLOT)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate the data using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    # get absolute value of position error
    zi = np.abs(zi)
    # Create a heatmap with color coding
    for i in range(nr_of_blurs):
        zi = ndimage.gaussian_filter(zi, sigma=1, radius=1)
    fig = px.imshow(zi, x=xi[0, :], y=yi[:, 0], color_continuous_scale='Viridis')
    # title
    fig.update_layout(title="Position error over space", xaxis_title="x [mm]", yaxis_title="y [mm]")
    # text to colorbar
    fig.update_layout(coloraxis_colorbar=dict(
        title="Position error",
        thicknessmode="pixels", thickness=50,
        lenmode="pixels",
        yanchor="top", y=1,
        ticks="outside", ticksuffix="",
        dtick=1
    ))
    if save_path is not None:
        fig.write_image(save_path)
    # Show the plot
    fig.show()
    # Create a new DataFrame for interpolated data
    interpolated_df = pd.DataFrame({'x': xi.flatten(), 'y': yi.flatten(), 'position_error': zi.flatten()})
    # Create a scatter plot with interpolated data
    # fig = px.scatter(interpolated_df, x="x", y="y", color="position_error")
    # # title
    # fig.update_layout(title="Position error over space", xaxis_title="x", yaxis_title="y")
    # # Show the plot
    # fig.show()


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
    v0 = np.load("v0_new.npy")
    # v0 = v0[keep_mask]

    compare_multiple_positions(gcode_device=ender, number_of_samples=400,
                               eit_data_path="../eit_data", )


if __name__ == '__main__':
    main()
    # df = pd.read_pickle("dataframe_evaluation.pkl")
    # plot_amplitude_response(df)
    # plot_position_error(df)

    # v0 = np.load("v0.npy")
    #
    # vo_new = np.load("v0_new.npy")
    #
    # diff = v0 - vo_new

    # plot all
    # plt.plot(v0, label="v0")
    # plt.plot(vo_new, label="v0_new")
    # plt.plot(diff, label="diff")
    # plt.legend()
    # plt.show()
