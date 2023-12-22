import datetime
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Data_Generation.utils import generate_random_anomaly_list, get_newest_file, wait_for_n_secs_with_print, \
    solve_eit_using_jac, wait_1_file_and_get_next, calibration_procedure, load_model_from_path
from G_Code_Device.GCodeDevice import GCodeDevice, list_serial_devices
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df, convert_multi_frequency_eit_to_df
from plot_utils import solve_and_plot_with_nural_network
from pyeit import mesh
from pyeit.eit import protocol
from utils import wait_for_start_of_measurement

"""How to use this script:
1. Connect the Sciospec device to the computer
2. Connect the G-Code device to the computer
3. Run this script
4. Start the measurement on the Sciospec device
5. G-Code device will move to the first position
"""

TIME_FORMAT = "%Y-%m-%d %H_%M_%S"

n_el = 32

RADIUS_TARGET_IN_MM = 40
RADIUS_TANK_IN_MM = 190

img_size = 64

RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM

# METADATA
TARGET = "CYLINDER"
MATERIAL_TARGET = "PLA"
TANK_ORIENTATION = "Klebeband auf Elektrode 26"
VOLTAGE_FREQUENCY = "1KHZ - 1MHZ"
NUMBER_OF_FREQUENCIES = 3
CURRENT = 0.1
CONDUCTIVITY_BG = 1000  # in S/m     # TODO: Measure this
CONDUCTIVITY_TARGET = 0.1  # in S/m
EIT_32_used = True

model_pca_path = "../Trainings_Data_EIT32/3_Freq_Even_orientation/Models/LinearModelWithDropout2/Test_without_superposition/model_2023-12-13_14-17-56_69_70.pth"
model, pca, normalize = load_model_from_path(path=model_pca_path, normalize=False)


def collect_one_sample(gcode_device: GCodeDevice, eit_path: str, last_position: np.ndarray):
    """
    Generates a sample simulation of electrode voltages with a random anomaly.
    """
    """ 1. problem setup """
    anomaly_list = generate_random_anomaly_list(max_number_of_anomalies=1, min_radius=RELATIVE_RADIUS_TARGET,
                                                max_radius=RELATIVE_RADIUS_TARGET, min_perm=1000,
                                                max_perm=1000, outer_circle_radius=1 - RELATIVE_RADIUS_TARGET)

    if len(anomaly_list) > 1:
        raise Exception("More than one anomaly generated")

    """ 2. generate corresponding image """
    img = np.zeros([img_size, img_size])
    # set to 1 the pixels corresponding to the anomaly unsing cv2.circle
    anomaly = anomaly_list[0]
    center = np.array((anomaly.center[0], anomaly.center[1]))
    # map center from [-1, 1] to [0, img_size]
    center_for_image = (center + 1) * img_size / 2
    center_for_image = center_for_image.astype(int)
    if gcode_device is not None:
        cv2.circle(img, tuple(center_for_image), int(anomaly.r * img_size / 2), 1, -1)
    # flip the image vertically because the mesh is flipped vertically
    img = np.flip(img, axis=0)

    PLOT = True
    if PLOT:
        img_show = img.copy()
        # plot big circle
        # convert to color image
        img_show = np.stack([img_show, img_show, img_show], axis=2)
        cv2.circle(img_show, (img_size // 2, img_size // 2), int(img_size / 2), (255, 0, 255), 1)
        cv2.imshow("Target Location", cv2.resize(img_show, (256, 256)))
        cv2.waitKey(100)

    """ 3. send gcode to the device """
    if gcode_device is not None:
        # convert center from [-1, 1] to [0, max_moving_space]
        center_for_moving = (center + 1) * gcode_device.maximal_limits[0] / 2
        # invert x axis
        center_for_moving[0] = gcode_device.maximal_limits[0] - center_for_moving[0]
        center_for_moving = center_for_moving.astype(int)
        print("center_for_moving", center_for_moving)
        gcode_device.move_to(x=center_for_moving[0], y=0, z=center_for_moving[1])
        move_time = gcode_device.calculate_moving_time(last_position,
                                                       center_for_moving)
        wait_for_n_secs_with_print(move_time)
    else:
        time.sleep(2)
        center_for_moving = last_position

    """ 4. collect data """
    # get the newest file in the folder
    file_path = wait_1_file_and_get_next(eit_path)
    print(file_path)
    df = convert_multi_frequency_eit_to_df(file_path)
    df_alternating = pd.DataFrame({"real": df["real"], "imaginary": df["imaginary"]}).stack().reset_index(drop=True)
    df_alternating = df_alternating.to_frame(name="amplitude")
    v1 = df_alternating["amplitude"].to_numpy(dtype=np.float64)
    return img, v1, center_for_moving


def collect_data(gcode_device: GCodeDevice, number_of_samples: int, eit_data_path: str, save_path: str):
    """
    Collects a number of samples.
    :param gcode_device:
    :param number_of_samples:
    :return:
    """
    # create txt file with the metadata
    metadata = {"number_of_samples": number_of_samples, "img_size": img_size, "n_el": n_el,
                "target": TARGET, "material_target": MATERIAL_TARGET, "voltage_frequency": VOLTAGE_FREQUENCY,
                "radius_target_in_mm": RADIUS_TARGET_IN_MM, "radius_tank_in_mm": RADIUS_TANK_IN_MM,
                "conductivity_bg": CONDUCTIVITY_BG, "conductivity_target": CONDUCTIVITY_TARGET,
                "current": CURRENT, "number_of_freqs": NUMBER_OF_FREQUENCIES, "eit_32_used": EIT_32_used,
                "tank orientation": TANK_ORIENTATION,
                }
    with open(os.path.join(save_path, "metadata.txt"), 'w') as file:
        file.write(json.dumps(metadata))
    images = []
    voltages = []
    timestamps = []
    if gcode_device is None:
        last_centers = [np.array([0, 0])]
    else:
        last_centers = [np.array([gcode_device.maximal_limits[0] / 2, gcode_device.maximal_limits[2] / 2])]
    eit_path = wait_for_start_of_measurement(
        eit_data_path)  # Wait for the start of the measurement and return the path to the data
    time.sleep(1)
    for i in range(number_of_samples):
        img, v1, center_for_moving = collect_one_sample(gcode_device=gcode_device, eit_path=eit_path,
                                                        last_position=last_centers[-1])
        if pca is not None:
            v1_plot = pca.transform(v1.reshape(1, -1))
        solve_and_plot_with_nural_network(model=model, model_input=v1_plot, chow_center_of_mass=False,
                                          use_opencv_for_plotting=True)
        images.append(img)
        voltages.append(v1)
        timestamps.append(datetime.datetime.now())
        #
        last_centers.append(center_for_moving)
        print(f"Sample {i} collected")
        # save the images and voltages in a dataframe every 10 samples
        if i % 10 == 0:
            df = pd.DataFrame(
                {"timestamp": timestamps, "images": images, "voltages": voltages})
            save_path_data = os.path.join(save_path,
                                          f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
            df.to_pickle(save_path_data)
            print(f"Saved data to {save_path_data}")
            images = []
            voltages = []
            timestamps = []
    # save the images and voltages in a dataframe
    df = pd.DataFrame({"images": images, "voltages": voltages})
    save_path_data = os.path.join(save_path, f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
    df.to_pickle(save_path_data)
    print(f"Saved data to {save_path_data}")


def collect_data_circle_pattern(gcode_device: GCodeDevice, number_of_runs: int, eit_data_path: str, save_path: str,
                                debug_plots: bool = True):
    """
    Moves the target in circular pattern at multiple radii and collects the data.
    :param number_of_runs:
    :param save_path:
    :param eit_data_path:
    :param gcode_device:
    :param debug_plots:
    :return:
    """
    # create txt file with the metadata
    metadata = {"number_of_samples": number_of_runs, "img_size": img_size, "n_el": n_el,
                "target": TARGET, "material_target": MATERIAL_TARGET, "voltage_frequency": VOLTAGE_FREQUENCY,
                "radius_target_in_mm": RADIUS_TARGET_IN_MM, "radius_tank_in_mm": RADIUS_TANK_IN_MM,
                "conductivity_bg": CONDUCTIVITY_BG, "conductivity_target": CONDUCTIVITY_TARGET,
                "current": CURRENT, "number_of_freqs": NUMBER_OF_FREQUENCIES,
                }
    with open(os.path.join(save_path, "metadata.txt"), 'w') as file:
        file.write(json.dumps(metadata))
    images = []
    voltages = []
    timestamps = []
    """ Crate Circle Pattern """
    degree_resolution = 20
    radii = np.linspace(0.1, 1 - RELATIVE_RADIUS_TARGET - 0.05, 5)
    # reverse the order of the radii
    radii = radii[::-1]
    num_of_angles = 360 // degree_resolution
    angles = np.linspace(0, 2 * np.pi, num_of_angles)
    # plot the circle pattern
    if debug_plots:
        plt.figure()
        for radius in radii:
            plt.plot(radius * np.cos(angles), radius * np.sin(angles), 'o')
        plt.show()

    last_centers = [np.array([gcode_device.maximal_limits[0] / 2, gcode_device.maximal_limits[2] / 2])]
    eit_path = wait_for_start_of_measurement(
        eit_data_path)  # Wait for the start of the measurement and return the path to the data
    time.sleep(1)
    overall_nr_of_samples = len(radii) * len(angles) * number_of_runs

    i = 0
    for a in range(0, number_of_runs):
        for radius in radii:
            print(f"Measuring at radius: {radius}")
            for angle in angles:
                print(f"Measuring at radius: {radius}, angle: {angle}")
                print(f"Sample {i} of {overall_nr_of_samples}")
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                center = np.array([x, y])
                center_for_moving = (center + 1) * gcode_device.maximal_limits[0] / 2
                # invert x axis
                center_for_moving[0] = gcode_device.maximal_limits[0] - center_for_moving[0]
                center_for_moving = center_for_moving.astype(int)
                gcode_device.move_to(x=center_for_moving[0], y=0, z=center_for_moving[1])
                move_time = gcode_device.calculate_moving_time(last_centers[-1],
                                                               center_for_moving)
                wait_for_n_secs_with_print(move_time)  # 1 seconds for safety and measurement
                last_centers.append(center_for_moving)
                if i == 0:
                    last_centers = last_centers[1:]
                    # wait for the first movement to finish
                    time.sleep(1)
                i += 1
                """ 4. collect data """
                # get the newest file in the folder
                file_path = wait_1_file_and_get_next(eit_path)
                print(file_path)
                time.sleep(0.1)
                df = convert_multi_frequency_eit_to_df(file_path)
                df_alternating = pd.DataFrame({"real": df["real"], "imaginary": df["imaginary"]}).stack().reset_index(
                    drop=True)
                df_alternating = df_alternating.to_frame(name="amplitude")
                v1 = df_alternating["amplitude"].to_numpy(dtype=np.float64)
                if pca is not None:
                    v1_plot = pca.transform(v1.reshape(1, -1))
                solve_and_plot_with_nural_network(model=model, model_input=v1_plot, chow_center_of_mass=False,
                                                  use_opencv_for_plotting=True)

                """ 5. create image """
                img = np.zeros([img_size, img_size])
                # set to 1 the pixels corresponding to the anomaly unsing cv2.circle
                # map center from [-1, 1] to [0, img_size]
                center_for_image = (center + 1) * img_size / 2
                center_for_image = center_for_image.astype(int)
                if gcode_device is not None:
                    cv2.circle(img, tuple(center_for_image), int(RELATIVE_RADIUS_TARGET * img_size / 2), 1, -1)
                # flip the image vertically because the mesh is flipped vertically
                img = np.flip(img, axis=0)
                PLOT = True
                if PLOT:
                    img_show = img.copy()
                    # plot big circle
                    # convert to color image
                    img_show = np.stack([img_show, img_show, img_show], axis=2)
                    cv2.circle(img_show, (img_size // 2, img_size // 2), int(img_size / 2), (255, 0, 255), 1)
                    cv2.imshow("Target Location", cv2.resize(img_show, (256, 256)))
                    cv2.waitKey(100)
                images.append(img)
                voltages.append(v1)
                timestamps.append(datetime.datetime.now())
                print(f"Sample {i} collected")
                # save the images and voltages in a dataframe every 10 samples
                df = pd.DataFrame(
                    {"timestamp": timestamps, "images": images, "voltages": voltages})
                save_path_data = os.path.join(save_path,
                                              f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
                df.to_pickle(save_path_data)
                print(f"Saved data to {save_path_data}")
                images = []
                voltages = []
                timestamps = []
    # save the images and voltages in a dataframe
    df = pd.DataFrame(
        {"timestamp": timestamps, "images": images, "voltages": voltages})
    save_path_data = os.path.join(save_path,
                                  f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
    df.to_pickle(save_path_data)
    print(f"Saved data to {save_path_data}")
    # move to the center
    gcode_device.move_to(x=gcode_device.maximal_limits[0] / 2, y=0, z=gcode_device.maximal_limits[2] / 2)


def collect_data_pattern_in_csv(gcode_device: GCodeDevice, eit_data_path: str, save_path: str,
                                df_coords_complete: pd.DataFrame):
    images = []
    voltages = []
    timestamps = []
    overall_nr_of_samples = len(df_coords_complete)

    last_centers = [np.array([gcode_device.maximal_limits[0] / 2, gcode_device.maximal_limits[2] / 2])]
    radii = df_coords_complete["radius"].unique()
    # reverse the order of the radii
    radii = radii[::-1]
    # get the newest file in the folder
    eit_path = wait_for_start_of_measurement(
        eit_data_path)
    # collect v0
    # x_positions = df_coords_complete["x"].to_numpy()
    #
    # # bin in x direction by bin sizes of 0.1
    # bin_size = 0.1
    # bins = np.arange(0, 1 + bin_size, bin_size)
    #
    # # only plot the points with x positions in bin 1
    # x_positions = x_positions[(x_positions >= bins[1]) & (x_positions < bins[2])]
    # # get the corresponding y positions
    # y_positions = df_coords_complete[df_coords_complete["x"].isin(x_positions)]["y"].to_numpy()
    #
    # plt.figure()
    # plt.plot(x_positions, y_positions, 'o')
    # plt.show()

    # Assuming df is your DataFrame
    # Create bins and add a new column 'Y_bin' to store the bin labels
    bin_size = 0.05
    df_coords_complete['Y_bin'] = pd.cut(df_coords_complete['y'], bins=np.arange(min(df_coords_complete['y']), max(
        df_coords_complete['y']) + bin_size, bin_size), labels=False)

    # Filter out NaN values (if any)
    df_filtered = df_coords_complete.dropna(subset=['Y_bin'])

    # Plot each bin separately
    for bin_label, group in df_filtered.groupby('Y_bin'):
        plt.scatter(group['x'], group['y'], label=f'Bin {bin_label}')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    i = 0
    j = 0
    # for radius in radii:
    #     angles = df_coords_complete[df_coords_complete["radius"] == radius]["angles"].to_numpy()
    #     for angle in angles:
    for bin_label, group in df_filtered.groupby('Y_bin'):
        # sort the group by x position
        if j % 2 == 1:
            group = group.sort_values(by=['x'])
        else:
            group = group.sort_values(by=['x'], ascending=False)  # reverse the order, to move in zig zag pattern
        radii = group["radius"].to_numpy()
        angles = group["angles"].to_numpy()
        j += 1
        for radius, angle in zip(radii, angles):
            # skip the first 560 samples  # TODO: For the case if collection got interrupted
            if i < 400:
                i += 1
                print(f"Skipping sample {i}/{overall_nr_of_samples}")
                continue
            print(f"Measuring at radius: {radius}, angle: {angle}")
            center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            center_for_moving = (center + 1) * gcode_device.maximal_limits[0] / 2
            # invert x axis
            center_for_moving[0] = gcode_device.maximal_limits[0] - center_for_moving[0]
            center_for_moving = center_for_moving.astype(int)
            gcode_device.move_to(x=center_for_moving[0], y=0, z=center_for_moving[1])
            move_time = gcode_device.calculate_moving_time(last_centers[-1],
                                                           center_for_moving)
            wait_for_n_secs_with_print(move_time)
            last_centers.append(center_for_moving)

            file_path = wait_1_file_and_get_next(eit_path)
            print(file_path)
            time.sleep(0.1)
            df = convert_multi_frequency_eit_to_df(file_path)
            df_alternating = pd.DataFrame({"real": df["real"], "imaginary": df["imaginary"]}).stack().reset_index(
                drop=True)
            df_alternating = df_alternating.to_frame(name="amplitude")
            v1 = df_alternating["amplitude"].to_numpy(dtype=np.float64)
            # Solve with trained model
            v1_view = pca.transform(v1.reshape(1, -1)) if pca is not None else v1
            solve_and_plot_with_nural_network(model=model, model_input=v1_view,
                                              chow_center_of_mass=False,
                                              use_opencv_for_plotting=True)
            """create image """
            img = np.zeros([img_size, img_size])
            # set to 1 the pixels corresponding to the anomaly unsing cv2.circle
            # map center from [-1, 1] to [0, img_size]
            center_for_image = (center + 1) * img_size / 2
            center_for_image = center_for_image.astype(int)
            if gcode_device is not None:
                cv2.circle(img, tuple(center_for_image), int(RELATIVE_RADIUS_TARGET * img_size / 2), 1, -1)
            # flip the image vertically because the mesh is flipped vertically
            img = np.flip(img, axis=0)
            images.append(img)
            voltages.append(v1)
            timestamps.append(datetime.datetime.now())
            i += 1
            print(f"Sample {i}/{overall_nr_of_samples} collected")
            if i % 20 == 0:
                df = pd.DataFrame(
                    {"timestamp": timestamps, "images": images, "voltages": voltages})
                save_path_data = os.path.join(save_path,
                                              f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
                df.to_pickle(save_path_data)
                print(f"Saved data to {save_path_data}")
                images = []
                voltages = []
                timestamps = []
            # save the images and voltages in a dataframe
    df = pd.DataFrame(
        {"timestamp": timestamps, "images": images, "voltages": voltages})
    save_path_data = os.path.join(save_path,
                                  f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
    df.to_pickle(save_path_data)
    print(f"Saved data to {save_path_data}")
    # move to the center
    gcode_device.move_to(x=gcode_device.maximal_limits[0] / 2, y=0, z=gcode_device.maximal_limits[2] / 2)

def main():
    devices = list_serial_devices()
    ender = None
    for device in devices:
        if "USB-SERIAL CH340" in device.description:
            home = input("Do you want to home the device? (y/n)")
            home = True if home == "y" else False
            ender = GCodeDevice(device.device, movement_speed=6000,
                                home_on_init=home
                                )
            MAX_RADIUS = RADIUS_TANK_IN_MM
            ender.maximal_limits = [MAX_RADIUS, MAX_RADIUS, MAX_RADIUS]
            # ask user if he wants to calibrate
            calibrate = input("Do you want to calibrate the device? (y/n)")
            if calibrate == "y":
                calibration_procedure(ender, RADIUS_TARGET_IN_MM)
            else:
                # move to the center
                limit_x = ender.maximal_limits[0]
                limit_z = ender.maximal_limits[2]
                ender.move_to(x=limit_x / 2, y=0, z=limit_z / 2)
                input("Press enter when the device is in the center...")
            break
    if ender is None:
        raise Exception("No Ender 3 found")

    TEST_NAME = "data_22_12_kartoffel_3_freq"
    save_path = f"C:/Users/lgudjons/PycharmProjects/EIT_reconstruction/Collected_Data/{TEST_NAME}"
    # warn if the folder already exists
    if os.path.exists(save_path):
        input("WARNING: The folder already exists. Press enter to continue")
    else:
        os.makedirs(save_path)
    # warn if folder name has other
    # number before mm than the actual radius
    if f"{RADIUS_TARGET_IN_MM}mm" not in TEST_NAME:
        input("WARNING: The folder name does not contain the radius. Press enter to continue")
    # collect_data(gcode_device=ender, number_of_samples=3000,
    #              eit_data_path="C:\\Users\\lgudjons\\Desktop\\eit_data",
    #              save_path=save_path)
    collect_data_circle_pattern(gcode_device=ender, number_of_runs=5,
                                eit_data_path="C:\\Users\\lgudjons\\Desktop\\eit_data",
                                save_path=save_path)

    # df_coords_complete = pd.read_csv("../points.csv")
    # collect_data_pattern_in_csv(gcode_device=ender,
    #                             eit_data_path="C:\\Users\\lgudjons\\Desktop\\eit_data",
    #                             save_path=f"C:/Users/lgudjons/PycharmProjects/EIT_reconstruction/Collected_Data/{TEST_NAME}",
    #                             df_coords_complete=df_coords_complete)


if __name__ == '__main__':
    main()
    # df = pd.read_pickle("../Collected_Data/Test_Set_06_10/Data_measured2023-10-06 13_54_21.pkl")
    # print(df)
