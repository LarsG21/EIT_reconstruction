import datetime
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_Generation.utils import generate_random_anomaly_list, get_newest_file, wait_for_n_secs_with_print, \
    solve_eit_using_jac, calibration_procedure, wait_1_file_and_get_next, add_electrode_normalizations
from G_Code_Device.GCodeDevice import GCodeDevice, list_serial_devices
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df
from pyeit import mesh
from pyeit.eit import protocol
from pyeit.eit.fem import EITForward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from reconstruction_algorithims import solve_and_plot_greit
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
mesh_obj = mesh.create(n_el, h0=0.1)
dist_exc = 1
step_meas = 1
protocol_obj = protocol.create(n_el, dist_exc=dist_exc, step_meas=step_meas, parser_meas="std")
# Dist_exc is the distance between the excitation and measurement electrodes (in number of electrodes)

keep_mask = protocol_obj.keep_ba
df_keep_mask = pd.DataFrame(keep_mask, columns=["keep"])

RADIUS_TARGET_IN_MM = 40
RADIUS_TANK_IN_MM = 190

img_size = 64

RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM

v0 = None

# METADATA
TARGET = "CYLINDER"
MATERIAL_TARGET = "PLA"
VOLTAGE_FREQUENCY = 1000
CURRENT = 0.1
CONDUCTIVITY_BG = 1000  # in S/m     # TODO: Measure this
CONDUCTIVITY_TARGET = 0.1  # in S/m


# TODO: Add some kind of metadata to the dataframes like Target used, Tank used, etc. (Like in ScioSpec Repo)


def collect_one_sample(gcode_device: GCodeDevice, eit_path: str, last_position: np.ndarray,
                       debug_plots: bool = False):
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
        center_for_moving = (center + 1) * RADIUS_TANK_IN_MM / 2
        # invert x axis
        center_for_moving[0] = gcode_device.maximal_limits[0] - center_for_moving[0]
        center_for_moving = center_for_moving.astype(int)
        print("center_for_moving", center_for_moving)
        gcode_device.move_to(x=center_for_moving[0], y=0, z=center_for_moving[1])
        move_time = gcode_device.calculate_moving_time(last_position,
                                                       center_for_moving)  # 4 seconds for safety and measurement
        wait_for_n_secs_with_print(move_time)
    else:
        time.sleep(2)
        center_for_moving = last_position
    """ 4. collect data """
    # get the newest file in the folder
    file_path = wait_1_file_and_get_next(eit_path)
    print(file_path)
    df_1 = convert_single_frequency_eit_file_to_df(file_path)
    v1 = df_1["amplitude"].to_numpy(dtype=np.float64)

    """ 5. solve EIT using Jac"""
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly_list)
    # select the relevant voltages
    v0_solve = v0[keep_mask]
    v1_solve = v1[keep_mask]
    # subtract the mean  HIGHLIGHT: DONT DO THIS !!
    solve_eit_using_jac(mesh_new, mesh_obj, protocol_obj, v1_solve, v0_solve)
    if debug_plots:
        plt.plot(v1)
        plt.plot(v0)
        plt.title("v1 and v0")
        plt.legend(["v1", "v0"])
        plt.show()
        plt.plot((v1 - v0) / v0)
        plt.title("relative difference")
        plt.show()
    return img, v1, center_for_moving


def collect_data(gcode_device: GCodeDevice, number_of_samples: int, eit_data_path: str, save_path: str):
    """
    Collects a number of samples.
    :param gcode_device:
    :param number_of_samples:
    :return:
    """
    global v0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create txt file with the metadata
    metadata = {"number_of_samples": number_of_samples, "img_size": img_size, "n_el": n_el,
                "target": TARGET, "material_target": MATERIAL_TARGET, "voltage_frequency": VOLTAGE_FREQUENCY,
                "radius_target_in_mm": RADIUS_TARGET_IN_MM, "radius_tank_in_mm": RADIUS_TANK_IN_MM,
                "conductivity_bg": CONDUCTIVITY_BG, "conductivity_target": CONDUCTIVITY_TARGET,
                "current": CURRENT, "dist_exc": dist_exc, "step_meas": step_meas,
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
    file_path = wait_1_file_and_get_next(eit_path)
    print(file_path)
    os.chdir(cwd)
    time.sleep(1)
    # collect v0:
    input("Remove the target and press enter to start the measurement...")
    file_path = wait_1_file_and_get_next(eit_path)
    print(file_path)
    df_0 = convert_single_frequency_eit_file_to_df(file_path)
    v0 = df_0["amplitude"].to_numpy(dtype=np.float64)
    # save v0
    np.save(os.path.join(save_path, "v0.npy"), v0)
    input("Place the target and press enter to start the measurement...")

    for i in range(number_of_samples):
        img, v1, center_for_moving = collect_one_sample(gcode_device=gcode_device, eit_path=eit_path,
                                                        last_position=last_centers[-1])
        images.append(img)
        voltages.append(v1)
        timestamps.append(datetime.datetime.now())
        last_centers.append(center_for_moving)
        print(f"Sample {i} collected")
        # save the images and voltages in a dataframe every 10 samples
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
    df = pd.DataFrame({"images": images, "voltages": voltages})
    save_path_data = os.path.join(save_path, f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
    df.to_pickle(save_path_data)


def collect_data_circle_pattern(gcode_device: GCodeDevice, number_of_runs: int, eit_data_path: str, save_path: str,
                                debug_plots: bool = True):
    """
    Moves the target in circular pattern at multiple radii and collects the data.
    :param number_of_runs:
    :param save_path:
    :param eit_data_path:
    :param gcode_device:
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create txt file with the metadata
    metadata = {"number_of_samples": number_of_runs, "img_size": img_size, "n_el": n_el,
                "target": TARGET, "material_target": MATERIAL_TARGET, "voltage_frequency": VOLTAGE_FREQUENCY,
                "radius_target_in_mm": RADIUS_TARGET_IN_MM, "radius_tank_in_mm": RADIUS_TANK_IN_MM,
                "conductivity_bg": CONDUCTIVITY_BG, "conductivity_target": CONDUCTIVITY_TARGET,
                "current": CURRENT, "dist_exc": dist_exc, "step_meas": step_meas,
                }
    with open(os.path.join(save_path, "metadata.txt"), 'w') as file:
        file.write(json.dumps(metadata))
    images = []
    voltages = []
    timestamps = []
    """ Crate Circle Pattern """
    degree_resolution = 15
    radii = np.linspace(0.2, 1 - RELATIVE_RADIUS_TARGET - 0.05, 4)
    # radii = np.array([0.1, 0.3, 0.5, 0.6, 1 - RELATIVE_RADIUS_TARGET - 0.05])
    # reverse the order of the radii
    radii = radii[::-1]
    num_of_angles = 360 // degree_resolution
    angles = np.linspace(0, 2 * np.pi, num_of_angles)
    # print overall number of samples
    print(f"Number of samples that will be collected: {len(radii) * len(angles) * number_of_runs}")

    last_centers = [np.array([gcode_device.maximal_limits[0] / 2, gcode_device.maximal_limits[2] / 2])]
    eit_path = wait_for_start_of_measurement(
        eit_data_path)  # Wait for the start of the measurement and return the path to the data
    time.sleep(1)

    input("Remove the target and press enter to start the measurement...")
    file_path = wait_1_file_and_get_next(eit_path)
    print(file_path)
    df_0 = convert_single_frequency_eit_file_to_df(file_path)
    v0 = df_0["amplitude"].to_numpy(dtype=np.float64)
    # save v0
    np.save(os.path.join(save_path, "v0.npy"), v0)
    input("Place the target and press enter to start the measurement...")

    # v0 = np.load("../Collected_Data/Test_Set_1_Freq_23_10_circular/v0.npy")

    i = 0
    for a in range(0, number_of_runs):
        print(f"Run {a} of {number_of_runs}")
        for radius in radii:
            print(f"Measuring at radius: {radius}")
            for angle in angles:
                print(f"Measuring at radius: {radius}, angle: {angle}")
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                center = np.array([x, y])
                anomaly = PyEITAnomaly_Circle(center=center, r=RELATIVE_RADIUS_TARGET, perm=CONDUCTIVITY_TARGET)
                mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=CONDUCTIVITY_BG)
                fwd = EITForward(mesh_obj, protocol_obj)
                v0_simulated = fwd.solve_eit()
                v1_simulated = fwd.solve_eit(perm=mesh_new.perm)
                if debug_plots:
                    # plt.plot(v1_simulated)
                    # plt.plot(v0_simulated)
                    # plt.title("v1 and v0")
                    # plt.legend(["v1_simulated", "v0_simulated"])
                    # plt.show()
                    plt.plot((v1_simulated - v0_simulated) / v0_simulated)
                    plt.title("relative difference simulated")
                    plt.show()
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
                    time.sleep(4)
                i += 1
                """ 4. collect data """
                # get the newest file in the folder
                file_path = wait_1_file_and_get_next(eit_path)
                print(file_path)
                time.sleep(0.1)
                df_1 = convert_single_frequency_eit_file_to_df(file_path)
                v1 = df_1["amplitude"].to_numpy(dtype=np.float64)
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
                """6. solve with trained model """
                # v1 = add_electrode_normalizations(v1=v1, NORMALIZE_PER_ELECTRODE=True) # HIGHLIGHT: DOESNT WORK !!!
                # v0 = add_electrode_normalizations(v1=v0, NORMALIZE_PER_ELECTRODE=True)
                mesh_new = mesh.set_perm(mesh_obj, anomaly=[])
                # select the relevant voltages
                v0_solve = v0[keep_mask]
                v1_solve = v1[keep_mask]
                # subtract the mean # HIGHLIGHT: DONT DO THAT !!!
                if debug_plots:
                    # plt.plot(v1_solve)
                    # plt.plot(v0_solve)
                    # plt.title("v1 and v0")
                    # plt.legend(["v1", "v0"])
                    # plt.show()
                    plt.plot((v1_solve - v0_solve) / v0_solve)
                    plt.title("relative difference measured")
                    plt.show()
                solve_eit_using_jac(mesh_new, mesh_obj, protocol_obj, v1_solve, v0_solve)
                PLOT = False
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




def main():
    print("MAKE SURE THAT YOU DELETE OLD DATA FROM THE EIT_DATA FOLDER !")
    devices = list_serial_devices()
    ender = None
    for device in devices:
        if "USB-SERIAL CH340" in device.description:
            home = input("Do you want to home the device? (y/n)")
            home = True if home == "y" else False
            ender = GCodeDevice(device.device, movement_speed=6000,
                                home_on_init=home
                                )
            MAX_RADIUS = RADIUS_TANK_IN_MM  # half at the top and half at the bottom
            print(f"Maximal limits: {MAX_RADIUS}")
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

    TEST_NAME = "Data_25_10_40mm"
    save_path = f"C:/Users/lgudjons/PycharmProjects/EIT_reconstruction/Collected_Data/{TEST_NAME}"
    if os.path.exists(save_path):
        input("The save path already exists. Press enter to continue...")
        input("Are you really sure? Press enter to continue...")
    collect_data(gcode_device=ender, number_of_samples=4000,
                 eit_data_path="../eit_data",
                 save_path=save_path)
    # collect_data_circle_pattern(gcode_device=ender, number_of_runs=6,
    #                             eit_data_path="../eit_data",
    #                             save_path=f"C:/Users/lgudjons/PycharmProjects/EIT_reconstruction/Collected_Data/{TEST_NAME}")

if __name__ == '__main__':
    cwd = os.getcwd()
    main()
