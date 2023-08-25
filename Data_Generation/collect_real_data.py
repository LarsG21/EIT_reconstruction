import datetime
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from Data_Generation.utils import generate_random_anomaly_list, get_newest_file, wait_for_n_secs_with_print, \
    solve_eit_using_jac
from G_Code_Device.GCodeDevice import GCodeDevice, list_serial_devices
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df
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
mesh_obj = mesh.create(n_el, h0=0.1)
dist_exc = 1
step_meas = 1
protocol_obj = protocol.create(n_el, dist_exc=dist_exc, step_meas=step_meas, parser_meas="std")
# Dist_exc is the distance between the excitation and measurement electrodes (in number of electrodes)

keep_mask = protocol_obj.keep_ba
df_keep_mask = pd.DataFrame(keep_mask, columns=["keep"])

RADIUS_TARGET_IN_MM = 20
RADIUS_TANK_IN_MM = 190

img_size = 64

RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM

v0 = None

# METADATA
TARGET = "CYLINDER"
MATERIAL_TARGET = "PLA"
VOLTAGE_FREQUENCY = 1000
CURRENT = 0.1
CONDUCTIVITY_BG = 0.1  # in S/m     # TODO: Measure this
CONDUCTIVITY_TARGET = 1000  # in S/m


# TODO: Add some kind of metadata to the dataframes like Target used, Tank used, etc. (Like in ScioSpec Repo)


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
                                                       center_for_moving) + 4  # 4 seconds for safety and measurement
        wait_for_n_secs_with_print(move_time)
    else:
        time.sleep(2)
        center_for_moving = last_position
    """ 4. collect data """
    # get the newest file in the folder
    file_path = get_newest_file(eit_path)
    print(file_path)
    df_1 = convert_single_frequency_eit_file_to_df(file_path)
    df1 = pd.concat([df_1, df_keep_mask], axis=1)
    v1 = df1["amplitude"].to_numpy(dtype=np.float64)

    """ 5. solve EIT using Jac"""
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly_list)
    v0_solve = v0[keep_mask]
    v1_solve = v1[keep_mask]
    solve_eit_using_jac(mesh_new, mesh_obj, protocol_obj, v1_solve, v0_solve)

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
    if gcode_device is None:
        last_centers = [np.array([0, 0])]
    else:
        last_centers = [np.array([gcode_device.maximal_limits[0] / 2, gcode_device.maximal_limits[2] / 2])]
    eit_path = wait_for_start_of_measurement(
        eit_data_path)  # Wait for the start of the measurement and return the path to the data
    time.sleep(4)
    file_path = get_newest_file(eit_path)
    print(file_path)
    v0_df = convert_single_frequency_eit_file_to_df(file_path)
    # save df to pickle
    save_path_v0 = os.path.join(save_path, "v0_df.pickle")
    v0_df.to_pickle(save_path_v0)
    v0 = v0_df["amplitude"].to_numpy(dtype=np.float64)
    time.sleep(1)
    for i in range(number_of_samples):
        # add possibility to pause using cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("Paused")
            print("Press p to continue")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    print("Unpaused")
                    break
        img, v1, center_for_moving = collect_one_sample(gcode_device=gcode_device, eit_path=eit_path,
                                                        last_position=last_centers[-1])
        images.append(img)
        voltages.append(v1)
        last_centers.append(center_for_moving)
        print(f"Sample {i} collected")
        # save the images and voltages in a dataframe every 10 samples
        if i % 20 == 0:
            df = pd.DataFrame({"images": images, "voltages": voltages})
            save_path_data = os.path.join(save_path,
                                          f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
            df.to_pickle(save_path_data)
            print(f"Saved data to {save_path_data}")
            images = []
            voltages = []
    # save the images and voltages in a dataframe
    df = pd.DataFrame({"images": images, "voltages": voltages})
    save_path_data = os.path.join(save_path, f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
    df.to_pickle(save_path_data)


def calibration_procedure(gcode_device):
    """
    Moves the Target to specific positions for calibration.
    Moves to positions of electrodes 9, 25, 1, 17
    :param gcode_device:
    :return:
    """
    print("Moving to the center of the tank for calibration")
    limit_x = gcode_device.maximal_limits[0]
    limit_z = gcode_device.maximal_limits[2]
    gcode_device.move_to(x=limit_x / 2, y=0, z=limit_z / 2)
    print("Move enter so that target is in the center of the tank.")
    input("Press Enter to continue...")
    # move to top of the tank
    print("Moving to the top of the tank")
    gcode_device.move_to(x=limit_x / 2, y=0, z=limit_z - RADIUS_TARGET_IN_MM / 2)
    input("Press Enter to continue...")
    # move to the bottom of the tank#
    print("Moving to the bottom of the tank")
    gcode_device.move_to(x=limit_x / 2, y=0, z=0 + RADIUS_TARGET_IN_MM / 2)
    input("Press Enter to continue...")
    # move to the center of the tank
    gcode_device.move_to(x=limit_x / 2, y=0, z=limit_z / 2)
    # move to the right of the tank
    print("Moving to the right of the tank")
    gcode_device.move_to(x=0 + RADIUS_TARGET_IN_MM / 2, y=0, z=limit_z / 2)
    input("Press Enter to continue...")
    # move to the left of the tank
    print("Moving to the left of the tank")
    gcode_device.move_to(x=limit_x - RADIUS_TARGET_IN_MM / 2, y=0, z=limit_z / 2)
    input("Press Enter to continue...")
    # move to the center of the tank
    gcode_device.move_to(x=limit_x / 2, y=0, z=limit_z / 2)
    input("Press Enter to continue...")


def main():
    COLLECT_NEGATIVE_SAMPLES = False
    devices = list_serial_devices()
    ender = None
    for device in devices:
        if "USB-SERIAL CH340" in device.description:
            ender = GCodeDevice(device.device, movement_speed=6000,
                                home_on_init=False
                                )
            ender.maximal_limits = [RADIUS_TANK_IN_MM, RADIUS_TANK_IN_MM, RADIUS_TANK_IN_MM]
            calibration_procedure(ender)
            break
    if ender is None:
        if COLLECT_NEGATIVE_SAMPLES:
            print("No Ender 3 found")
            print("Collecting negative samples without target")
        else:
            raise Exception("No Ender 3 found")
    else:
        print("Ender 3 found")
    TEST_NAME = "Data_25_08_20mm_target"
    collect_data(gcode_device=ender, number_of_samples=4000,
                 eit_data_path="../eit_data",
                 save_path=f"C:/Users/lgudjons/PycharmProjects/EIT_reconstruction/Collected_Data/{TEST_NAME}")


if __name__ == '__main__':
    main()
