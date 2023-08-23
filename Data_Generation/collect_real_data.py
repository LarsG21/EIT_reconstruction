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
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
# Dist_exc is the distance between the excitation and measurement electrodes (in number of electrodes)

keep_mask = protocol_obj.keep_ba
df_keep_mask = pd.DataFrame(keep_mask, columns=["keep"])

RADIUS_TARGET_IN_MM = 40
RADIUS_TANK_IN_MM = 200

img_size = 64

RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM

v0 = None


# TODO: Add some kind of metadata to the dataframes like Target used, Tank used, etc. (Like in ScioSpec Repo)


def collect_one_sample(gcode_device: GCodeDevice, eit_path: str, last_position: np.ndarray):
    """
    Generates a sample simulation of electrode voltages with a random anomaly.
    """
    """ 1. problem setup """
    anomaly_list = generate_random_anomaly_list(max_number_of_anomalies=1, min_radius=RELATIVE_RADIUS_TARGET,
                                                max_radius=RELATIVE_RADIUS_TARGET, min_perm=1000,
                                                max_perm=1000, outer_circle_radius=0.75)

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
        cv2.imshow("img", cv2.resize(img_show, (256, 256)))
        cv2.waitKey(100)

    """ 3. send gcode to the device """
    # convert center from [-1, 1] to [0, max_moving_space]
    center_for_moving = (center + 1) * gcode_device.maximal_limits[0] / 2
    # invert x axis
    center_for_moving[0] = gcode_device.maximal_limits[0] - center_for_moving[0]
    center_for_moving = center_for_moving.astype(int)
    print("center_for_moving", center_for_moving)
    gcode_device.move_to(x=center_for_moving[0], y=0, z=center_for_moving[1])
    move_time = calculate_moving_time(last_position, center_for_moving) + 4  # 4 seconds for safety and measurement
    wait_for_n_secs_with_print(move_time)
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
        os.makedirs(save_path)  # TODO: First move to the center of the tank for calibration
    # create txt file with the metadata
    metadata = {"number_of_samples": number_of_samples, "img_size": img_size, "n_el": n_el,
                "radius_target_in_mm": RADIUS_TARGET_IN_MM, "radius_tank_in_mm": RADIUS_TANK_IN_MM,
                "dist_exc": protocol_obj.dist_exc, "step_meas": protocol_obj.step_meas,
                }
    with open(os.path.join(save_path, "metadata.txt"), 'w') as file:
        file.write(json.dumps(metadata))
    print("Moving to the center of the tank for calibration")
    gcode_device.move_to(x=gcode_device.maximal_limits[0] / 2, y=0, z=gcode_device.maximal_limits[2] / 2)
    print("Move enter so that target is in the center of the tank and press enter")
    wait_for_n_secs_with_print(30)
    images = []
    voltages = []
    last_centers = [np.array([gcode_device.maximal_limits[0] / 2, gcode_device.maximal_limits[2] / 2])]
    eit_path = wait_for_start_of_measurement(
        eit_data_path)  # Wait for the start of the measurement and return the path to the data
    time.sleep(4)
    file_path = get_newest_file(eit_path)
    print(file_path)
    v0_df = convert_single_frequency_eit_file_to_df(file_path)
    # save df to pickle
    save_path_v0 = os.path.join(save_path, "v0_df")
    v0_df.to_pickle(save_path_v0)  # TODO: Change save path
    v0 = v0_df["amplitude"].to_numpy(dtype=np.float64)
    time.sleep(1)
    for i in range(number_of_samples):
        img, v1, center_for_moving = collect_one_sample(gcode_device=gcode_device, eit_path=eit_path,
                                                        last_position=last_centers[-1])
        images.append(img)
        voltages.append(v1)
        last_centers.append(center_for_moving)
        print(f"Sample {i} collected")
    # save the images and voltages in a dataframe
    df = pd.DataFrame({"images": images, "voltages": voltages})
    save_path_data = os.path.join(save_path, f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")
    df.to_pickle(save_path_data)


def calculate_moving_time(last_position: np.ndarray, center_for_moving: np.ndarray):
    """
    Calculates the time to move from the last position to the new position.
    :param last_position: Last position in the format [x, z] in mm
    :param center_for_moving: New position in the format [x, z] in mm
    :return:
    """
    MOVING_SEED_Z = 5  # in mm per second
    MOVING_SEED_X = 60  # in mm per second

    time_to_move = int(np.linalg.norm(last_position[0] - center_for_moving[0]) / MOVING_SEED_X +
                       np.linalg.norm(last_position[1] - center_for_moving[1]) / MOVING_SEED_Z)
    print("time_to_move", time_to_move)

    return time_to_move


def main():
    devices = list_serial_devices()
    ender = None
    for device in devices:
        if "USB-SERIAL CH340" in device.description:
            ender = GCodeDevice(device.device, movement_speed=6000,
                                home_on_init=True)
            ender.maximal_limits = [200, 200, 200]
            break
    if ender is None:
        raise Exception("No Ender 3 found")
    else:
        print("Ender 3 found")
    TEST_NAME = "Test_1000_Samples"
    collect_data(gcode_device=ender, number_of_samples=1000,
                 eit_data_path="../eit_data",
                 save_path=f"C:/Users/lgudjons/PycharmProjects/EIT_reconstruction/Collected_Data/{TEST_NAME}")


if __name__ == '__main__':
    main()
