import datetime
import os
import time

import cv2
import numpy as np
import pandas as pd

from Data_Generation.utils import generate_random_anomaly_list, get_newest_file, wait_for_n_secs_with_print
from G_Code_Device.GCodeDevice import GCodeDevice, list_serial_devices
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df
from pyeit.eit import protocol
from utils import wait_for_start_of_measurement

TIME_FORMAT = "%Y-%m-%d %H_%M_%S"

n_el = 32
protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")
# TODO Find out how dist_exc work

keep_mask = protocol_obj.keep_ba
print(keep_mask)
df_keep_mask = pd.DataFrame(keep_mask, columns=["keep"])

RADIUS_TARGET_IN_MM = 40
RADIUS_TANK_IN_MM = 200

img_size = 64

RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM


def collect_one_sample(gcode_device: GCodeDevice, eit_path: str, last_position: np.ndarray):
    """
    Generates a sample simulation of electrode voltages with a random anomaly.
    """
    global v0
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
        cv2.imshow("img", cv2.resize(img, (256, 256)))
        cv2.waitKey(100)

    """ 3. send gcode to the device """

    # convert center from [-1, 1] to [0, max_moving_space]
    center_for_moving = (center + 1) * gcode_device.maximal_limits[0] / 2
    center_for_moving = center_for_moving.astype(int)
    print("center_for_moving", center_for_moving)
    gcode_device.move_to(x=center_for_moving[0], y=0, z=center_for_moving[1])
    move_time = calculate_moving_time(last_position, center_for_moving) + 2  # 2 seconds for safety and measurement
    wait_for_n_secs_with_print(move_time)
    """ 4. collect data """

    # v1 = ....
    # v1  = convert_single_frequency_eit_file_to_df(path1)
    # get the newest file in the folder
    file_path = get_newest_file(eit_path)
    print(file_path)
    df_1 = convert_single_frequency_eit_file_to_df(file_path)
    df1 = pd.concat([df_1, df_keep_mask], axis=1)
    v1 = df1["amplitude"].to_numpy(dtype=np.float64)

    print(v1)

    return img, v1, center_for_moving

    # Return img and v1

    # TODO: Calculate the FEM in the future as well for comparison

    # """ 2. FEM simulation """
    # # setup EIT scan conditions
    # protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")
    #
    # # calculate simulated data
    # fwd = EITForward(mesh_obj, protocol_obj)
    # if v0 is None:
    #     v0 = fwd.solve_eit()
    # v0 = fwd.solve_eit()
    # v1 = fwd.solve_eit(perm=mesh_new.perm)
    #
    # # For control you can solve the eit and plot the result
    # if PLOT:
    #     solve_eit_using_jac(mesh_new, mesh_obj, protocol_obj, v0, v1)
    #     cv2.waitKey(1000)

    # return v0, v1, img


def collect_data(gcode_device: GCodeDevice, number_of_samples: int, eit_data_path: str):
    """
    Collects a number of samples.
    :param gcode_device:
    :param number_of_samples:
    :return:
    """
    images = []
    voltages = []
    last_centers = [np.array([0, 0])]
    eit_path = wait_for_start_of_measurement(
        eit_data_path)  # Wait for the start of the measurement and return the path to the data
    for i in range(number_of_samples):
        img, v1, center_for_moving = collect_one_sample(gcode_device=gcode_device, eit_path=eit_path,
                                                        last_position=last_centers[-1])
        images.append(img)
        voltages.append(v1)
        last_centers.append(center_for_moving)
        print(f"Sample {i} collected")
    # save the images and voltages in a dataframe
    df = pd.DataFrame({"images": images, "voltages": voltages})
    df.to_pickle(f"Data_measured{datetime.datetime.now().strftime(TIME_FORMAT)}.pkl")


def calculate_moving_time(last_position: np.ndarray, center_for_moving: np.ndarray):
    """
    Calculates the time to move from the last position to the new position.
    :param last_position: Last position in the format [x, z] in mm
    :param center_for_moving: New position in the format [x, z] in mm
    :return:
    """
    MOVING_SEED_Z = 5  # in mm per second
    MOVING_SEED_X = 50  # in mm per second

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
            break
    if ender is None:
        raise Exception("No Ender 3 found")
    else:
        print("Ender 3 found")

    collect_data(gcode_device=ender, number_of_samples=20, eit_data_path="../eit_data")


if __name__ == '__main__':
    main()
