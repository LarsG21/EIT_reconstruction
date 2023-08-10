import cv2
import numpy as np

from Data_Generation.utils import generate_random_anomaly_list
from G_Code_Device.GCodeDevice import GCodeDevice
from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df
from ScioSpec_EIT_Device.serial_communicator import list_serial_devices

RADIUS_TARGET_IN_MM = 0.5
RADIUS_TANK_IN_MM = 1000

img_size = 64

RELATIVE_RADIUS_TARGET = RADIUS_TARGET_IN_MM / RADIUS_TANK_IN_MM


def collect_one_sample(gcode_device: GCodeDevice):
    """
    Generates a sample simulation of electrode voltages with a random anomaly.
    """
    global v0
    """ 1. problem setup """
    anomaly_list = generate_random_anomaly_list(max_number_of_anomalies=1, min_radius=RELATIVE_RADIUS_TARGET, max_radius=RELATIVE_RADIUS_TARGET, min_perm=1000,
                                                max_perm=1000, outer_circle_radius=0.75)

    if len(anomaly_list) > 1:
        raise Exception("More than one anomaly generated")

    """ 2. generate corresponding image """
    img = np.zeros([img_size, img_size])
    # set to 1 the pixels corresponding to the anomaly unsing cv2.circle
    anomaly = anomaly_list[0]
    center = np.array((anomaly.center[0], anomaly.center[1]))
    # map center from [-1, 1] to [0, 256] using numpy
    center = (center + 1) * img_size / 2
    center = center.astype(int)
    cv2.circle(img, tuple(center), int(anomaly.r * img_size / 2), 1, -1)
    # flip the image vertically because the mesh is flipped vertically
    img = np.flip(img, axis=0)

    PLOT = False
    if PLOT:
        cv2.imshow("img", cv2.resize(img, (256, 256)))

    """ 3. send gcode to the device """
    gcode_device.move_to(x=center[0], y=0, z=center[1])

    """ 4. collect data """

    # v1 = ....
    # v1  = convert_single_frequency_eit_file_to_df(path1)

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


def main():
    devices = list_serial_devices()
    ender = None
    for device in devices:
        if "USB-SERIAL CH340" in device.description:
            ender = GCodeDevice(device.device)
            break
    if ender is None:
        raise Exception("No Ender 3 found")
    else:
        print("Ender 3 found")
