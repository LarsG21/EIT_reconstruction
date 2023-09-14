from __future__ import division, absolute_import, print_function

import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from pyeit.eit import jac as jac
from pyeit.eit.interp2d import sim2pts

from pyeit.mesh.wrapper import PyEITAnomaly_Circle


def generate_random_anomaly_list(max_number_of_anomalies, min_radius, max_radius, min_perm, max_perm,
                                 outer_circle_radius=0.8):
    """
    Generates a list of random anomalies
    :param max_number_of_anomalies: maximum number of anomalies
    :param min_radius: minimum radius of the anomaly circle
    :param max_radius: maximum radius of the anomaly circle
    :param min_perm: minimum permittivity of the anomaly circle
    :param max_perm: maximum permittivity of the anomaly circle
    :param outer_circle_radius: radius of the outer circle
    :return: list of anomalies
    """
    anomaly_list = []
    if max_number_of_anomalies != 1:
        number_of_anomalies = np.random.randint(1, max_number_of_anomalies)
    else:
        number_of_anomalies = 1
    for i in range(number_of_anomalies):
        center, r, perm = generate_random_anomaly_parameters(min_radius, max_radius, min_perm, max_perm,
                                                             outer_circle_radius)
        anomaly_list.append(PyEITAnomaly_Circle(center=center, r=r, perm=perm))
    return anomaly_list


def generate_random_anomaly_parameters(min_radius, max_radius, min_perm, max_perm, outer_circle_radius=0.8):
    """
    Generates random parameters for an anomaly circle inside a Tank
    :param min_radius: minimum radius of the anomaly circle
    :param max_radius: maximum radius of the anomaly circle
    :param min_perm: minimum permittivity of the anomaly circle
    :param max_perm: maximum permittivity of the anomaly circle
    :param outer_circle_radius: radius of the outer circle
    :return: center, radius, permittivity of the anomaly circle
    """
    if max_radius + outer_circle_radius > 1:
        raise ValueError("max_radius + outer_circle_radius > 1 --> anomaly can be outside the circle")
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, outer_circle_radius)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    center = np.array([x, y])
    r = np.random.uniform(min_radius, max_radius)
    perm = np.random.uniform(min_perm, max_perm)
    return center, r, perm


def wait_for_start_of_measurement(path):
    """
    Waits for the first file to be written. Searches for the setup folder and returns the path to it.
    :param eit_path:
    :param path:
    :return:
    """
    eit_path = ""
    while len(os.listdir(path)) == 0:
        print("Waiting for files to be written")
        time.sleep(0.5)
    print("EIT capture started")
    time.sleep(1)
    for file_or_folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_or_folder)):
            os.chdir(os.path.join(path, file_or_folder))
            print(os.getcwd())
            os.chdir((os.path.join(os.getcwd(), "setup")))
            eit_path = os.getcwd()
            print(eit_path)
            break
    return eit_path


def get_newest_file(path):
    """
    Returns the newest file in a directory and deletes all other files
    :param path:
    :return:
    """
    cwd = os.getcwd()
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    # select only files that g´have ending .eit
    paths = [path for path in paths if path.endswith(".eit")]
    newest = max(paths, key=os.path.getctime)
    # delete all files except the newest
    for file in files:
        if file != os.path.basename(newest):
            os.remove(os.path.join(path, file))
    os.chdir(cwd)
    return newest


def wait_for_n_secs_with_print(n_secs):
    """
    Waits for n seconds and prints the remaining time
    :param n_secs:
    :return:
    """
    for i in range(n_secs):
        print("Waiting for {} seconds".format(n_secs - i))
        time.sleep(1)
    print("Waiting finished")
    return True


def look_at_dataset(img_array, v1_array, v0=None):
    """
    Shows the images and voltage difference of a dataset
    :param img_array: The images with the anomaly
    :param v1_array: The voltages measured with the anomaly
    :param v0: The voltages measured without the anomaly
    :return:
    """
    print(img_array.shape)
    print(v1_array.shape)
    average_image = np.mean(img_array, axis=0) * 10
    # clip between 0 and 255
    average_image = np.clip(average_image, 0, 255)

    cv2.imshow('average', cv2.resize(average_image, (256, 256)))
    for i, img in enumerate(img_array):
        if v0 is not None:
            voltage_differece = (v1_array[i] - v0) / v0
        else:
            voltage_differece = v1_array[i]
        # show voltage difference and image in one plot
        plt.subplot(1, 2, 1)
        plt.imshow(img * 10)
        plt.title('image')
        plt.subplot(1, 2, 2)
        plt.plot(voltage_differece)
        plt.title('voltage difference')
        plt.show()
        cv2.imshow('img', cv2.resize(img * 10, (256, 256)))

        cv2.waitKey(100)
    # cv2.waitKey(0)


def solve_eit_using_jac(mesh_new, mesh_obj, protocol_obj, v0, v1):
    """
    Solves the EIT problem using the jac method.
    :param mesh_new:
    :param mesh_obj:
    :param protocol_obj:
    :param v0: The voltages measured without the anomaly (already selected relevant electrodes)
    :param v1: The voltages measured with the anomaly (already selected relevant electrodes)
    :return:
    """

    # Note: if the jac and the real-problem are generated using the same mesh,
    # then, data normalization in solve are not needed.
    # However, when you generate jac from a known mesh, but in real-problem
    # (mostly) the shape and the electrode positions are not exactly the same
    # as in mesh generating the jac, then data must be normalized.
    pts = mesh_obj.node
    # pts is the list of the nodes of the mesh (with coordinates)
    tri = mesh_obj.element
    # tri is the list of the elements of the mesh (with the nodes that compose them)
    x, y = pts[:, 0], pts[:, 1]
    eit = jac.JAC(mesh_obj, protocol_obj)
    eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    ds_n = sim2pts(pts, tri, np.real(ds))
    # plot ground truth
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    fig.set_size_inches(9, 4)
    ax = axes[0]
    delta_perm = mesh_new.perm - mesh_obj.perm
    im = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
    ax.set_aspect("equal")
    # plot EIT reconstruction
    ax = axes[1]
    im = ax.tripcolor(x, y, tri, ds_n, shading="flat")
    for i, e in enumerate(mesh_obj.el_pos):
        ax.annotate(str(i + 1), xy=(x[e], y[e]), color="r")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=axes.ravel().tolist())
    # plt.savefig('../doc/images/demo_jac.png', dpi=96)
    plt.show()


def calibration_procedure(gcode_device, RADIUS_TARGET_IN_MM):
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


if __name__ == '__main__':
    for i in range(100):
        print(generate_random_anomaly_parameters(0.1, 0.1, 0.1, 0.1, 0.8))
