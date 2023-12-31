# coding: utf-8
""" demo on dynamic eit using JAC method """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyeit.mesh as mesh
from Data_Generation.utils import generate_random_anomaly_list, solve_eit_using_jac, look_at_dataset
from plot_utils import plot_results_fem_forward
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol

n_el = 32
img_size = 64

v0 = None


def generate_sample_mesh_simulation(mesh_obj, n_el=32):
    """
    Generates a sample simulation of electrode voltages with a random anomaly.
    """
    global v0
    """ 1. problem setup """
    anomaly_list = generate_random_anomaly_list(max_number_of_anomalies=0, min_radius=0.21, max_radius=0.21,
                                                min_perm=1000,
                                                max_perm=1000, outer_circle_radius=0.75)
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly_list)

    img = np.zeros([img_size, img_size])
    # set to 1 the pixels corresponding to the anomaly unsing cv2.circle
    for anomaly in anomaly_list:
        center = np.array((anomaly.center[0], anomaly.center[1]))
        # map center from [-1, 1] to [0, 256] using numpy
        center = (center + 1) * img_size / 2
        center = center.astype(int)
        cv2.circle(img, tuple(center), int(anomaly.r * img_size / 2), 1, -1)
    # flip the image vertically because the mesh is flipped vertically
    img = np.flip(img, axis=0)
    THESIS_SAMPLE = False
    if THESIS_SAMPLE:
        # plot the image
        plt.imshow(img)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Image with anomaly")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.show()

    PLOT = False
    if PLOT:
        cv2.imshow("img", cv2.resize(img, (256, 256)))

    """ 2. FEM simulation """
    # setup EIT scan conditions
    protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")

    # calculate simulated data
    fwd = EITForward(mesh_obj, protocol_obj)
    if v0 is None:
        v0 = fwd.solve_eit()
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    # For control you can solve the eit and plot the result
    if PLOT:
        solve_eit_using_jac(mesh_new, mesh_obj, protocol_obj, v0, v1)
        cv2.waitKey(1000)
        plot_results_fem_forward(mesh=mesh_new, line=protocol_obj.ex_mat[0])

    return v0, v1, img


def generate_dataset(dataset_name, samples, noise_amplitude=0.01):
    """
    Generates a dataset of samples with the same mesh.
    :param dataset_name:
    :param samples:
    :return:
    """
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    else:
        raise Exception("The dataset already exists. Please delete it before generating a new one.")
    DATA_COLLECTION_RUN = 0
    mesh_obj = mesh.create(n_el, h0=0.1)  # TODO: Experiment with higher mesh resolutions
    # The mesh has 704 elements
    # extract node, element, alpha
    pts = mesh_obj.node
    # pts is the list of the nodes of the mesh (with coordinates)
    tri = mesh_obj.element
    # tri is the list of the elements of the mesh (with the nodes that compose them)
    x, y = pts[:, 0], pts[:, 1]

    img_array = []
    v0_array = []
    v1_array = []
    # Simulate 1 sample to get the v0
    v0, v1, img = generate_sample_mesh_simulation(mesh_obj=mesh_obj, n_el=32)
    # add random noise to v1 and v0
    v_diff = v1 - v0
    # scaled_noise_amplitude = noise_amplitude * np.max(v_diff)
    scaled_noise_amplitude = noise_amplitude
    print(f"Scaled noise amplitude: {scaled_noise_amplitude}")
    v0 = v0 + np.random.normal(0, scaled_noise_amplitude, v0.shape)
    v1 = v1 + np.random.normal(0, scaled_noise_amplitude, v1.shape)
    np.save(f"{dataset_name}/v0.npy", v0)
    #
    # Simulate the rest of the samples
    # for i in range(10):
    start = time.time()
    DATA_COLLECTION_RUN += 1
    for i in range(samples):
        print(i)
        v0, v1, img = generate_sample_mesh_simulation(mesh_obj=mesh_obj, n_el=32)
        # add random noise to v1 and v0
        v0 = v0 + np.random.normal(0, scaled_noise_amplitude, v0.shape)
        v1 = v1 + np.random.normal(0, scaled_noise_amplitude, v1.shape)
        img_array.append(img)
        v0_array.append(v0)
        v1_array.append(v1)
    print()
    end = time.time()
    print(f"Time elapsed for {samples} samples: {end - start}")
    print("Average time per sample: ", (end - start) / samples)
    img_array_np = np.array(img_array)
    np.save(f"{dataset_name}/img_array_{DATA_COLLECTION_RUN}.npy", img_array_np)
    v1_array_np = np.array(v1_array)
    np.save(f"{dataset_name}/v1_array_{DATA_COLLECTION_RUN}.npy", v1_array_np)
    v0_array_np = np.array(v0_array)
    np.save(f"{dataset_name}/v0_array_{DATA_COLLECTION_RUN}.npy", v0_array_np)
    print("OK")
    df = pd.DataFrame()
    df["images"] = img_array
    df["voltages"] = v1_array
    df.to_pickle(f"{dataset_name}/dataset_{DATA_COLLECTION_RUN}_negative.pkl")


if __name__ == '__main__':
    DATASET_NAME = "Simulated_negative_016noise"
    SAMPLES = 300

    generate_dataset(DATASET_NAME, SAMPLES, noise_amplitude=0.0008328659934874609)

    img_array = np.load(f"{DATASET_NAME}/img_array_1.npy")
    v1_array = np.load(f"{DATASET_NAME}/v1_array_1.npy")
    v0 = np.load(f"{DATASET_NAME}/v0.npy")
    look_at_dataset(img_array, v1_array, v0)
