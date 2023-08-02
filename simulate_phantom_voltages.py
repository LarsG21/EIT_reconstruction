# coding: utf-8
""" demo on dynamic eit using JAC method """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.eit.interp2d import sim2pts
from pyeit.mesh.shape import thorax
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
import pickle


n_el = 32

img_size = 64

def generate_random_anomaly_parameters(min_radius, max_radius, min_perm, max_perm, outer_circle_radius=0.8):
    if max_radius + outer_circle_radius > 1:
        raise ValueError("max_radius + outer_circle_radius > 1 --> anomaly can be outside the circle")
    center = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    # allow only center position that are inside a circle of radius 1
    while np.linalg.norm(center) > outer_circle_radius:
        center = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    r = np.random.uniform(min_radius, max_radius)
    perm = np.random.uniform(min_perm, max_perm)
    return center, r, perm

def generate_random_anomaly_list(number_of_anomalies, min_radius, max_radius, min_perm, max_perm, outer_circle_radius=0.8):
    anomaly_list = []
    for i in range(number_of_anomalies):
        center, r, perm = generate_random_anomaly_parameters(min_radius, max_radius, min_perm, max_perm, outer_circle_radius)
        anomaly_list.append(PyEITAnomaly_Circle(center=center, r=r, perm=perm))
    return anomaly_list


v0 = None

def generate_sample_mesh_simulation(mesh_obj, n_el=32):
    global v0
    """ 1. problem setup """
    anomaly_list = generate_random_anomaly_list(number_of_anomalies=1, min_radius=0.1, max_radius=0.2, min_perm=1000,
                                                max_perm=1000, outer_circle_radius=0.8)
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

    return v0, v1, img


def solve_eit_using_jac(mesh_new, mesh_obj, protocol_obj, v0, v1):
    """ 3. JAC solver """
    # Note: if the jac and the real-problem are generated using the same mesh,
    # then, data normalization in solve are not needed.
    # However, when you generate jac from a known mesh, but in real-problem
    # (mostly) the shape and the electrode positions are not exactly the same
    # as in mesh generating the jac, then data must be normalized.
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


if __name__ == '__main__':
    """ 0. build mesh """
    DATA_COLLECTION_RUN = 2
    SAMPLES = 5000
    mesh_obj = mesh.create(n_el, h0=0.1)
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
    start = time.time()
    # Simulate 1 sample to get the v0
    # v0, v1, img = generate_sample_mesh_simulation(mesh_obj=mesh_obj, n_el=32)
    # np.save("Own_Simulation_Dataset/v0.npy", v0)

    # Simulate the rest of the samples
    for i in range(SAMPLES):
        print(i)
        v0, v1, img = generate_sample_mesh_simulation(mesh_obj=mesh_obj, n_el=32)
        img_array.append(img)
        v0_array.append(v0)
        v1_array.append(v1)


    end = time.time()
    print(f"Time elapsed for {SAMPLES} samples: {end - start}")
    print("Average time per sample: ", (end - start) / SAMPLES)

    img_array = np.array(img_array)
    np.save(f"img_array_{DATA_COLLECTION_RUN}.npy", img_array)
    v1_array = np.array(v1_array)
    np.save(f"v1_array_{DATA_COLLECTION_RUN}.npy", v1_array)
    v0_array = np.array(v0_array)
    print("OK")

    average_image = np.mean(img_array, axis=0)
    cv2.imshow('average', cv2.resize(average_image, (256, 256)))
    cv2.waitKey(0)

    # load the data
    # img_array1 = np.load("Own_Simulation_Dataset/img_array_1.npy")
    # v1_array1 = np.load("Own_Simulation_Dataset/v1_array_1.npy")
    # img_array2 = np.load("Own_Simulation_Dataset/img_array_2.npy")
    # v1_array2 = np.load("Own_Simulation_Dataset/v1_array_2.npy")
    #
    # img_array = np.concatenate((img_array1, img_array2), axis=0)
    # v1_array = np.concatenate((v1_array1, v1_array2), axis=0)

    # np.save("Own_Simulation_Dataset/img_array.npy", img_array)
    # np.save("Own_Simulation_Dataset/v1_array.npy", v1_array)
    #
    # print(img_array.shape)
    # print(v1_array.shape)
    # print("OK")





    # # time generate_random_anomaly_parameters with timeit
    # import timeit
    # print(timeit.timeit("generate_random_anomaly_parameters(0.1, 0.2, 1, 1)", setup="from __main__ import generate_random_anomaly_parameters", number=100000))



