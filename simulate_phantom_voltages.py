# coding: utf-8
""" demo on dynamic eit using JAC method """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

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
def gernerate_random_anomaly_list(number_of_anomalies, min_radius, max_radius, min_perm, max_perm):
    anomaly_list = []
    for i in range(number_of_anomalies):
        anomaly_list.append(PyEITAnomaly_Circle(center=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)], r=np.random.uniform(min_radius, max_radius), perm=np.random.uniform(min_perm, max_perm)))
    return anomaly_list


def generate_sample_mesh_simulation(n_el):
    """ 0. build mesh """
      # nb of electrodes
    use_customize_shape = False
    if use_customize_shape:
        # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
        mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
    else:
        mesh_obj = mesh.create(n_el, h0=0.1)

    # The mesh has 704 elements

    # extract node, element, alpha
    pts = mesh_obj.node
    # pts is the list of the nodes of the mesh (with coordinates)
    tri = mesh_obj.element
    # tri is the list of the elements of the mesh (with the nodes that compose them)
    x, y = pts[:, 0], pts[:, 1]

    """ 1. problem setup """
    anomaly_list = gernerate_random_anomaly_list(number_of_anomalies=1, min_radius=0.1, max_radius=0.2, min_perm=0.001, max_perm=2)

    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly_list)
    # mesh is the mesh without anomaly
    # mesh_new is the mesh with anomaly

    """ 2. FEM simulation """
    # setup EIT scan conditions
    protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")

    # save protocol file as pickle
    pickle.dump(protocol_obj, open("protocol.pickle", "wb"))
    read_protocol = pickle.load(open("protocol.pickle", "rb"))
    print(read_protocol)

    # The protocol is the list of the electrodes to inject current and to measure voltage
    # Overall 32*32 = 1024 measurements
    # But only protocol_obj.keep_ba.sum() = 896 measurements are independent


    # calculate simulated data
    fwd = EITForward(mesh_obj, protocol_obj)
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=mesh_new.perm)
    # These are the Voltages measured at the electrodes
    # v0 is the voltage without anomaly
    # v1 is the voltage with anomaly

    return v0, v1

""" 3. JAC solver """
# Note: if the jac and the real-problem are generated using the same mesh,
# then, data normalization in solve are not needed.
# However, when you generate jac from a known mesh, but in real-problem
# (mostly) the shape and the electrode positions are not exactly the same
# as in mesh generating the jac, then data must be normalized.
eit = jac.JAC(mesh_obj, protocol_obj)
eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
ds = eit.solve(v1, v0, normalize=True)
# ds is the delta sigma, the difference between the permittivity with and without anomaly
# A list of 704 values
# pts is the list of the nodes of the mesh
# tri is the list of the elements of the mesh (triangles denote connectivity [[i, j, k]])
ds_n = sim2pts(pts, tri, np.real(ds))
# ds_n is the delta sigma interpolated on the mesh nodes

# plot ground truth
fig, axes = plt.subplots(1, 2, constrained_layout=True)
fig.set_size_inches(9, 4)

ax = axes[0]
delta_perm = mesh_new.perm - mesh_obj.perm
# delta_perm is the difference between the permittivity with and without anomaly
# ( The same what we want to reconstruct )
im = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
# Create a pseudocolor plot of an unstructured triangular grid.
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
