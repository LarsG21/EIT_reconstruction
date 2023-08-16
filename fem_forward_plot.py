# coding: utf-8
""" demo on forward 2D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from plot_utils import plot_results_fem_forward
from pyeit.mesh.shape import thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle

""" 0. build mesh """
n_el = 16  # nb of electrodes
use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.05)

mesh_obj.print_stats()

# change permittivity
anomaly = PyEITAnomaly_Circle(center=[0.4, 0.5], r=0.2, perm=100.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

""" 1. FEM forward simulations """
# setup EIT scan conditions
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

for line in protocol_obj.ex_mat:
    plot_results_fem_forward(mesh=mesh_new, line=line)
