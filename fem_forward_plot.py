# coding: utf-8
""" demo on forward 2D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import absolute_import, division, print_function

import tikzplotlib

import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from plot_utils import plot_results_fem_forward
from pyeit.mesh.shape import thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

""" 0. build mesh """
n_el = 16  # nb of electrodes
use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.07)

mesh_obj.print_stats()

# change permittivity
anomaly = PyEITAnomaly_Circle(center=[0.4, 0.5], r=0.3, perm=100.0)
# anomaly = []
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

""" 1. FEM forward simulations """
# setup EIT scan conditions
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")


def plot_potentail_and_e_field_for_all_injections(mesh, protocol_obj):
    """
    Plot the equi-potential lines and the E-field lines for all injections in the protocol object.
    :param mesh:
    :param protocol_obj:
    :return:
    """
    for line in protocol_obj.ex_mat:
        # plot_results_fem_forward(mesh=mesh_new, line=line)

        equi_potential_image, e_field_image, fig_e_field, fig_equipotential = plot_results_fem_forward(mesh=mesh_new, line=line)

        plt.figure(figsize=(12, 6))

        # Display equi-potential image
        plt.subplot(1, 2, 1)
        plt.imshow(equi_potential_image)
        plt.title("Equi-potential Image")
        plt.axis('off')

        # Display E-field image
        plt.subplot(1, 2, 2)
        plt.imshow(e_field_image)
        plt.title("E-Field Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def plot_potential_lines_for_all_injections(mesh, protocol_obj):
    """
    Plot the equi-potential lines for all injections in the protocol object.
    :param mesh:
    :param protocol_obj:
    :return:
    """
    equi_potential_images = []
    plt.rcParams.update({'font.size': 12})
    # set font to charter
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Charter'] + plt.rcParams['font.serif']

    for line in protocol_obj.ex_mat:
        equi_potential_image, e_field_image, fig_e_field, fig_equipotential = plot_results_fem_forward(mesh=mesh, line=line)
        equi_potential_images.append(equi_potential_image)
    # take only every 4th image
    equi_potential_images = equi_potential_images[::4]
    # Calculate the number of rows and columns for the subplot grid
    num_images = len(equi_potential_images)
    num_cols = 4  # Number of columns in the subplot grid
    num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division

    # Create a new figure and subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Iterate through the images and axes to plot them
    for i, (equi_potential_image, ax) in enumerate(zip(equi_potential_images, axes.flatten())):
        ax.imshow(equi_potential_image)
        # ax.set_title(f"Equi-potential Image {i+1}")
        ax.axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    # save as tikz file
    # tikzplotlib.save("equi-potential-lines.tex")
    # save as pdf
    plt.savefig("equi-potential-lines.pdf")
    # save the plot
    plt.savefig("equi-potential-lines.png")
    plt.show()


if __name__ == '__main__':
    # plot_potentail_and_e_field_for_all_injections(mesh=mesh_new, protocol_obj=protocol_obj)
    plot_potential_lines_for_all_injections(mesh=mesh_new, protocol_obj=protocol_obj)
