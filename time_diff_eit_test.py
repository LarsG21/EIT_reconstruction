import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from ScioSpec_EIT_Device.data_reader import convert_single_frequency_eit_file_to_df, convert_multi_frequency_eit_to_df
from eidnburgh_cnn_test import CNNModel
from model_plot_utils import plot_single_reconstruction
from pyeit import mesh
from pyeit.eit import protocol, jac, greit, bp
from pyeit.eit.interp2d import sim2pts
from pyeit.mesh.shape import thorax
import os

""" 0. build mesh """
n_el = 32  # nb of electrodes
use_customize_shape = False
if use_customize_shape:
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
    mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.1)

""" 1. problem setup """
anomaly = []
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
delta_perm = np.real(mesh_new.perm - mesh_obj.perm)

# extract node, element, alpha
pts = mesh_obj.node
# pts is the list of the nodes of the mesh (with coordinates)
tri = mesh_obj.element
# tri is the list of the elements of the mesh (with the nodes that compose them)
x, y = pts[:, 0], pts[:, 1]

""" 2. FEM simulation """
# setup EIT scan conditions
protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")

keep_mask = protocol_obj.keep_ba
print(keep_mask)
df_keep_mask = pd.DataFrame(keep_mask, columns=["keep"])

default_frame = None


def solve_and_plot_cnn(model, v1):
    plot_single_reconstruction(model, v1)


def plot_time_diff_eit_image(path1, path2, frequency=1000):
    global default_frame
    # df1 = convert_multi_frequency_eit_to_df(path1)
    # df2 = convert_multi_frequency_eit_to_df(path2)
    # df1 = df1[df1["frequency"] == frequency]
    # df2 = df2[df2["frequency"] == frequency]
    # df1 = df1.reset_index(drop=True)
    # df2 = df2.reset_index(drop=True)
    df1_old = convert_single_frequency_eit_file_to_df(path1)
    if default_frame is None:
        df2_old = convert_single_frequency_eit_file_to_df(path2)
    else:
        df2_old = default_frame

    df1 = pd.concat([df1_old, df_keep_mask], axis=1)
    df2 = pd.concat([df2_old, df_keep_mask], axis=1)
    df1 = df1[df1["keep"] == True].drop("keep", axis=1)
    df2 = df2[df2["keep"] == True].drop("keep", axis=1)
    # print some statistics about the data min max mean std
    # print(df1.describe())

    v0 = df1["amplitude"].to_numpy(dtype=np.float64)
    v1 = df2["amplitude"].to_numpy(dtype=np.float64)

    solve_and_plot_jack(path1, path2, v0, v1)
    # solve_and_plot_greit(path1, path2, v0, v1)
    # solve_and_plot_bp(path1, path2, v0, v1)
    # solve_and_plot_stack(path1, path2, v0, v1)
    # solve_and_plot_cnn(model=model, v1=v1-v0)


def plot_frequencies_diff_eit_image(path, f1,f2):
    """
    Plot the difference between two frequencies
    :param path1:
    :param f1:
    :param f2:
    :return:
    """
    if not type(f1) == int or not type(f2) == int:
        raise Exception("f1 and f2 must be integers")
    df = convert_multi_frequency_eit_to_df(path)
    df1 = df[df["frequency"] == f1]
    df2 = df[df["frequency"] == f2]
    df1 = df.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    df1 = pd.concat([df1, df_keep_mask], axis=1)
    df2 = pd.concat([df2, df_keep_mask], axis=1)
    df1 = df1[df1["keep"] == True].drop("keep", axis=1)
    df2 = df2[df2["keep"] == True].drop("keep", axis=1)
    v0 = df1["amplitude"].to_numpy(dtype=np.float64)
    v1 = df2["amplitude"].to_numpy(dtype=np.float64)

    solve_and_plot_jack(f"{f1}", f"{f2}", v0, v1)

def solve_and_plot_greit(path1, path2, v0, v1):
    """ 3. Construct using GREIT """
    eit = greit.GREIT(mesh_obj, protocol_obj)
    eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    x, y, ds = eit.mask_value(ds, mask_value=np.NAN)

    # show alpha
    fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))
    name1 = path1.split("\\")[-1]
    name2 = path2.split("\\")[-1]
    plt.title(f"{name1} - {name2}")

    ax = axes[0]
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")
    ax.axis("equal")
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_title(r"$\Delta$ Conductivity")
    # fig.set_size_inches(6, 4)

    # plot
    ax = axes[1]
    im = ax.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)
    ax.axis("equal")

    fig.colorbar(im, ax=axes.ravel().tolist())
    # fig.savefig('../doc/images/demo_greit.png', dpi=96)
    plt.show()


def solve_and_plot_jack(path1, path2, v0, v1):
    """ 3. JAC solver """
    # Note: if the jac and the real-problem are generated using the same mesh,
    # then, data normalization in solve are not needed.
    # However, when you generate jac from a known mesh, but in real-problem
    # (mostly) the shape and the electrode positions are not exactly the same
    # as in mesh generating the jac, then data must be normalized.
    eit = jac.JAC(mesh_obj, protocol_obj)
    eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
    start_time = time.time()
    ds = eit.solve(v1, v0, normalize=True)
    print(f"JAC took {(time.time() - start_time)*1000} ms")
    # ds is the delta sigma, the difference between the permittivity with and without anomaly
    # A list of 704 values
    # pts is the list of the nodes of the mesh
    # tri is the list of the elements of the mesh (triangles denote connectivity [[i, j, k]])
    ds_n = sim2pts(pts, tri, np.real(ds))
    # ds_n is the delta sigma interpolated on the mesh nodes
    # plot ground truth
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    name1 = path1.split("\\")[-1]
    name2 = path2.split("\\")[-1]
    plt.title(f"{name1} - {name2}")
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
    plt.show()
    # plt.draw()    # For non Pycharm Plotting
    # plt.pause(0.1)
    # # close all figures to display the next one
    # print("done")


def solve_and_plot_bp(path1, path2, v0, v1):
    """ 3. naive inverse solver using back-projection """
    eit = bp.BP(mesh_obj, protocol_obj)
    eit.setup(weight="none")
    # the normalize for BP when dist_exc>4 should always be True
    ds = 192.0 * eit.solve(v1, v0, normalize=True)

    # extract node, element, alpha
    pts = mesh_obj.node
    tri = mesh_obj.element

    # draw
    fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))
    name1 = path1.split("\\")[-1]
    name2 = path2.split("\\")[-1]
    plt.title(f"{name1} - {name2}")
    # original
    ax = axes[0]
    ax.axis("equal")
    ax.set_title(r"Input $\Delta$ Conductivities")
    delta_perm = np.real(mesh_new.perm - mesh_obj.perm)
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")
    # reconstructed
    ax1 = axes[1]
    im = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, ds)
    ax1.set_title(r"Reconstituted $\Delta$ Conductivities")
    ax1.axis("equal")
    fig.colorbar(im, ax=axes.ravel().tolist())
    # fig.savefig('../doc/images/demo_bp.png', dpi=96)
    plt.show()

def solve_and_plot_stack(path1, path2, v0, v1):
    """ 3. solving using dynamic EIT """
    # number of stimulation lines/patterns
    eit = jac.JAC(mesh_obj, protocol_obj)
    eit.setup(p=0.40, lamb=1e-3, method="kotre", jac_normalized=False)
    ds = eit.solve(v1, v0, normalize=False)

    # extract node, element, alpha
    pts = mesh_obj.node
    tri = mesh_obj.element
    delta_perm = mesh_new.perm - mesh_obj.perm

    # # show alpha
    # fig, ax = plt.subplots(figsize=(6, 4))
    # im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(delta_perm), shading="flat")
    # fig.colorbar(im)
    # ax.set_aspect("equal")
    # ax.set_title(r"$\Delta$ Permittivity")

    """ 4. plot """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.tripcolor(
        pts[:, 0],
        pts[:, 1],
        tri,
        np.real(ds),
        shading="flat",
        alpha=0.90,
        cmap=plt.cm.viridis,
    )
    fig.colorbar(im)
    ax.set_aspect("equal")
    ax.set_title(r"$\Delta$ Permittivity Reconstructed")
    # plt.savefig('quasi-demo-eit.pdf')
    plt.show()




def plot_eit_images_in_folder(path):
    old_path = os.getcwd()
    eit_path = ""
    for file_or_folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_or_folder)):
            os.chdir(os.path.join(path, file_or_folder))
            print(os.getcwd())
            os.chdir((os.path.join(os.getcwd(), "setup")))
            eit_path = os.getcwd()
            print(eit_path)
            break

    default_frame = None
    for current_frame in os.listdir(os.getcwd()):
        if current_frame.endswith(".eit"):
            if default_frame is None:
                default_frame = current_frame
            else:
                print(default_frame, current_frame)
                plot_time_diff_eit_image(os.path.join(eit_path, current_frame), os.path.join(eit_path, default_frame))
                # last_frame = current_frame
    # reset path
    os.chdir(old_path)

def plot_eit_video(path):
    """
    Plots the eit video from the given path.
    There are new files in the folder every few seconds.
    Do the same as above continuously.
    :param path:
    :return:
    """
    eit_path = ""
    seen_files = []
    if len(os.listdir(path)) == 0:
        print("No files in folder")
        return
    for file_or_folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_or_folder)):
            os.chdir(os.path.join(path, file_or_folder))
            print(os.getcwd())
            os.chdir((os.path.join(os.getcwd(), "setup")))
            eit_path = os.getcwd()
            print(eit_path)
            break
        else:
            print("No folder found")
            return
    default_frame = None
    while True:
        for current_frame in os.listdir(os.getcwd()):
            if current_frame.endswith(".eit") and current_frame not in seen_files:
                if default_frame is None:
                    default_frame = current_frame
                else:
                    time.sleep(0.01) # wait for file to be written
                    # print(default_frame, current_frame)
                    plot_time_diff_eit_image(os.path.join(eit_path, current_frame),
                                             os.path.join(eit_path, default_frame))
                    seen_files.append(current_frame)
        # time.sleep(0.1)

path = "eit_data"

# path_t1 = "setup_00002.eit"
# path_t2 = "setup_00006.eit"
# plot_eit_image(path_t1, path_t2)

# start = time.time()
# plot_eit_images_in_folder(path)
# end = time.time()
# print("Time taken: ", end - start)
VOLTAGE_VECTOR_LENGTH = 896
OUT_SIZE = 64
print("Loading the model")
model = CNNModel(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE**2)
# model.load_state_dict(torch.load(
#     "Edinburgh mfEIT Dataset/models_new_loss_methode/2/model_2023-07-27_16-38-33_60_150.pth"))
model.load_state_dict(torch.load(
    "Own_Simulation_Dataset/Models/Test_noise_03_regularization_1e-5/model_2023-08-02_19-13-16_epoche_143_of_150_best_model.pth"))
model.eval()
plot_eit_video(path)
