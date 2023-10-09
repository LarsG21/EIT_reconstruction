import time

import numpy as np
from matplotlib import pyplot as plt

from pyeit.eit import greit, jac, bp
from pyeit.eit.interp2d import sim2pts


def solve_and_plot_greit(v0, v1, mesh_obj, protocol_obj, path1_for_name_only, path2_for_name_only, plot=True):
    """
    Solve the EIT problem with GREIT and plot the result
    :param v0: default voltages (no anomaly)
    :param v1: voltages with anomaly
    :param mesh_obj: mesh object
    :param protocol_obj: protocol object
    :param path1_for_name_only:
    :param path2_for_name_only:
    :return:
    """

    # extract node, element, alpha
    pts = mesh_obj.node
    # pts is the list of the nodes of the mesh (with coordinates)
    tri = mesh_obj.element
    # tri is the list of the elements of the mesh (with the nodes that compose them)
    x, y = pts[:, 0], pts[:, 1]
    eit = greit.GREIT(mesh_obj, protocol_obj)
    eit.setup(p=0.50, lamb=0.01, perm=1, jac_normalized=True)
    ds = eit.solve(v1, v0, normalize=True)
    x, y, ds = eit.mask_value(ds, mask_value=0)

    # show alpha
    name1 = path1_for_name_only.split("\\")[-1]
    name2 = path2_for_name_only.split("\\")[-1]
    plt.title(f"{name1} - {name2}")
    image = np.real(ds)
    # flip the image upside down
    image = np.flipud(image)
    if plot:
        fig, axes = plt.subplots(1, 1, constrained_layout=True)
        im = axes.imshow(image, interpolation="none", cmap=plt.cm.viridis)
        axes.axis("equal")
        fig.colorbar(im, ax=axes)
        # fig.savefig('../doc/images/demo_greit.png', dpi=96)
        plt.show()
    return image


def solve_and_plot_jack(v0, v1, mesh_obj, protocol_obj, save_path=None, path1_for_name_only=None,
                        path2_for_name_only=None):
    """
    Reconstruct using JAC
    :param v0:
    :param v1:
    :param mesh_obj:
    :param protocol_obj:
    :param save_path:
    :param path1_for_name_only:
    :param path2_for_name_only:
    :return:
    """
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]
    eit = jac.JAC(mesh_obj, protocol_obj)
    eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
    start_time = time.time()
    ds = eit.solve(v1, v0, normalize=True)
    # print(f"JAC took {(time.time() - start_time)*1000} ms")
    ds_n = sim2pts(pts, tri, np.real(ds))
    # ds_n is the delta sigma interpolated on the mesh nodes
    # plot ground truth
    fig, axes = plt.subplots(1, 1, constrained_layout=True)
    if path1_for_name_only is None or path2_for_name_only is None:
        plt.title("JAC solver")
    else:
        name1 = path1_for_name_only.split("\\")[-1]
        name2 = path2_for_name_only.split("\\")[-1]
        plt.title(f"{name1} - {name2}")
    # plot EIT reconstruction
    im = axes.tripcolor(x, y, tri, ds_n, shading="flat")
    for i, e in enumerate(mesh_obj.el_pos):
        axes.annotate(str(i + 1), xy=(x[e], y[e]), color="r")
    axes.set_aspect("equal")
    fig.colorbar(im, ax=axes)
    if save_path:
        fig.savefig(save_path, dpi=96)
    plt.show()
    # plt.draw()    # For non Pycharm Plotting
    # plt.pause(0.1)
    # # close all figures to display the next one
    # print("done")


def solve_and_plot_bp(v0, v1, mesh_obj, protocol_obj, path1_for_name_only=None, path2_for_name_only=None):
    """
    Reconstruct using BP
    :param v0:
    :param v1:
    :param mesh_obj:
    :param protocol_obj:
    :param path1_for_name_only:
    :param path2_for_name_only:
    :return:
    """
    eit = bp.BP(mesh_obj, protocol_obj)
    eit.setup(weight="none")
    # the normalize for BP when dist_exc>4 should always be True
    ds = 192.0 * eit.solve(v1, v0, normalize=True)

    # extract node, element, alpha
    pts = mesh_obj.node
    tri = mesh_obj.element

    # draw
    fig, axes = plt.subplots(1, 1, constrained_layout=True)
    if path1_for_name_only is None or path2_for_name_only is None:
        plt.title("BP solver")
    else:
        name1 = path1_for_name_only.split("\\")[-1]
        name2 = path2_for_name_only.split("\\")[-1]
        plt.title(f"{name1} - {name2}")
    # original
    axes.axis("equal")
    axes.set_title(r"Input $\Delta$ Conductivities")
    # reconstructed
    im = axes.tripcolor(pts[:, 0], pts[:, 1], tri, ds)
    axes.set_title(r"Reconstituted $\Delta$ Conductivities")
    axes.axis("equal")
    fig.colorbar(im, ax=axes)
    # fig.savefig('../doc/images/demo_bp.png', dpi=96)
    plt.show()
