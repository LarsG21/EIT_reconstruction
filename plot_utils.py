from __future__ import division, absolute_import, print_function

import numpy as np
from matplotlib import pyplot as plt

from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import pdegrad, sim2pts


def plot_results_fem_forward(mesh, line):
    """
    Plot results of FEM forward simulation. Plots the equi-potential lines and the E-Field lines.
    :param mesh: the mesh object used for the FEM forward simulation
    :param line: Input and output electrodes as a 2x1 nparray
    :return:
    """
    print(f"plot_results_fem_forward between {line[0]} and {line[1]}")
    # extract node, element, alpha
    pts = mesh.node
    tri = mesh.element
    x, y = pts[:, 0], pts[:, 1]
    perm = mesh.perm
    el_pos = mesh.el_pos

    ex_line = line.ravel()
    # calculate simulated data using FEM
    fwd = Forward(mesh)
    f = fwd.solve(ex_line)
    f = np.real(f)
    """ 2. plot """
    fig, ax1 = plt.subplots(figsize=(9, 6))
    # draw equi-potential lines
    vf = np.linspace(min(f), max(f), 64)
    # vf = np.sort(f[el_pos])
    # Draw contour lines on an unstructured triangular grid.
    ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)
    # draw mesh structure
    # Create a pseudocolor plot of an unstructured triangular grid
    ax1.tripcolor(
        x,
        y,
        tri,
        np.real(perm),
        edgecolors="k",
        shading="flat",
        alpha=0.5,
        cmap=plt.cm.Greys,
    )
    # draw electrodes
    ax1.plot(x[el_pos], y[el_pos], "ro")
    for i, e in enumerate(el_pos):
        ax1.text(x[e], y[e], str(i + 1), size=12)
    ax1.set_title("equi-potential lines")
    # clean up
    ax1.set_aspect("equal")
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)
    # fig.savefig('demo_bp.png', dpi=96)
    plt.show()
    ux, uy = pdegrad(pts, tri, f)
    uf = ux ** 2 + uy ** 2
    uf_pts = sim2pts(pts, tri, uf)
    uf_logpwr = 10 * np.log10(uf_pts)
    fig, ax = plt.subplots(figsize=(9, 6))
    # Draw contour lines on an unstructured triangular grid.
    ax.tripcolor(x, y, tri, uf_logpwr, cmap=plt.cm.viridis)
    ax.tricontour(x, y, tri, uf_logpwr, 10, cmap=plt.cm.hot)
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    ax.set_title("E field (logmag)")
    plt.show()
