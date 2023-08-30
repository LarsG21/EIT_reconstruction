import cv2
import numpy as np

from Data_Generation.utils import solve_eit_using_jac
from plot_utils import plot_results_fem_forward
from pyeit import mesh
from pyeit.eit import protocol
from pyeit.eit.fem import EITForward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from scipy.ndimage import rotate

img_size = 64
v0 = None


def generate_sample_mesh_simulation(mesh_obj, n_el=32):
    """
    Generates a sample simulation of electrode voltages with a random anomaly.
    """
    global v0
    """ 1. problem setup """
    anomaly_list = [PyEITAnomaly_Circle(center=[0.8, 0], r=0.2, perm=10.0),
                    # PyEITAnomaly_Circle(center=[-0.74, 0], r=0.2, perm=10.0),
                    # PyEITAnomaly_Circle(center=[0, 0.72], r=0.2, perm=0.1)
                    ]
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

    PLOT = True
    if PLOT:
        cv2.imshow("img", cv2.resize(img, (256, 256)))
        cv2.waitKey(1000)

    """ 2. FEM simulation """
    # setup EIT scan conditions
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

    # calculate simulated data
    fwd = EITForward(mesh_obj, protocol_obj)
    if v0 is None:
        v0 = fwd.solve_eit()
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    # # rotate the voltages and the image by 90 degrees clockwise
    img = rotate(img, +90, reshape=False)

    cv2.imshow("img_rot", cv2.resize(img, (256, 256)))
    cv2.waitKey(1000)
    # rotate the voltage vector by n_el/4
    v0 = np.roll(v0, 32 * int(n_el / 4))
    v1 = np.roll(v1, 32 * int(n_el / 4))

    # For control you can solve the eit and plot the result
    if PLOT:
        solve_eit_using_jac(mesh_new, mesh_obj, protocol_obj, v0, v1)
        cv2.waitKey(1000)
        plot_results_fem_forward(mesh=mesh_new, line=protocol_obj.ex_mat[0])

    return v0, v1, img


if __name__ == '__main__':
    mesh_obj = mesh.create(32, h0=0.1)
    generate_sample_mesh_simulation(mesh_obj=mesh_obj)
