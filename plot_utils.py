from __future__ import division, absolute_import, print_function

import os

import cv2
from matplotlib import pyplot as plt

from Model_Training.model_plot_utils import infer_single_reconstruction
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import pdegrad, sim2pts

import numpy as np
import matplotlib.pyplot as plt

from utils import find_center_of_mass


def plot_results_fem_forward(mesh, line):
    """
    Plot results of FEM forward simulation. Plots the equi-potential lines and the E-Field lines.
    :param mesh: the mesh object used for the FEM forward simulation
    :param line: Input and output electrodes as a 2x1 nparray
    :return: Tuple of PIL Image objects containing the equi-potential and E-field plots
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

    """ 2. plot equi-potential lines """
    fig_equipotential, ax1 = plt.subplots(figsize=(9, 6))
    # draw equi-potential lines
    vf = np.linspace(min(f), max(f), 64)
    # vf = np.sort(f[el_pos])
    # Draw contour lines on an unstructured triangular grid.
    contour = ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)
    # draw mesh structure
    # Create a pseudocolor plot of an unstructured triangular grid
    mesh_plot = ax1.tripcolor(
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
    # ax1.set_title("equi-potential lines")
    # clean up
    ax1.set_aspect("equal")
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_xlim([-1.2, 1.2])
    fig_equipotential.set_size_inches(6, 6)

    # Render the plot to an image
    # Draw the content
    fig_equipotential.canvas.draw()
    width, height = fig_equipotential.canvas.get_width_height()
    image_array = np.frombuffer(fig_equipotential.canvas.tostring_rgb(), dtype='uint8')
    equi_potential_image = image_array.reshape(height, width, 3)

    """ 3. plot E field (logmag) """
    ux, uy = pdegrad(pts, tri, f)
    uf = ux ** 2 + uy ** 2
    uf_pts = sim2pts(pts, tri, uf)
    uf_logpwr = 10 * np.log10(uf_pts)
    fig_e_field, ax = plt.subplots(figsize=(9, 6))
    # Draw contour lines on an unstructured triangular grid.
    field_plot = ax.tripcolor(x, y, tri, uf_logpwr, cmap=plt.cm.viridis)
    ax.tricontour(x, y, tri, uf_logpwr, 10, cmap=plt.cm.hot)
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    ax.set_title("E field (logmag)")

    # Render the plot to an image
    fig_e_field.canvas.draw()
    width, height = fig_e_field.canvas.get_width_height()
    image_array = np.frombuffer(fig_e_field.canvas.tostring_rgb(), dtype='uint8')
    e_field_image = image_array.reshape(height, width, 3)
    # clear the figure
    plt.close(fig_equipotential)
    plt.close(fig_e_field)
    return equi_potential_image, e_field_image


def solve_and_plot_with_nural_network(model, model_input, original_image=None, save_path=None,
                                      title="Reconstructed image",
                   chow_center_of_mass=False, use_opencv_for_plotting=False):
    """
    Plot the results of the model inference. Plots the reconstructed image, the EIT image and the EIT image with
    the center of mass of the image.
    :param model: A trained model (nn.Module)
    :param model_input: Numpy array or torch Tensor
    :param original_image:
    :param save_path:
    :param title:
    :param chow_center_of_mass:
    :param use_opencv_for_plotting:
    :return:
    """
    img, img_binary, img_non_thres = infer_single_reconstruction(model, model_input, title=title,
                                                                 original_image=original_image,
                                                                 save_path=save_path, detection_threshold=0.25,
                                                                 show=False, debug=False)
    # GREIT EVAL PARAMETERS USE THRESHOLD 0.25
    SCALE_FACTOR = 8
    # upscale image by 2
    imshow = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    # add circle to image
    cv2.circle(imshow, (imshow.shape[0] // 2, imshow.shape[0] // 2), imshow.shape[0] // 2, 1, 1)
    if chow_center_of_mass:
        center_of_mass = find_center_of_mass(img)
        cv2.circle(imshow, (center_of_mass[0] * SCALE_FACTOR, center_of_mass[1] * SCALE_FACTOR), 5, -1, -1)
    if use_opencv_for_plotting:
        cv2.imshow(title, imshow)
        cv2.waitKey(1)
    if not use_opencv_for_plotting or save_path is not None:
        plt.imshow(imshow)
        plt.title(title)
        # save as pdf
        if save_path is not None:  # PLOT_THESIS
            plt.savefig(os.path.join(save_path, title + ".pdf"))
        # plt.show()
    return img


def solve_and_get_center_with_nural_network(model, model_input,
                                            debug=False):
    """
    Solve the reconstruction and return the center of mass of the image
    :param model:
    :param model_input:
    :return:
    """
    img, img_binary, img_non_thresh = infer_single_reconstruction(model, model_input, detection_threshold=0.25,
                                                                  show=False)
    center_of_mass = find_center_of_mass(img)
    # show center of mass in image matplotlib
    if debug:  # PLOT_THESIS
        SCALE_FACTOR = 1
        imshow = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        # add an marker with matplotlib
        plt.plot(center_of_mass[0], center_of_mass[1], "ro")
        plt.legend(["center of gravity"])
        plt.imshow(imshow)
        plt.colorbar()
        plt.title("Detected center of gravity")
        plt.xlabel("x [pixels]")
        plt.ylabel("y [pixels]")
        plt.savefig(os.path.join("C:\\Users\\lgudjons\\Desktop", "COG" + ".pdf"))
        plt.show()
        # # plot slice row and column of the image at the center of mass
        # plt.plot(img[:, center_of_mass[0]])
        # plt.plot(img[center_of_mass[1], :])
        # plt.title("Slice row and column at center of mass")
        # plt.legend(["row", "column"])
        # plt.show()
    return img, center_of_mass, img_non_thresh


def preprocess_greit_img(img_greit):
    """
    Preprocess the greit image. This includes:
    - subtracting the mean
    - normalizing the image
    - setting all abs(values) below 0.25 to 0
    - inverting the colors to show negative conductivity changes as positive
    :param img_greit:
    :return:
    """
    # Normalize image
    img_greit = img_greit - np.mean(img_greit)
    img_greit = img_greit / np.max(np.abs(img_greit))
    # set all values below 0.25 to 0
    img_greit[np.abs(img_greit) < 0.25] = 0
    # invert colors
    img_greit = -img_greit
    # Mask the image to only show the region of interest
    mask = np.load("mask.npy")
    mask = mask.reshape((32, 32))
    # invert mask
    mask = 1 - mask
    # # erode mask
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    img_greit = img_greit * mask
    # increase resolution by 2
    img_greit = cv2.resize(img_greit, (0, 0), fx=2, fy=2)
    img_greit = np.clip(img_greit, 0, 1)
    return img_greit


def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):
    '''Display grid search results

    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          '''
    from matplotlib import pyplot as plt
    from IPython.display import display
    import pandas as pd

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()
