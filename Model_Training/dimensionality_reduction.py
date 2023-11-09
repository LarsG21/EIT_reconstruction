import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def perform_pca_on_input_data(voltage_data_tensor, image_data_tensor, train_voltage, val_voltage, test_voltage,
                              model_path, device,
                              n_components=128, debug=True, train_images=None):
    """
    Performs PCA on the input data and returns the transformed data.
    Saves the PCA model for later reconstruction-
    Load the oca again: pca = pickle.load(open("pca.pkl", "rb"))
    Transform original data: v1 = pca.transform(v1.reshape(1, -1))
    :param voltage_data_tensor:
    :param train_voltage:
    :param val_voltage:
    :param test_voltage:
    :param model_path: path to save the pca model
    :param device: cuda or cpu
    :param n_components: number of principal components to keep
    :return:
    """
    transform_back_to_tensor = False
    pca = PCA(n_components=n_components)
    if type(voltage_data_tensor) == torch.Tensor:
        voltage_data_tensor_np = voltage_data_tensor.cpu().numpy()
        transform_back_to_tensor = True
    elif type(voltage_data_tensor) == np.ndarray:
        voltage_data_tensor_np = voltage_data_tensor
    else:
        raise ValueError("voltage_data_tensor must be a numpy array or torch tensor")
    pca.fit(voltage_data_tensor_np)
    # save the pca for later reconstruction
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    with open(os.path.join(model_path, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    if transform_back_to_tensor:
        train_voltage_to_transform = train_voltage.cpu().numpy()
        val_voltage_to_transform = val_voltage.cpu().numpy()
        test_voltage_to_transform = test_voltage.cpu().numpy()
    else:
        train_voltage_to_transform = train_voltage
        val_voltage_to_transform = val_voltage
        test_voltage_to_transform = test_voltage
    # transform the data
    train_voltage = pca.transform(train_voltage_to_transform)
    val_voltage = pca.transform(val_voltage_to_transform)
    test_voltage = pca.transform(test_voltage_to_transform)
    #
    if debug:
        # how mouch variance is explained by the first n components
        print(f"Explained variance ratio of the first {n_components} components: "
              f"{np.sum(pca.explained_variance_ratio_[:n_components])}")
        for i in range(0, 127):
            analyze_principal_component(train_images, train_voltage, component_index=i)
    # reconstruct_images_from_pca(pca, train_images, train_voltage, voltage_data_tensor,
    #                             image_data_tensor, n_components=80)
    reconstruct_voltages_from_pca(pca, voltage_data_tensor, n_components=128)
    # transform back to tensor
    if transform_back_to_tensor:
        train_voltage = torch.tensor(train_voltage, dtype=torch.float32).to(device)
        val_voltage = torch.tensor(val_voltage, dtype=torch.float32).to(device)
        test_voltage = torch.tensor(test_voltage, dtype=torch.float32).to(device)
    return train_voltage, val_voltage, test_voltage, pca


def reconstruct_voltages_from_pca(pca, voltage_data_tensor, n_components=128):
    """
    Reconstructs the voltages from the pca.
    :param pca: pca model

    :param n_components: number of principal components to use for reconstruction
    :return:
    """
    voltage_data_tensor_np = voltage_data_tensor.cpu().numpy()
    # transform the data
    pca_transformation_of_voltage = pca.transform(voltage_data_tensor)
    # get the eigenvoltages of the pca
    eigen_voltages = pca.components_
    print(f"Eigen voltages shape: {eigen_voltages.shape}")
    # reconstruct the voltages
    reconstructed_voltages = []
    for i in range(0, len(voltage_data_tensor_np)):
        voltage_tensor_rec = np.zeros_like(voltage_data_tensor_np[0])
        for j in range(0, n_components):
            voltage_tensor_rec += pca_transformation_of_voltage[0, j] * eigen_voltages[j]
            # plt.plot(voltage_tensor_rec)
            # plt.title(f"Reconstructed voltage {i} after {j} components")
            # plt.show()
            # print(f"Reconstructed voltage {i} after {j} components")
        # add the mean
        voltage_tensor_rec += pca.mean_
        reconstructed_voltages.append(voltage_tensor_rec)
        # plot the original and reconstructed voltages
        plt.plot(voltage_data_tensor_np[i], label="original")
        plt.plot(voltage_tensor_rec, label="reconstructed")
        plt.legend()
        plt.title(f"Original and reconstructed voltage {i}")
        plt.show()
        # plot the difference
        plt.plot(voltage_data_tensor_np[i] - voltage_tensor_rec)
        plt.title(f"Difference between original and reconstructed voltage {i}")
        plt.show()

        print("ok")


def reconstruct_images_from_pca(pca, train_images, train_voltage, voltage_data_tensor,
                                image_data_tensor, n_components=128):
    """
    Reconstructs the images from the pca.
    :param pca: pca model
    :param voltage_data_tensor: voltage data tensor
    :param n_components: number of principal components to use for reconstruction
    :return:
    """
    voltage_data_tensor_np = voltage_data_tensor.cpu().numpy()
    image_data_tensor_np = image_data_tensor.cpu().numpy()
    # transform the data
    voltage_data_tensor = pca.transform(voltage_data_tensor_np)
    # get the eigenimages of the pca
    eigenimages_pos = []
    eigenimages_neg = []
    for i in range(0, n_components):
        mean_train_images_pc1_neg, mean_train_images_pc1_pos = get_pos_and_neg_pc_image(i, train_images,
                                                                                        train_voltage)
        # plt.imshow(mean_train_images_pc1_pos)
        # plt.title(f"mean of train images with pc{i} > 0")
        # plt.show()
        # plt.imshow(mean_train_images_pc1_neg)
        # plt.title(f"mean of train images with pc{i} < 0")
        # plt.show()
        print(f"Current principal component: {i}")
        eigenimages_pos.append(mean_train_images_pc1_pos)
        eigenimages_neg.append(mean_train_images_pc1_neg)

    # reconstruct the images
    reconstructed_images = []
    for i in range(0, len(voltage_data_tensor)):
        reconstructed_image = np.zeros((64, 64))
        for j in range(0, n_components):
            if voltage_data_tensor[i][j] > 0:
                reconstructed_image += voltage_data_tensor[i][j] * eigenimages_pos[j]
            else:
                reconstructed_image -= voltage_data_tensor[i][j] * eigenimages_neg[j]
            # if j % 3 == 0:
            # reconstructed_image -= np.mean(image_data_tensor_np, axis=0)
            # # set the values to 0 that are smaller than 70% of the max value
            # reconstructed_image[reconstructed_image < 0.6 * np.max(reconstructed_image)] = 0
            # reconstructed_image_copy_plot = reconstructed_image.copy()
            # reconstructed_image_copy_plot -= np.ones_like(reconstructed_image_copy_plot)
            # reconstructed_image_copy_plot[reconstructed_image_copy_plot < 0.6 * np.max(reconstructed_image_copy_plot)] = 0
            # plt.imshow(reconstructed_image_copy_plot)
            # plt.title(f"reconstructed image of voltage {i} after {j} principal components")
            # plt.show()
        # subtract the mean image
        reconstructed_image -= np.ones_like(reconstructed_image)
        # set the values to 0 that are smaller than 70% of the max value
        reconstructed_image[reconstructed_image < 0.6 * np.max(reconstructed_image)] = 0
        # plot the reconstructed image
        plt.imshow(reconstructed_image)
        plt.title(f"reconstructed image final")
        plt.show()
        # plot the original image
        plt.imshow(image_data_tensor[i])
        plt.title(f"original image of voltage {i}")
        plt.show()
        print("OK")

        reconstructed_images.append(reconstructed_image)


def analyze_principal_component(train_images, train_voltage, component_index):
    """
    Shows the mean of the train images of the corresponding voltages with principal_component > 0 and principal_component < 0
    :param train_images: train images to look at
    :param train_voltage: pca of the train voltages in the same order as train_images (pc0, pc1, pc2, ...)
    :param component_index: index of the principal component to look at
    """
    mean_train_images_pc1_neg, mean_train_images_pc1_pos = get_pos_and_neg_pc_image(component_index, train_images,
                                                                                    train_voltage)
    # compare those images
    import matplotlib.pyplot as plt
    # same but with subplots
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(mean_train_images_pc1_pos)
    axs[0].set_title(f"mean of train images with pc{component_index} > 0")
    axs[1].imshow(mean_train_images_pc1_neg)
    axs[1].set_title(f"mean of train images with pc{component_index} < 0")
    # save the figure
    plt.savefig(f"mean_train_images_pc{component_index}.png")
    plt.show()


def get_pos_and_neg_pc_image(component_index, train_images, train_voltage):
    train_images = train_images.cpu().numpy()
    # get indices of train_voltages with pc0 > 0
    indices = np.where(train_voltage[:, component_index] > 0)[0]
    # get the corresponding train_images
    train_images_pc1_pos = train_images[indices]
    # get mean of those
    mean_train_images_pc1_pos = np.mean(train_images_pc1_pos, axis=0)
    # get indices of train_voltages with pc0 < 0
    indices = np.where(train_voltage[:, component_index] < 0)[0]
    # get the corresponding train_images
    train_images_pc1_neg = train_images[indices]
    # get mean of those
    mean_train_images_pc1_neg = np.mean(train_images_pc1_neg, axis=0)
    return mean_train_images_pc1_neg, mean_train_images_pc1_pos
