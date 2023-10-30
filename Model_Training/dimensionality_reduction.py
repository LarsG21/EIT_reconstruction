import os
import pickle

import numpy as np
import torch
from sklearn.decomposition import PCA


def perform_pca_on_input_data(voltage_data_tensor, train_voltage, val_voltage, test_voltage, model_path, device,
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
    # transform back to tensor
    if transform_back_to_tensor:
        train_voltage = torch.tensor(train_voltage, dtype=torch.float32).to(device)
        val_voltage = torch.tensor(val_voltage, dtype=torch.float32).to(device)
        test_voltage = torch.tensor(test_voltage, dtype=torch.float32).to(device)
    return train_voltage, val_voltage, test_voltage, pca


def analyze_principal_component(train_images, train_voltage, component_index):
    """
    Shows the mean of the train images of the corresponding voltages with principal_component > 0 and principal_component < 0
    :param train_images: train images to look at
    :param train_voltage: pca of the train voltages in the same order as train_images (pc0, pc1, pc2, ...)
    :param component_index: index of the principal component to look at
    """
    train_images = train_images.cpu().numpy()
    # get indices of train_voltages with pc0 > 0
    indices = np.where(train_voltage[:, component_index] > 0)[0]
    # get the corresponding train_images
    train_images_pc1_pos = train_images[indices]
    # get mean of those
    mean_train_images_pc1_pos = np.mean(train_images_pc1_pos, axis=0)
    # get indices of train_voltages with pc0 < 0
    indices = np.where(train_voltage[:, component_index] < -0)[0]
    # get the corresponding train_images
    train_images_pc1_neg = train_images[indices]
    # get mean of those
    mean_train_images_pc1_neg = np.mean(train_images_pc1_neg, axis=0)
    # compare thos images
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
