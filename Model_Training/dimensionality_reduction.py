import os
import pickle

import torch
from sklearn.decomposition import PCA


def perform_pca_on_input_data(voltage_data_tensor, train_voltage, val_voltage, test_voltage, model_path, device,
                              n_components=128):
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
    pca = PCA(n_components=n_components)
    voltage_data_tensor_np = voltage_data_tensor.cpu().numpy()
    pca.fit(voltage_data_tensor_np)
    # save the pca for later reconstruction
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    with open(os.path.join(model_path, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    # transform the data
    train_voltage = pca.transform(train_voltage.cpu().numpy())
    val_voltage = pca.transform(val_voltage.cpu().numpy())
    test_voltage = pca.transform(test_voltage.cpu().numpy())
    # transform back to tensor
    train_voltage = torch.tensor(train_voltage, dtype=torch.float32).to(device)
    val_voltage = torch.tensor(val_voltage, dtype=torch.float32).to(device)
    test_voltage = torch.tensor(test_voltage, dtype=torch.float32).to(device)
    return train_voltage, val_voltage, test_voltage
