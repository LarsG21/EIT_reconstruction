import os

import numpy as np
import torch
from torch.utils import data as data


class CustomDataset(data.Dataset):
    def __init__(self, voltage_data, image_data):
        self.voltage_data = voltage_data
        self.image_data = image_data

    def __len__(self):
        return len(self.voltage_data)

    def __getitem__(self, index):
        voltage = self.voltage_data[index]
        image = self.image_data[index]

        # if self.transform:
        #     voltage = self.transform(voltage)

        return voltage, image


def get_test_data_loader(test_data_path, device, pca=None):
    """
    Loads the test data from the given path and returns a dataloader
    :param test_data_path:
    :return:
    """
    voltage_data_np = np.load(os.path.join(test_data_path, "v1_array.npy"))
    image_data_np = np.load(os.path.join(test_data_path, "img_array.npy"))
    if pca is not None:
        voltage_data_np = pca.transform(voltage_data_np)
    test_voltage = torch.tensor(voltage_data_np, dtype=torch.float32).to(device)
    test_images = torch.tensor(image_data_np, dtype=torch.float32).to(device)
    test_dataset = CustomDataset(test_voltage, test_images)
    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return test_dataloader, test_images, test_voltage
