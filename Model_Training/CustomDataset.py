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
