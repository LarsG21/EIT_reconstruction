import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms

# Step 2: Prepare the dataset (assuming you have custom dataset in numpy arrays)
class CustomDataset(data.Dataset):
    def __init__(self, voltage_data, image_data, transform=None):
        self.voltage_data = voltage_data
        self.image_data = image_data
        self.transform = transform

    def __len__(self):
        return len(self.voltage_data)

    def __getitem__(self, index):
        voltage = self.voltage_data[index]
        image = self.image_data[index]

        if self.transform:
            voltage = self.transform(voltage)

        return voltage, image


# Step 3: Create the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(104, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64 * 64),
            nn.Sigmoid(),  # Sigmoid activation to ensure pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # Step 1: Install required libraries (PyTorch)

    # Step 2: Prepare the dataset
    # Assuming you have 'voltage_data' and 'image_data' as your numpy arrays
    # Convert them to PyTorch tensors and create DataLoader

    voltage_data_np = np.load('Edinburgh mfEIT Dataset/img1.npy')
    image_data_np = np.load('Edinburgh mfEIT Dataset/V1.npy')

    voltage_data_tensor = torch.tensor(voltage_data_np, dtype=torch.float32)
    image_data_tensor = torch.tensor(image_data_np, dtype=torch.float32)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(voltage_data_np, image_data_tensor, transform)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 3: Create the CNN model
    model = CNNModel()

    # Step 4: Define the training loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_voltages, batch_images in dataloader:
            # Forward pass
            outputs = model(batch_voltages)

            # Compute loss
            loss = criterion(outputs, batch_images.view(-1, 64 * 64))

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # After training, you can use the model to reconstruct images
    # by passing voltage data to the model's forward method.