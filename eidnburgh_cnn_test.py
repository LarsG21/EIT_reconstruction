import os.path
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

LOSS_SCALE_FACTOR = 1

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

        # if self.transform:
        #     voltage = self.transform(voltage)

        return voltage, image

VOLTAGE_VECTOR_LENGTH = 104
OUTPUT_IMAGE_SIZE = 64 * 64

# Step 3: Create the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(VOLTAGE_VECTOR_LENGTH, 128),
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
            nn.Linear(128, OUTPUT_IMAGE_SIZE),
            # nn.Sigmoid(),  # Sigmoid activation to ensure pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def calc_average_loss_completly_black(image_data_tensor, criterion):
    """
    To check if the network is learning anything at all, we can check the average loss of the network when the input is
    completly black. If the network is learning anything, the average loss should be high.
    (Check if criterion is suitable for this)
    :param image_data_tensor:
    :param voltage_data_tensor:
    :param criterion:
    :return:
    """
    # Display all reconstructed images
    average_loss = 0
    for img in image_data_tensor:
        output = np.zeros((64, 64))
        loss = round(criterion(torch.tensor(output), img.view(64, 64)).item(), 4) * LOSS_SCALE_FACTOR
        average_loss += loss
    average_loss /= len(image_data_tensor) # easier to read
    print(f"Average loss black: {average_loss}")
    return average_loss


def calc_average_loss_completly_white(image_data_tensor, criterion):
    # Display all reconstructed images
    average_loss = 0
    for img in image_data_tensor:
        output = np.ones((64, 64))
        loss = round(criterion(torch.tensor(output), img.view(64, 64)).item(), 4) * LOSS_SCALE_FACTOR
        average_loss += loss
    average_loss /= len(image_data_tensor) # easier to read
    print(f"Average loss white: {average_loss}")
    return average_loss


def plot_reconstruction(image_data_tensor, voltage_data_tensor, model, criterion, num_images=40,
                        save_path=""):
    global output
    # Display all reconstructed images
    average_loss = 0
    # select random images
    random_indices = np.random.randint(0, len(image_data_tensor), num_images)
    for i in random_indices:
        img = image_data_tensor[i]
        img_numpy = img.view(64, 64).detach().numpy()
        volt = voltage_data_tensor[i]
        output = model(volt)
        output = output.view(64, 64).detach().numpy()
        cv2.imshow("Reconstructed Image", cv2.resize(output, (256, 256)))
        cv2.imshow("Original Image", cv2.resize(img_numpy, (256, 256)))
        # print loss for each image
        a = torch.tensor(output)
        b = img.view(64, 64)
        loss = round(criterion(a, b).item(), 4)
        loss = loss * LOSS_SCALE_FACTOR  # easier to read
        average_loss += loss
        cv2.waitKey(1)
        print(f"Loss: {loss}")
        # plot comparison with matplotlib
        plt.subplot(1, 2, 1)
        plt.imshow(output)
        plt.title(f"Loss: {loss}")
        plt.subplot(1, 2, 2)
        plt.imshow(img_numpy)
        plt.title("Original")
        # add colorbar
        plt.colorbar()
        if save_path != "":
            plt.savefig(os.path.join(save_path, f"reconstruction_{i}.png"))
        plt.show()
        time.sleep(0.1)
    return average_loss / num_images


def plot_loss(val_loss_list, loss_list=None, save_name=""):
    """
    Plots the loss and validation loss
    :param loss_list:
    :param val_loss_list:
    :return:
    """
    if loss_list is not None:
        plt.plot(loss_list)
    plt.plot(val_loss_list)
    if loss_list is not None:
        plt.legend(["Train", "Validation"])
    else:
        plt.legend(["Validation"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    if save_name != "":
        plt.savefig(save_name)


if __name__ == "__main__":
    TRAIN = False
    load_model_and_continue_trainig = False
    # Step 1: Install required libraries (PyTorch)

    # Step 2: Prepare the dataset
    # Assuming you have 'voltage_data' and 'image_data' as your numpy arrays
    # Convert them to PyTorch tensors and create DataLoader

    voltage_data_np = np.load("Edinburgh mfEIT Dataset/voltages.npy")
    image_data_np = np.load("Edinburgh mfEIT Dataset/images.npy")

    print("Overall data shape: ", voltage_data_np.shape)

    voltage_data_tensor = torch.tensor(voltage_data_np, dtype=torch.float32)
    image_data_tensor = torch.tensor(image_data_np, dtype=torch.float32)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(voltage_data_tensor, image_data_tensor)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # # Step 3: Create the CNN model
    model = CNNModel()

    print("model summary: ", model)


    # Step 4: Split the data into train, test, and validation sets
    # Assuming you have 'voltage_data_tensor' and 'image_data_tensor' as your PyTorch tensors
    # Note: Adjust the test_size and validation_size according to your preference.
    train_voltage, test_voltage, train_images, test_images = train_test_split(
        voltage_data_tensor, image_data_tensor, test_size=0.2, random_state=42
    )
    train_voltage, val_voltage, train_images, val_images = train_test_split(
        train_voltage, train_images, test_size=0.2, random_state=42
    )

    # Step 5: Create the DataLoader for train, test, and validation sets
    train_dataset = CustomDataset(train_voltage, train_images)
    train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    # number of training samples
    print("Number of training samples: ", len(train_dataset))

    test_dataset = CustomDataset(test_voltage, test_images)
    test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Number of test samples: ", len(test_dataset))

    val_dataset = CustomDataset(val_voltage, val_images)
    val_dataloader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("Number of validation samples: ", len(val_dataset))

    # # Step 5: Define the training loop
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()  # Didnt work well
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_black_img = calc_average_loss_completly_black(image_data_tensor=image_data_tensor,
                                      criterion=criterion)

    loss_white_img = calc_average_loss_completly_white(image_data_tensor=image_data_tensor,
                                      criterion=criterion)

    if TRAIN:
        if load_model_and_continue_trainig:
            model.load_state_dict(torch.load(
                "Edinburgh mfEIT Dataset/models_new_loss_methode/1/model_2023-07-27_16-11-48_250_epochs.pth"))
        num_epochs = 150
        loss_list = []
        val_loss_list = []
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            for batch_voltages, batch_images in train_dataloader:
                # Forward pass
                outputs = model(batch_voltages)

                # Compute loss
                loss = criterion(outputs, batch_images.view(-1, 64 * 64)) * LOSS_SCALE_FACTOR

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # After each epoch, evaluate the model on the validation set
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                val_loss = 0.0
                for batch_voltages, batch_images in val_dataloader:
                    outputs = model(batch_voltages)
                    val_loss += criterion(outputs, batch_images.view(-1, 64 * 64)).item()

                val_loss /= len(val_dataloader) * LOSS_SCALE_FACTOR
                val_loss_list.append(val_loss)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f} Loss: {loss.item():.4f}")

            loss_list.append(loss.item())
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            # plot loss every 10 epochs
            if epoch % 10 == 0:
                plot_loss(val_loss_list=val_loss_list, loss_list=loss_list,  save_name="")
                # save the model
                torch.save(model.state_dict(),
                           f"Edinburgh mfEIT Dataset/model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}_{num_epochs}.pth")

        # save the model
        torch.save(model.state_dict(),
                   f"Edinburgh mfEIT Dataset/model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{num_epochs}_epochs.pth")
        # plot the loss

        plot_loss(val_loss_list=val_loss_list, loss_list=loss_list, save_name="Edinburgh mfEIT Dataset/loss_plot.png")
    # load the model
    else:  # load the model
        print("Loading the model")
        model = CNNModel()
        model.load_state_dict(torch.load(
            "Edinburgh mfEIT Dataset/models_new_loss_methode/2/model_2023-07-27_16-38-33_60_150.pth"))
        model.eval()

    # After training, you can use the model to reconstruct images
    # by passing voltage data to the model's forward method.

    # Try inference on train images

    # plot_reconstruction(voltage_data_tensor=voltage_data_tensor, image_data_tensor=image_data_tensor,
    #                     model=model, criterion=criterion)

    # calc_average_loss_completly_black(voltage_data_tensor=voltage_data_tensor, image_data_tensor=image_data_tensor,
    #                                   criterion=criterion)
    #
    # calc_average_loss_completly_white(voltage_data_tensor=voltage_data_tensor, image_data_tensor=image_data_tensor,
    #                                   criterion=criterion)

    # Try inference on train images
    # print("Train images")
    # plot_reconstruction(train_images, train_voltage, model, criterion, num_images=20)

    # Try inference on test images
    print("Test images")
    plot_reconstruction(test_images, test_voltage, model, criterion, num_images=20,
                        save_path="Edinburgh mfEIT Dataset/models_new_loss_methode/2"
                        )
