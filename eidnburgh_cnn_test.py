import copy
import os.path
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from CNN_Models import CNNModel, UNetModel, CNNModelWithDropout

LOSS_SCALE_FACTOR = 1000
VOLTAGE_VECTOR_LENGTH = 896
OUT_SIZE = 64


# VOLTAGE_VECTOR_LENGTH = 104
# OUT_SIZE = 64

# Step 2: Prepare the dataset (assuming you have custom dataset in numpy arrays)
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
        output = np.zeros((OUT_SIZE, OUT_SIZE))
        loss = round(criterion(torch.tensor(output), img.view(OUT_SIZE, OUT_SIZE)).item(), 4) * LOSS_SCALE_FACTOR
        average_loss += loss
    average_loss /= len(image_data_tensor)  # easier to read
    print(f"Average loss black: {average_loss}")
    return average_loss


def calc_average_loss_completly_white(image_data_tensor, criterion):
    # Display all reconstructed images
    average_loss = 0
    for img in image_data_tensor:
        output = np.ones((OUT_SIZE, OUT_SIZE))
        loss = round(criterion(torch.tensor(output), img.view(OUT_SIZE, OUT_SIZE)).item(), 4) * LOSS_SCALE_FACTOR
        average_loss += loss
    average_loss /= len(image_data_tensor)  # easier to read
    print(f"Average loss white: {average_loss}")
    return average_loss


def plot_sample_reconstructions(image_data_tensor, voltage_data_tensor, model, criterion, num_images=40,
                                save_path=""):
    global output
    # Display all reconstructed images
    average_loss = 0
    # select random images
    random_indices = np.random.randint(0, len(image_data_tensor), num_images)
    for i in random_indices:
        img = image_data_tensor[i]
        img_numpy = img.view(OUT_SIZE, OUT_SIZE).detach().numpy()
        volt = voltage_data_tensor[i]
        output = model(volt)
        output = output.view(OUT_SIZE, OUT_SIZE).detach().numpy()
        cv2.imshow("Reconstructed Image", cv2.resize(output, (256, 256)))
        cv2.imshow("Original Image", cv2.resize(img_numpy, (256, 256)))
        # print loss for each image
        a = torch.tensor(output)
        b = img.view(OUT_SIZE, OUT_SIZE)
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


def plot_single_reconstruction(model, voltage_data, title = "Reconstructed image", original_image:np.array=None):
    """
    Plots a single reconstruction using the model
    :param model:
    :param voltage_data:
    :return:
    """
    if type(voltage_data) == np.ndarray:
        voltage_data = torch.tensor(voltage_data, dtype=torch.float32)
    elif type(voltage_data) == torch.Tensor:
        # if dtype is not float32, convert it
        if voltage_data.dtype != torch.float32:
            voltage_data = voltage_data.type(torch.float32)
    else:
        raise TypeError(
            f"voltage_data_tensor must be either a numpy array or a torch tensor but is {type(voltage_data)}")
    start = time.time()
    output = model(voltage_data)
    stop = time.time()
    print(f"Time for reconstruction: {(stop - start) * 1000} ms")
    output = output.view(OUT_SIZE, OUT_SIZE).detach().numpy()
    if original_image is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(output)
        plt.title(title)
        plt.subplot(1, 2, 2)
        plt.imshow(original_image)
        plt.title("Original")
        # add colorbar
        plt.colorbar()
        plt.show()
    else:
        plt.imshow(output)
        plt.title(title)
        # add colorbar
        plt.colorbar()
        plt.show()


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
    if save_name != "":
        plt.savefig(save_name)
    plt.show()



if __name__ == "__main__":
    TRAIN = True
    load_model_and_continue_trainig = False
    SAVE_CHECKPOINTS = False
    LOSS_PLOT_INTERVAL = 10
    # Training parameters
    num_epochs = 150
    NOISE_LEVEL = 0.05
    # NOISE_LEVEL = 0
    LEARNING_RATE = 0.0005
    # Define the weight decay factor
    weight_decay = 1e-5  # Adjust this value as needed (L2 regularization)
    # weight_decay = 0  # Adjust this value as needed (L2 regularization)
    # Define early stopping parameters
    patience = 30  # Number of epochs to wait for improvement

    best_val_loss = float('inf')  # Initialize with a very high value
    counter = 0  # Counter to track epochs without improvement

    # path = "Edinburgh mfEIT Dataset"
    path = "Own_Simulation_Dataset"
    model_name = "Test_noise_03_regularization_1e-5"
    model_name = "TESTING"
    # model_name = f"model{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_path = os.path.join(path, "Models", model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # MODEL_PATH = "Own_Simulation_Dataset/Models"

    # Step 1: Install required libraries (PyTorch)

    # Step 2: Prepare the dataset
    # Assuming you have 'voltage_data' and 'image_data' as your numpy arrays
    # Convert them to PyTorch tensors and create DataLoader

    # voltage_data_np = np.load(os.path.join(path, "voltages.npy"))
    # image_data_np = np.load(os.path.join(path, "images.npy"))

    voltage_data_np = np.load("Own_Simulation_Dataset/v1_array.npy")
    image_data_np = np.load("Own_Simulation_Dataset/img_array.npy")
    v0 = np.load("Own_Simulation_Dataset/v0.npy")
    # subtract v0 from all voltages
    voltage_data_np = voltage_data_np - v0
    # Now the model should learn the difference between the voltages and v0 (default state)

    print("Overall data shape: ", voltage_data_np.shape)

    voltage_data_tensor = torch.tensor(voltage_data_np, dtype=torch.float32)
    image_data_tensor = torch.tensor(image_data_np, dtype=torch.float32)

    dataset = CustomDataset(voltage_data_tensor, image_data_tensor)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # # Step 3: Create the model
    model = CNNModel(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
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

    # Step 4.1 Adding noise to the training data in % of max value
    maximum = torch.max(torch.abs(train_voltage)).item()
    noise_amplitude = NOISE_LEVEL * maximum
    print("Absolute max value: ", torch.max(torch.abs(train_voltage)).item())
    train_voltage = train_voltage + torch.randn(train_voltage.shape) * noise_amplitude
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

    # Step 6: Define the loss function and optimizer
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()  # Didnt work well

    # Initialize the optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    loss_black_img = calc_average_loss_completly_black(image_data_tensor=image_data_tensor,
                                                       criterion=criterion)

    loss_white_img = calc_average_loss_completly_white(image_data_tensor=image_data_tensor,
                                                       criterion=criterion)

    if TRAIN:
        # Step 7: Define the training loop
        if load_model_and_continue_trainig:
            model.load_state_dict(torch.load(
                os.path.join(model_path, "MODEL_NAME.pth")))
        loss_list = []
        val_loss_list = []
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            for batch_voltages, batch_images in train_dataloader:
                # Forward pass
                outputs = model(batch_voltages)

                # Compute loss
                loss = criterion(outputs, batch_images.view(-1, OUT_SIZE ** 2)) * LOSS_SCALE_FACTOR

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
                    val_loss += criterion(outputs, batch_images.view(-1, OUT_SIZE ** 2)).item() * LOSS_SCALE_FACTOR

                val_loss /= len(val_dataloader)
                if val_loss < best_val_loss:    # Early stopping
                    best_val_loss = val_loss
                    counter = 0
                    best_model = copy.deepcopy(model)
                else:
                    counter += 1
                    print(f"Early stopping in {patience-counter} epochs")
                    if counter >= patience:
                        print("Early stopping triggered. No improvement in validation loss.")
                        # save the model
                        torch.save(best_model.state_dict(),
                                      os.path.join(model_path,
                                                    f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_epoche_{epoch}_of_{num_epochs}_best_model.pth"))
                        break
                val_loss_list.append(val_loss)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {round(val_loss, 4)} Training Loss: {round(loss.item(), 4)}")

            loss_list.append(loss.item())
            # plot loss every N epochs
            if epoch % LOSS_PLOT_INTERVAL == 0 and epoch != 0:
                plot_loss(val_loss_list=val_loss_list, loss_list=loss_list, save_name="")
                # save the model
                if SAVE_CHECKPOINTS:
                    torch.save(model.state_dict(),
                               os.path.join(model_path,
                                            f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}_{num_epochs}.pth"))
                # also create a sample reconstruction with the current model
                test_voltage_data = test_voltage[0]
                plot_single_reconstruction(model=model, voltage_data=test_voltage_data,
                                           title=f"Reconstruction after {epoch} epochs", original_image=test_images[0])
                # plot the corresponding image
        # save the final model
        torch.save(model.state_dict(),
                   os.path.join(model_path,
                                f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{num_epochs}_epochs.pth"))
        # plot the final loss
        plot_loss(val_loss_list=val_loss_list, loss_list=loss_list, save_name=os.path.join(model_path, "loss_plot.png"))
    # load the model
    else:  # load the model
        print("Loading the model")
        model = CNNModel(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2)
        # model.load_state_dict(torch.load(
        #     "Edinburgh mfEIT Dataset/models_new_loss_methode/2/model_2023-07-27_16-38-33_60_150.pth"))
        model.load_state_dict(torch.load(
            "Own_Simulation_Dataset/Models/model2023-08-02_18-32-46/model_2023-08-02_18-33-07_10_epochs.pth"))
        model.eval()

    # After training, you can use the model to reconstruct images
    # by passing voltage data to the model's forward method.

    # Try inference on train images
    # Try inference on train images
    # print("Train images")
    # plot_sample_reconstructions(train_images, train_voltage, model, criterion, num_images=20)

    # Try inference on test images
    print("Test_max_noise_05 images")
    plot_sample_reconstructions(test_images, test_voltage, model, criterion, num_images=20,
                                save_path=model_path)

    # single_datapoint = voltage_data_np[0]
    # voltage_data_tensor = torch.tensor(single_datapoint, dtype=torch.float32)
    # plot_single_reconstruction(model=model, voltage_data=voltage_data_tensor)

    # Try inference on test images
    # print("2023_08_02_15_30 images")
    # plot_reconstruction(test_images, test_voltage, model, criterion, num_images=20,
    #                     save_path="Edinburgh mfEIT Dataset/models_new_loss_methode/2"
    #                     )
