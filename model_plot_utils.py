import os
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

# from eidnburgh_cnn_test import OUT_SIZE, LOSS_SCALE_FACTOR

LOSS_SCALE_FACTOR = 1000
OUT_SIZE = 64

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
    """
    Creates a few random reconstructions and plots them
    :param image_data_tensor:
    :param voltage_data_tensor:
    :param model:
    :param criterion:
    :param num_images:
    :param save_path:
    :return:
    """
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


def plot_single_reconstruction(model, voltage_data, title = "Reconstructed image", original_image:np.array=None, save_path=None):
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
    else:
        plt.imshow(output)
        plt.title(title)
        # add colorbar
        plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
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
