import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split

from CustomDataset import CustomDataset
import torch.utils.data as data


def add_noise_augmentation(train_voltage, train_images, number_of_augmentations, noise_level, device="cpu",
                           show_examples=False, save_examples=False):
    """"
    Adds noise to the training data and augments the data
    :param train_voltage: The training voltages
    :param train_images: The training images
    :param number_of_augmentations: How many copies of the training data should be created
    :param noise_level: The noise level in % of the standard deviation of the data
    :return: The augmented training voltages and images
    """
    # Step 4.1 Adding noise to the training data in % of max value
    if number_of_augmentations == 0:
        return train_voltage, train_images
    std_of_data = torch.std(train_voltage).item()
    noise_amplitude = noise_level * std_of_data
    print("Absolute max value: ", std_of_data)
    train_voltages_combined = train_voltage
    train_images_combined = train_images
    for i in range(number_of_augmentations):
        train_voltage_augmented = train_voltage + noise_amplitude * torch.randn(train_voltage.shape).to(device)
        train_voltages_combined = torch.cat((train_voltages_combined, train_voltage_augmented), dim=0)
        train_images_combined = torch.cat((train_images_combined, train_images), dim=0)
    if show_examples:
        # plot the first 10 examples
        for i in range(10):
            plt.figure()
            plt.title("Example " + str(i))
            plt.plot(train_voltages_combined[i, :].cpu().numpy())
            plt.show()
    if save_examples:
        # save the first 10 examples
        for i in range(10):
            plt.figure()
            plt.title("Example " + str(i))
            plt.plot(train_voltages_combined[i, :].cpu().numpy())
            plt.savefig("Example_noise_augment" + str(i) + ".png")
    return train_voltages_combined, train_images_combined


def add_rotation_augmentation(train_voltage, train_images, number_of_augmentations=1, show_examples=False,
                              save_examples=False, device="cpu"):
    """
    Rotates the training data and augments the data
    :param train_voltage:
    :param train_images:
    :param number_of_augmentations:
    :return:
    """
    if number_of_augmentations == 0:
        return train_voltage, train_images
    print("Rotating the training data")
    # convert tensors to numpy arrays
    train_voltage_numpy = train_voltage.cpu().numpy()
    train_images_numpy = train_images.cpu().numpy()
    # rotate all images by 90° using rotate function scipy
    train_voltage_rotated_combined = train_voltage
    train_images_rotated_combined = train_images
    train_voltage_rotated_combined = train_voltage_rotated_combined.cpu()
    train_images_rotated_combined = train_images_rotated_combined.cpu()
    for i in range(number_of_augmentations):
        train_images_rotated, train_voltage_rotated = generate_rotation_augmentation(train_images_numpy,
                                                                                     train_voltage_numpy,
                                                                                     show_examples=show_examples,
                                                                                     save_examples=save_examples,
                                                                                     device=device)
        train_voltage_rotated = train_voltage_rotated.cpu()
        train_images_rotated = train_images_rotated.cpu()
        train_voltage_rotated_combined = np.concatenate((train_voltage_rotated_combined, train_voltage_rotated), axis=0)
        train_images_rotated_combined = np.concatenate((train_images_rotated_combined, train_images_rotated), axis=0)
    # transfer back to gpu
    train_voltage_rotated_combined = torch.from_numpy(train_voltage_rotated_combined).to(device)
    train_images_rotated_combined = torch.from_numpy(train_images_rotated_combined).to(device)

    return train_voltage_rotated_combined, train_images_rotated_combined


def generate_rotation_augmentation(train_images_numpy, train_voltage_numpy, device="cpu", show_examples=False,
                                   save_examples=False):
    img_rotated_list = []
    voltage_rotated_list = []
    DEGREES_PER_ELECTRODE = 11.25
    for img, voltage in zip(train_images_numpy, train_voltage_numpy):
        # use 11.25° steps for the rotation
        angle = np.random.randint(0, 32) * DEGREES_PER_ELECTRODE
        # angle = 90
        # print(f"Rotating image by {angle}°")
        img_rotated = rotate(img, angle, reshape=False)
        if show_examples:
            cv2.imshow("Original image", cv2.resize(img, (500, 500)))
            cv2.imshow("Rotated image", cv2.resize(img_rotated, (500, 500)))
            cv2.waitKey(100)
            # plot both images next to each other with matplotlib
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(img_rotated)
            plt.title(f"Rotated by {int(angle)}°")
            if save_examples:
                plt.savefig(f"rotated_img_{int(angle)}.png")
            plt.show()
            plt.plot(voltage)
        electrodes_to_rotate = int(angle / 360 * len(voltage))
        voltage_rotated = np.roll(voltage, shift=electrodes_to_rotate)
        if show_examples:
            plt.plot(voltage_rotated)
            plt.legend(["Original", "Rotated"])
            plt.title(f"Rotated by {int(angle)}°")
            if save_examples:
                plt.savefig(f"rotated_voltage_{int(angle)}.png")
            plt.show()
        img_rotated_list.append(img_rotated)
        voltage_rotated_list.append(voltage_rotated)
    # convert back to tensors
    img_rotated_numpy = np.array(img_rotated_list)
    voltage_rotated_numpy = np.array(voltage_rotated_list)
    train_images_rotated = torch.tensor(img_rotated_numpy).to(device)
    train_voltage_rotated = torch.tensor(voltage_rotated_numpy).to(device)
    return train_images_rotated, train_voltage_rotated


if __name__ == '__main__':
    path = "..//Collected_Data/Data_05_09_40mm_multifreq"

    device = "cpu"

    voltage_data_np = np.load(os.path.join(path, "v1_array.npy"))
    image_data_np = np.load(os.path.join(path, "img_array.npy"))
    # v0 = np.load(os.path.join(path, "v0.npy"))
    # subtract v0 from all voltages
    # voltage_data_np = (voltage_data_np - v0) / v0  # normalized voltage difference

    # reduce the number of images
    image_data_np = image_data_np[:100]
    voltage_data_np = voltage_data_np[:100]

    # Now the model should learn the difference between the voltages and v0 (default state)

    print("Overall data shape: ", voltage_data_np.shape)

    voltage_data_tensor = torch.tensor(voltage_data_np, dtype=torch.float32).to(device)
    image_data_tensor = torch.tensor(image_data_np, dtype=torch.float32).to(device)

    dataset = CustomDataset(voltage_data_tensor, image_data_tensor)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 4: Split the data into train, test, and validation sets
    # Assuming you have 'voltage_data_tensor' and 'image_data_tensor' as your PyTorch tensors
    # Note: Adjust the test_size and validation_size according to your preference.
    train_voltage, val_voltage, train_images, val_images = train_test_split(
        voltage_data_tensor, image_data_tensor, test_size=0.2, random_state=42)

    val_voltage, test_voltage, val_images, test_images = train_test_split(
        val_voltage, val_images, test_size=0.2, random_state=42)

    train_voltage = train_voltage[:1]
    train_images = train_images[:1]
    # train_voltage, train_images = add_noise_augmentation(train_voltage, train_images,
    #                                                      10, 0.05,
    #                                                      show_examples=True, save_examples=True)

    train_voltage, train_images = add_rotation_augmentation(train_voltage, train_images,
                                                            1, show_examples=True, save_examples=False)
