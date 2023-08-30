import numpy as np
import torch
from scipy.ndimage import rotate


def add_noise_augmentation(train_voltage, train_images, number_of_augmentations=1, noise_level=0.1, device="cpu"):
    """"
    Adds noise to the training data and augments the data
    :param train_voltage: The training voltages
    :param train_images: The training images
    :param number_of_augmentations: How many copies of the training data should be created
    :param noise_level: The noise level in % of the standard deviation of the data
    :return: The augmented training voltages and images
    """
    # Step 4.1 Adding noise to the training data in % of max value
    std_of_data = torch.std(train_voltage).item()
    noise_amplitude = noise_level * std_of_data
    print("Absolute max value: ", std_of_data)
    train_voltages_combined = train_voltage
    train_images_combined = train_images
    for i in range(number_of_augmentations):
        train_voltage_augmented = train_voltage + noise_amplitude * torch.randn(train_voltage.shape).to(device)
        train_voltages_combined = torch.cat((train_voltages_combined, train_voltage_augmented), dim=0)
        train_images_combined = torch.cat((train_images_combined, train_images), dim=0)
    return train_voltages_combined, train_images_combined


def add_rotation_augmentation(train_voltage, train_images, number_of_augmentations=1):
    """
    Rotates the training data and augments the data
    :param train_voltage:
    :param train_images:
    :param number_of_augmentations:
    :return:
    """
    print("Rotating the training data")
    # convert tensors to numpy arrays
    train_voltage_numpy = train_voltage.cpu().numpy()
    train_images_numpy = train_images.cpu().numpy()
    # rotate all images by 90° using rotate function scipy
    train_voltage_rotated_combined = train_voltage
    train_images_rotated_combined = train_images
    for i in range(number_of_augmentations):
        train_images_rotated, train_voltage_rotated = generate_rotation_augmentation(train_images_numpy,
                                                                                     train_voltage_numpy)
        train_voltage_rotated_combined = np.concatenate((train_voltage_rotated_combined, train_voltage_rotated), axis=0)
        train_images_rotated_combined = np.concatenate((train_images_rotated_combined, train_images_rotated), axis=0)
    # convert back to tensors

    return train_voltage_rotated_combined, train_images_rotated_combined


def generate_rotation_augmentation(train_images_numpy, train_voltage_numpy, device="cpu"):
    img_rotated_list = []
    voltage_rotated_list = []
    DEGREES_PER_ELECTRODE = 11.25
    for img, voltage in zip(train_images_numpy, train_voltage_numpy):
        # use 11.25° steps for the rotation
        angle = np.random.randint(0, 32) * DEGREES_PER_ELECTRODE
        # print(f"Rotating image by {angle}°")
        img_rotated = rotate(img, angle, reshape=False)
        # cv2.imshow("Original image", cv2.resize(img, (500, 500)))
        # cv2.imshow("Rotated image", cv2.resize(img_rotated, (500, 500)))
        # cv2.waitKey(100)
        # plt.plot(voltage)
        electrodes_to_rotate = int(angle / DEGREES_PER_ELECTRODE)
        electrodes_to_rotate = electrodes_to_rotate * 32  # because of the 32 measurements per electrode
        voltage_rotated = np.roll(voltage, shift=electrodes_to_rotate)
        # plt.plot(voltage_rotated)
        # plt.legend(["Original", "Rotated"])
        # plt.show()
        img_rotated_list.append(img_rotated)
        voltage_rotated_list.append(voltage_rotated)
    # convert back to tensors
    img_rotated_numpy = np.array(img_rotated_list)
    voltage_rotated_numpy = np.array(voltage_rotated_list)
    train_images_rotated = torch.tensor(img_rotated_numpy).to(device)
    train_voltage_rotated = torch.tensor(voltage_rotated_numpy).to(device)
    return train_images_rotated, train_voltage_rotated
