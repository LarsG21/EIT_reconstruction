import os
import time

import pandas as pd
import cv2
import numpy as np
import tikzplotlib
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split

import torch.utils.data as data

from Model_Training.CustomDataset import CustomDataset


def add_noise_augmentation(train_voltage: torch.Tensor,
                           train_images: torch.Tensor,
                           number_of_augmentations, noise_level,
                           device="cpu",
                           show_examples=False,
                           save_examples=False):
    """"
    Adds noise to the training data and augments the data
    :param train_voltage: The training voltages (PyTorch tensor or NumPy array)
    :param train_images: The training images (PyTorch tensor or NumPy array)
    :param number_of_augmentations: How many copies of the training data should be created
    :param noise_level: The noise level in % of the standard deviation of the data
    :param save_examples: Whether to save example plots
    :param show_examples: Whether to show example plots
    :return: The augmented training voltages and images
    """
    convert_to_numpy = False
    if number_of_augmentations == 0:
        return train_voltage, train_images
    print("INFO: Adding noise to the training data")
    # Check if the input is a NumPy array and convert it to a PyTorch tensor if needed
    if isinstance(train_voltage, np.ndarray):
        train_voltage = torch.from_numpy(train_voltage).to(device)
        convert_to_numpy = True
    if isinstance(train_images, np.ndarray):
        train_images = torch.from_numpy(train_images).to(device)
        convert_to_numpy = True

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
            # plt.title("Example " + str(i))
            plt.title("Input Voltage Vector")
            plt.xlabel("Measurement Index")
            plt.ylabel("Voltage")
            plt.plot(train_voltages_combined[i, :].cpu().numpy())
            plt.savefig("Example_noise_augment" + str(i) + ".png")
    if convert_to_numpy:
        # transfer back to numpy
        train_voltages_combined = train_voltages_combined.cpu().numpy()
        train_images_combined = train_images_combined.cpu().numpy()

    return train_voltages_combined, train_images_combined


def add_rotation_augmentation(train_voltage: torch.Tensor,
                              train_images: torch.Tensor,
                              number_of_augmentations=1,
                              show_examples=False,
                              save_examples=False,
                              device="cpu"):
    """
    Rotates the training data and augments the data
    :param device:
    :param save_examples:
    :param show_examples:
    :param train_voltage:
    :param train_images:
    :param number_of_augmentations:
    :return:
    """
    convert_back_to_tensor = False
    if number_of_augmentations == 0:
        return train_voltage, train_images
    print("INFO: Rotating the training data")
    if type(train_voltage) == torch.Tensor and type(train_images) == torch.Tensor:
        train_voltage_numpy = train_voltage.cpu().numpy()
        train_images_numpy = train_images.cpu().numpy()
        convert_back_to_tensor = True
    else:
        train_voltage_numpy = train_voltage
        train_images_numpy = train_images
    train_voltage_rotated_combined = train_voltage
    train_images_rotated_combined = train_images
    if type(train_voltage) == torch.Tensor and type(train_images) == torch.Tensor:
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
    if convert_back_to_tensor:
        # transfer back to gpu
        train_voltage_rotated_combined = torch.from_numpy(train_voltage_rotated_combined).to(device)
        train_images_rotated_combined = torch.from_numpy(train_images_rotated_combined).to(device)
    return train_voltage_rotated_combined, train_images_rotated_combined


def generate_rotation_augmentation(train_images_numpy, train_voltage_numpy, device="cpu", show_examples=False,
                                   save_examples=False):
    img_rotated_list = []
    voltage_rotated_list = []
    DEGREES_PER_ELECTRODE = 11.25
    plt.rcParams.update({'font.size': 12})
    # set font to charter
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Charter'] + plt.rcParams['font.serif']
    for img, voltage in zip(train_images_numpy, train_voltage_numpy):
        # use 11.25° steps for the rotation
        angle = np.random.randint(0, 32) * DEGREES_PER_ELECTRODE
        # angle = np.random.randint(1, 4) * 90
        # print(f"Rotating image by {angle}°")
        img_rotated = rotate(img, angle, reshape=False)
        # smaller than 0.001 set to 0
        img_rotated[img_rotated < 0.001] = 0
        # bigger than 0.8 set to 1
        img_rotated[img_rotated > 0.8] = 1
        if show_examples:  # PLOT_THESIS
            cv2.imshow("Original image", cv2.resize(img, (500, 500)))
            cv2.imshow("Rotated image", cv2.resize(img_rotated, (500, 500)))
            cv2.waitKey(100)
            # plot both images next to each other with matplotlib
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original")
            plt.subplot(1, 2, 2)
            plt.imshow(img_rotated)
            plt.title(f"Rotated by {int(angle)}°")
            # thigh layout
            plt.tight_layout()
            if save_examples:
                plt.savefig(f"rotated_img_{int(angle)}.png")
                plt.savefig(f"rotated_img_{int(angle)}.pdf")  # PLOT_THESIS
            plt.show()
            plt.plot(voltage)
            # put vertical lines at the electrodes
            NUMBER_OF_FREQUENCIES = 3
            if len(voltage) == 1024:
                AMPLITUDE_OR_COMPLEX = 1
            elif len(voltage) >= 1024:
                AMPLITUDE_OR_COMPLEX = 2
            else:
                raise Exception("Voltage length is not correct")
            # for i in range(0, len(voltage),
            #                32 * NUMBER_OF_FREQUENCIES * AMPLITUDE_OR_COMPLEX):  # *2 for real and imaginary part
            #     # plt.axvline(x=i, color="red", linestyle="--", label='_nolegend_')
        electrodes_to_rotate = int(angle / 360 * len(voltage))
        voltage_rotated = np.roll(voltage, shift=electrodes_to_rotate)
        if show_examples:
            plt.plot(voltage_rotated)
            plt.legend(["Original", "Rotated"])
            plt.title(f"Rotated by {int(angle)}°")
            if save_examples:
                tikzplotlib.save(f"rotated_voltage_{int(angle)}.tex")  # PLOT_THESIS
                plt.savefig(f"rotated_voltage_{int(angle)}.png")
                plt.savefig(f"rotated_voltage_{int(angle)}.pdf")  # PLOT_THESIS
            plt.show()
        img_rotated_list.append(img_rotated)
        voltage_rotated_list.append(voltage_rotated)
    # convert back to tensors
    img_rotated_numpy = np.array(img_rotated_list)
    voltage_rotated_numpy = np.array(voltage_rotated_list)
    train_images_rotated = torch.tensor(img_rotated_numpy).to(device)
    train_voltage_rotated = torch.tensor(voltage_rotated_numpy).to(device)
    return train_images_rotated, train_voltage_rotated


def add_gaussian_blur(train_images, device="cpu", nr_of_blurs=1):
    """
    Add gaussian blur to the images.
    :param train_images:
    :param device:
    :return:
    """
    if nr_of_blurs == 0:
        return train_images
    print(f"INFO: Adding {nr_of_blurs} gaussian blurs to the images")
    convert_back_to_tensor = False
    if type(train_images) == torch.Tensor:
        train_images_numpy = train_images.cpu().numpy()
        convert_back_to_tensor = True
    elif type(train_images) == np.ndarray:
        train_images_numpy = train_images
    else:
        raise Exception("Type of train images not recognized")
    train_images_blurred_numpy = []
    for img in train_images_numpy:
        img_blurred = cv2.GaussianBlur(img, (3, 3), 1)
        for i in range(0, nr_of_blurs):
            img_blurred = cv2.GaussianBlur(img_blurred, (3, 3), 1)
        train_images_blurred_numpy.append(img_blurred)
    train_images_blurred_numpy = np.array(train_images_blurred_numpy)
    # show example
    plt.subplot(1, 2, 1)
    plt.imshow(train_images_numpy[0])
    plt.subplot(1, 2, 2)
    plt.imshow(train_images_blurred_numpy[0])
    plt.show()
    if convert_back_to_tensor:
        train_images_blurred = torch.tensor(train_images_blurred_numpy).to(device)
    else:
        train_images_blurred = train_images_blurred_numpy

    return train_images_blurred


# add superposition augmentation
# Combine two images and voltages to one image and voltage by adding them together
def add_superposition_augmentation(train_voltages, train_images, device="cpu", nr_of_superpositions=1,
                                   nr_of_copies=1,
                                   debug=False):
    """
    Adds superposition augmentation to the training data
    :param train_images:
    :param train_voltages:
    :param device:
    :param nr_of_superpositions:
    :return:
    """
    if nr_of_superpositions == 0:
        return train_voltages, train_images
    print(f"INFO: Adding {nr_of_copies} copies with {nr_of_superpositions} superpositions to the training data")
    convert_back_to_tensor = False
    if type(train_images) == torch.Tensor:
        train_images_numpy = train_images.cpu().numpy()
        convert_back_to_tensor = True
    elif type(train_images) == np.ndarray:
        train_images_numpy = train_images
    else:
        raise Exception("Type of train images not recognized")
    if type(train_voltages) == torch.Tensor:
        train_voltages_numpy = train_voltages.cpu().numpy()
        convert_back_to_tensor = True
    elif type(train_voltages) == np.ndarray:
        train_voltages_numpy = train_voltages
    else:
        raise Exception("Type of train voltages not recognized")
    train_images_superposed_numpy = []
    train_voltages_superposed_numpy = []
    for _ in range(0, nr_of_copies):
        for img, voltage in zip(train_images_numpy, train_voltages_numpy):
            img_superposed = img
            voltage_superposed = voltage
            nr_of_superpositions_temp = np.random.randint(1, nr_of_superpositions + 1)
            for i in range(0, nr_of_superpositions_temp):
                # select a random image and voltage
                random_index = np.random.randint(0, len(train_images_numpy))
                img_to_add = train_images_numpy[random_index]
                voltage_to_add = train_voltages_numpy[random_index]
                # add them together
                img_superposed = img_superposed + img_to_add
                voltage_superposed = (voltage_superposed + voltage_to_add) / 2
                # normalize the image
                # img_superposed = img_superposed / np.max(img_superposed)
                # # normalize the voltage
                # voltage_superposed = voltage_superposed / np.max(voltage_superposed) * max(np.max(voltage),
                #                                                                            np.max(voltage_to_add))
                if debug:
                    plt.subplot(1, 2, 1)
                    plt.plot(voltage)
                    plt.subplot(1, 2, 2)
                    plt.plot(voltage_to_add)
                    plt.show()
                    # show the image and voltage
                    plt.subplot(1, 2, 1)
                    plt.imshow(img_superposed)
                    plt.subplot(1, 2, 2)
                    plt.plot(voltage_superposed)
                    plt.show()
                    THESIS_PLOT = True
                    if THESIS_PLOT:  # PLOT_THESIS
                        generate_thesis_plots(img, img_superposed, img_to_add, voltage, voltage_superposed,
                                              voltage_to_add)
                    print("OK")

            # add to the list
            train_images_superposed_numpy.append(img_superposed)
            train_voltages_superposed_numpy.append(voltage_superposed)
    # show one example
    plt.subplot(1, 2, 1)
    plt.imshow(train_images_numpy[0])
    plt.subplot(1, 2, 2)
    plt.imshow(train_images_superposed_numpy[0])
    plt.title("Superposed image")
    plt.show()
    # show the voltage
    plt.subplot(1, 2, 1)
    plt.plot(train_voltages_numpy[0])
    plt.subplot(1, 2, 2)
    plt.plot(train_voltages_superposed_numpy[0])
    plt.title("Superposed voltage")
    plt.show()
    train_images_superposed_numpy = np.array(train_images_superposed_numpy)
    train_voltages_superposed_numpy = np.array(train_voltages_superposed_numpy)
    # combine with the original data
    # TODO: Find solution for .cpu().numpy()
    train_voltages_superposed_numpy = np.concatenate((train_voltages, train_voltages_superposed_numpy), axis=0)
    train_images_superposed_numpy = np.concatenate((train_images, train_images_superposed_numpy), axis=0)
    if convert_back_to_tensor:
        train_images_superposed = torch.tensor(train_images_superposed_numpy).to(device)
        train_voltages_superposed = torch.tensor(train_voltages_superposed_numpy).to(device)
    else:
        train_images_superposed = train_images_superposed_numpy
        train_voltages_superposed = train_voltages_superposed_numpy
    return train_voltages_superposed, train_images_superposed


def generate_thesis_plots(img, img_superposed, img_to_add, voltage, voltage_superposed, voltage_to_add):
    plt.rcParams.update({'font.size': 12})
    # set font to charter
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Charter'] + plt.rcParams['font.serif']
    plt.plot(voltage)
    plt.title("V1")
    plt.tight_layout()
    plt.savefig("original_voltage.pdf")
    plt.show()
    plt.plot(voltage_to_add)
    plt.title("V2")
    plt.tight_layout()
    plt.savefig("voltage_to_add.pdf")
    plt.show()
    plt.imshow(img)
    plt.title("Image 1")
    plt.tight_layout()
    plt.savefig("original_image.pdf")
    plt.show()
    plt.imshow(img_to_add)
    plt.title("Image 2")
    plt.tight_layout()
    plt.savefig("image_to_add.pdf")
    plt.show()
    plt.imshow(img_superposed)
    plt.title("Superposed image")
    plt.tight_layout()
    plt.savefig("superposed_image.pdf")
    plt.show()
    plt.plot(voltage_superposed)
    plt.title("Superposed voltage")
    plt.tight_layout()
    plt.savefig("superposed_voltage.pdf")
    plt.show()


if __name__ == '__main__':
    # pass
    # path = "../Collected_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies"
    path = "../Collected_Data/Combined_dataset"
    device = "cpu"

    voltage_data_np = np.load(os.path.join(path, "v1_array.npy"))
    image_data_np = np.load(os.path.join(path, "img_array.npy"))
    print(f"Length of voltage data: {len(voltage_data_np)}")
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
        voltage_data_tensor, image_data_tensor, test_size=0.01, random_state=42)

    # train_voltage = train_voltage[:1]
    # train_images = train_images[:1]
    # train_voltage, train_images = add_noise_augmentation(train_voltage, train_images,
    #                                                      4, 0.04,
    #                                                      show_examples=True, save_examples=False)
    #
    # train_voltage, train_images = add_rotation_augmentation(train_voltage, train_images,
    #                                                         4, show_examples=True, save_examples=True)

    voltages, images = add_superposition_augmentation(train_voltage, train_images,
                                                      device=device, nr_of_superpositions=2,
                                                      nr_of_copies=2,
                                                      debug=True)
    # print("OK")
    # # convert both to numpy
    # train_voltage = train_voltage.cpu().numpy().tolist()
    # train_images = train_images.cpu().numpy().tolist()
    # # save in one df
    # df2 = pd.read_pickle("..//Collected_Data/Combined_dataset_multi/combined.pkl")
    # df = pd.DataFrame(data={"images": train_images, "voltages": train_voltage},
    #                   index=[0] * len(train_voltage)
    #                   )
    #
    # # save as pkl
    # print(f"Lenght of augmented data: {len(df)}")
    # df.to_pickle("..//Collected_Data/Combined_dataset_multi_augmented/augmented_data.pkl")
