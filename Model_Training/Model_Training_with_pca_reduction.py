import copy
import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from CustomDataset import CustomDataset
from Model_Training.dimensionality_reduction import perform_pca_on_input_data
from data_augmentation import add_noise_augmentation, add_rotation_augmentation, add_gaussian_blur
from Models import LinearModelWithDropout, LinearModelWithDropout2, LinearModel, LinearModel2, \
    LinearModelWithDropoutAndBatchNorm
from model_plot_utils import plot_sample_reconstructions, plot_loss, infer_single_reconstruction, \
    plot_loss_and_sample_reconstruction, plot_difference_for_some_sample_reconstruction_images
from utils import add_normalizations

from EarlyStoppingHandler import EarlyStoppingHandler

LOSS_SCALE_FACTOR = 1000
# VOLTAGE_VECTOR_LENGTH = 6144
VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64

# How to use Cuda gtx 1070: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

if torch.cuda.is_available():

    print("Torch is using CUDA")
    print("Cuda device count", torch.cuda.device_count())
    print("Cuda device name", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    if device == "cuda:0":
        print("Using CUDA")
        torch.cuda.set_device(0)  # or 1,2,3

    print(torch.cuda.current_device())
else:
    print("Using CPU")
    device = "cpu"


# torch.cuda.set_device(0)
# device = "cpu"


def evaluate_model_and_save_results(model, criterion, test_dataloader, train_dataloader, val_dataloader, save_path):
    """
    Evaluates the model and saves the results
    :param model: The model to evaluate
    :param criterion: The loss function
    :param test_dataloader:
    :param train_dataloader:
    :param val_dataloader:
    :return:
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_loss = 0.0
        for batch_voltages, batch_images in test_dataloader:
            outputs = model(batch_voltages)
            test_loss += criterion(outputs, batch_images.view(-1, OUT_SIZE ** 2)).item() * LOSS_SCALE_FACTOR

        test_loss /= len(test_dataloader)

        # do the same for the train and validation set
        train_loss = 0.0
        for batch_voltages, batch_images in train_dataloader:
            # batch_voltages = batch_voltages.view(-1, 1, VOLTAGE_VECTOR_LENGTH)  # Reshape the voltages vor CNNs
            outputs = model(batch_voltages)
            train_loss += criterion(outputs, batch_images.view(-1, OUT_SIZE ** 2)).item() * LOSS_SCALE_FACTOR

        train_loss /= len(train_dataloader)

        val_loss = 0.0
        for batch_voltages, batch_images in val_dataloader:
            outputs = model(batch_voltages)
            val_loss += criterion(outputs, batch_images.view(-1, OUT_SIZE ** 2)).item() * LOSS_SCALE_FACTOR

        val_loss /= len(val_dataloader)

        print(f"Test Loss: {round(test_loss, 4)}")
        print(f"Train Loss: {round(train_loss, 4)}")
        print(f"Val Loss: {round(val_loss, 4)}")
        # save in txt file
        with open(os.path.join(save_path, "test_loss.txt"), "w") as f:
            f.write(f"Test Loss: {round(test_loss, 4)}\n")
            f.write(f"Train Loss: {round(train_loss, 4)}\n")
            f.write(f"Val Loss: {round(val_loss, 4)}\n")


def trainings_loop(model_name: str, path_to_training_data: str, learning_rate: float, num_epochs: int,
                   early_stopping_handler: EarlyStoppingHandler, loading_path: str = "",
                   pca_components: int = 0, add_augmentation: bool = False, noise_level: float = 0.05,
                   number_of_noise_augmentations: int = 2, number_of_rotation_augmentations: int = 0,
                   weight_decay: float = 1e-3, normalize=True, electrode_level_normalization=False,
                   ):
    global VOLTAGE_VECTOR_LENGTH
    ABSOLUTE_EIT = True
    SAMPLE_RECONSTRUCTION_INDEX = 1  # Change this to see different sample reconstructions
    SAVE_CHECKPOINTS = False
    LOSS_PLOT_INTERVAL = 10

    ######################################################################################
    if pca_components > 0:
        VOLTAGE_VECTOR_LENGTH = pca_components
    # model = LinearModelWithDropoutAndBatchNorm(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2).to(device)
    model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2).to(device)
    #################################
    path = path_to_training_data
    #################################
    if "multi" in path.lower() and not ABSOLUTE_EIT:
        raise Exception("Are you trying to train a single frequency model on a multi frequency dataset?")
    # if not any(x in path.lower() for x in ["multi", "abolute"]) and ABSOLUTE_EIT:
    #     raise Exception("Are you trying to train a multi frequency model on a single frequency dataset?")
    if not ABSOLUTE_EIT and normalize:
        raise Exception("Relative EIT and Normalization didnt work well")

    model_class_name = model.__class__.__name__
    model_path = os.path.join(path, "Models", model_class_name, model_name)
    print(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        print("Creating model directory")
        os.makedirs(model_path)
    else:
        input("Model directory already exists. Press any key if you want to overwrite...")

    # Save settings in txt file
    with open(os.path.join(model_path, "settings.txt"), "w") as f:
        f.write(f"Model: {model_class_name}\n")
        f.write(f"Absolute EIT: {ABSOLUTE_EIT}\n")
        f.write(f"NOISE_LEVEL: {noise_level}\n")
        f.write(f"LEARNING_RATE: {learning_rate}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"patience: {early_stopping_handler.patience}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"Augmentations: {add_augmentation}\n")
        f.write(f"Number of augmentations: {number_of_noise_augmentations}\n")
        f.write(f"Number of rotation augmentations: {number_of_rotation_augmentations}\n")
        f.write(f"PCA_COMPONENTS: {pca_components}\n")
        f.write(f"normalize: {normalize}\n")
        f.write(f"electrode_level_normalization: {electrode_level_normalization}\n")
        f.write("\n")
    USE_DIFF_DIRECTLY = False
    if os.path.exists(os.path.join(path, "v1_array.npy")):
        voltage_data_np = np.load(os.path.join(path, "v1_array.npy"))
        print("INFO: Using v1 voltages and calculating voltage differences with one v0 as reference")
    else:
        try:
            voltage_data_np = np.load(os.path.join(path, "voltage_diff_array.npy"))
            print("INFO: Using voltage differences directly")
            USE_DIFF_DIRECTLY = True
        except FileNotFoundError:
            raise Exception("No voltage data found")
    image_data_np = np.load(os.path.join(path, "img_array.npy"))

    if OUT_SIZE == 128:
        # double the resolution of the images
        image_data_np = np.repeat(image_data_np, 2, axis=1)
        image_data_np = np.repeat(image_data_np, 2, axis=2)

    # reduce the number of images
    # image_data_np = image_data_np[:800]
    # voltage_data_np = voltage_data_np[:800]

    # Highlight Step 1: In case of time difference EIT, we need to normalize the data with v0
    if not ABSOLUTE_EIT:
        if not USE_DIFF_DIRECTLY:
            print("INFO: Single frequency EIT data is used. Normalizing the data with v0")
            v0 = np.load(os.path.join(path, "v0.npy"))
            # v0 = np.load("../ScioSpec_EIT_Device/v0.npy")
            # normalize the voltage data
            voltage_data_np = (voltage_data_np - v0) / v0  # normalized voltage difference
            # Now the model should learn the difference between the voltages and v0 (default state)
        else:
            print("INFO: Single frequency EIT data is used. Using voltage differences directly")

    # Highlight Step 2: Preprocess the data (independent if it is absolute or difference EIT)
    voltage_data_np = add_normalizations(v1=voltage_data_np, NORMALIZE_MEDIAN=normalize,
                                         NORMALIZE_PER_ELECTRODE=electrode_level_normalization)

    print("Overall data shape: ", voltage_data_np.shape)

    voltage_data_tensor = torch.tensor(voltage_data_np, dtype=torch.float32).to(device)
    image_data_tensor = torch.tensor(image_data_np, dtype=torch.float32).to(device)

    # dataset = CustomDataset(voltage_data_tensor, image_data_tensor)
    # dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Highlight Step 3: Save the model summary
    print("model summary: ", model)
    # print number of trainable parameters
    nr_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", nr_trainable_params)
    # write model summary to txt file
    with open(os.path.join(model_path, "model_summary.txt"), "w") as f:
        f.write(str(model))
        f.write("\n")
        f.write(f"Number of trainable parameters: {nr_trainable_params}\n")
        f.write("\n")

    # Highlight Step 4: Split the data into train, test, and validation sets
    print("INFO: Splitting data into train, validation and test sets")
    train_voltage, val_voltage, train_images, val_images = train_test_split(
        voltage_data_tensor, image_data_tensor, test_size=0.2, random_state=42)

    val_voltage, test_voltage, val_images, test_images = train_test_split(
        val_voltage, val_images, test_size=0.2, random_state=42)

    # Highlight Step 4.1: Augment the training data
    if add_augmentation:
        # augment the training data
        train_voltage, train_images = add_noise_augmentation(train_voltage, train_images,
                                                             number_of_noise_augmentations, noise_level, device=device)
        train_voltage, train_images = add_rotation_augmentation(train_voltage, train_images,
                                                                number_of_rotation_augmentations, device=device)
        # blurr the images with a gaussian filter
        train_images = add_gaussian_blur(train_images, device=device, nr_of_blurs=number_of_blur_augmentations)

    train_voltage_original = train_voltage.clone()
    test_voltage_original = test_voltage.clone()
    # Highlight Step4.2 Do PCA to reduce the number of input features
    if pca_components > 0:
        print("INFO: Performing PCA on input data")
        train_voltage, val_voltage, test_voltage, pca = perform_pca_on_input_data(voltage_data_tensor, train_voltage,
                                                                                  val_voltage, test_voltage, model_path,
                                                                                  device,
                                                                                  n_components=pca_components,
                                                                                  debug=False,
                                                                                  train_images=train_images)

    # Highlight Step 5: Create the DataLoader for train, test, and validation sets
    train_dataset = CustomDataset(train_voltage, train_images)
    train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    # number of training samples
    print("Number of training samples: ", len(train_dataset))

    val_dataset = CustomDataset(val_voltage, val_images)
    val_dataloader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    print("Number of validation samples: ", len(val_dataset))

    test_dataset = CustomDataset(test_voltage, test_images)
    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Number of test samples: ", len(test_dataset))

    # save number of samples in txt file
    with open(os.path.join(model_path, "settings.txt"), "a") as f:
        f.write(f"Number of training samples: {len(train_dataset)}\n")
        f.write(f"Number of validation samples: {len(val_dataset)}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n")

    # Highlight Step 6: Define the loss function and optimizer
    criterion = nn.MSELoss()

    # Initialize the optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # # add a scheduler to reduce the learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # loss_black_img = calc_average_loss_completly_black(image_data_tensor=image_data_tensor,
    #                                                    criterion=criterion)
    #
    # loss_white_img = calc_average_loss_completly_white(image_data_tensor=image_data_tensor,
    #                                                    criterion=criterion)

    # Highlight Step 7: Define the training loop
    if loading_path != "":
        model.load_state_dict(torch.load(
            os.path.join(model_path, "model_2023-09-28_16-06-34_299_300.pth")))
    loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        loop = tqdm(train_dataloader)
        for batch_voltages, batch_images in loop:
            # Forward pass
            # reshape the voltages to be [32, 1, INPUT_SIZE]
            # batch_voltages = batch_voltages.view(-1, 1, VOLTAGE_VECTOR_LENGTH)  # Reshape the voltages vor CNNs
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
                # batch_voltages = batch_voltages.view(-1, 1, VOLTAGE_VECTOR_LENGTH)  # Reshape the voltages vor CNNs
                outputs = model(batch_voltages)
                val_loss += criterion(outputs, batch_images.view(-1, OUT_SIZE ** 2)).item() * LOSS_SCALE_FACTOR

            val_loss /= len(val_dataloader)
            stop = early_stopping_handler.handle_early_stopping(model, val_loss, epoch, num_epochs, model_path)
            if stop:
                model = early_stopping_handler.get_best_model()
                break

            val_loss_list.append(val_loss)
            print(
                f"\nEpoch [{epoch + 1}/{num_epochs}], Val Loss: {round(val_loss, 4)} Training Loss: {round(loss.item(), 4)}")

        loss_list.append(loss.item())
        # plot loss and sample reconstruction every N epochs
        plot_loss_and_sample_reconstruction(
            epoch,
            LOSS_PLOT_INTERVAL,
            model,
            loss_list,
            val_loss_list,
            test_voltage,
            test_images,
            model_path,
            num_epochs,
            SAMPLE_RECONSTRUCTION_INDEX,
            SAVE_CHECKPOINTS
        )

        loop.set_postfix(loss=loss.item())
    # save the final model
    if loading_path != "":
        save_path = os.path.join(model_path,
                                 f"continued_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}_{num_epochs}.pth")
    else:
        save_path = os.path.join(model_path,
                                 f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}_{num_epochs}.pth")
    torch.save(model.state_dict(), save_path)
    # put loss lists into a dataframe and save it
    df = pd.DataFrame({"loss": loss_list, "val_loss": val_loss_list})
    # round the values
    df = df.round(4)
    df.to_csv(os.path.join(model_path, "losses.csv"))
    # plot the final loss
    plot_loss(val_loss_list=val_loss_list, loss_list=loss_list, save_name=os.path.join(model_path, "loss_plot.png"))

    # Highlight Step 8: Evaluate the model on the test set
    evaluate_model_and_save_results(model=model, criterion=criterion, test_dataloader=test_dataloader,
                                    train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                                    save_path=model_path)
    PLOT_EXAMPLES = True
    if PLOT_EXAMPLES:
        plot_sample_reconstructions(test_images, test_voltage, model, criterion, num_images=20,
                                    save_path=model_path)

    # plot_difference_for_some_sample_reconstruction_images(test_images, test_voltage, model, num_images=20)

    single_datapoint = voltage_data_np[0]
    voltage_data_tensor = torch.tensor(single_datapoint, dtype=torch.float32)
    infer_single_reconstruction(model=model, voltage_data=voltage_data_tensor)
    return df, model


if __name__ == "__main__":
    model_name = "Test_Run_less_neg_normalized_rot_aug"
    # path = "../Training_Data/1_Freq_with_individual_v0s"
    path = "../Trainings_Data_EIT32/3_Freq"
    # path = "../Collected_Data_Variation_Experiments/High_Variation_multi"
    # path = "../Collected_Data/Combined_dataset"
    # path = "../Collected_Data/Training_set_circular_08_11_3_freq_40mm"
    num_epochs = 100
    learning_rate = 0.001
    pca_components = 128
    add_augmentation = True
    noise_level = 0.05
    number_of_noise_augmentations = 4
    number_of_rotation_augmentations = 0
    number_of_blur_augmentations = 5
    weight_decay = 1e-5  # Adjust this value as needed (L2 regularization)

    early_stopping_handler = EarlyStoppingHandler(patience=20)
    trainings_loop(model_name=model_name, path_to_training_data=path,
                   num_epochs=num_epochs, learning_rate=learning_rate, early_stopping_handler=early_stopping_handler,
                   pca_components=pca_components, add_augmentation=add_augmentation, noise_level=noise_level,
                   number_of_noise_augmentations=number_of_noise_augmentations,
                   number_of_rotation_augmentations=number_of_rotation_augmentations,
                   weight_decay=weight_decay, normalize=True, electrode_level_normalization=False,
                   )
