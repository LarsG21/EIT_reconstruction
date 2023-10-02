import copy
import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from CustomDataset import CustomDataset
from Model_Training.dimensionality_reduction import perform_pca_on_input_data
from data_augmentation import add_noise_augmentation, add_rotation_augmentation
from Models import LinearModelWithDropout, LinearModelWithDropout2, LinearModel, LinearModel2, LinearModelWithDropoutAndBatchNorm
from model_plot_utils import plot_sample_reconstructions, plot_loss, infer_single_reconstruction
from utils import preprocess

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


def handle_early_stopping():
    global best_val_loss, counter, best_model, model
    if val_loss < best_val_loss:  # Early stopping
        best_val_loss = val_loss
        counter = 0
        best_model = copy.deepcopy(model)
    else:
        counter += 1
        print(f"Early stopping in {patience - counter} epochs")
        if counter >= patience:
            print("Early stopping triggered. No improvement in validation loss.")
            # save the model
            torch.save(best_model.state_dict(),
                       os.path.join(model_path,
                                    f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_epoche_{epoch}_of_{num_epochs}_best_model.pth"))
            model = best_model  # load the best model
            return True
    return False


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


ABSOLUTE_EIT = False
SAMPLE_RECONSTRUCTION_INDEX = 0  # Change this to see different sample reconstructions

if __name__ == "__main__":
    TRAIN = True
    ADD_AUGMENTATION = True
    NUMBER_OF_NOISE_AUGMENTATIONS = 1
    NUMBER_OF_ROTATION_AUGMENTATIONS = 1
    LOADING_PATH = "../Collectad_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies/Models/LinearModelWithDropout2/train_pca_64/model_2023-09-28_16-06-34_299_300.pth"
    load_model_and_continue_trainig = False
    SAVE_CHECKPOINTS = False
    LOSS_PLOT_INTERVAL = 5
    # Training parameters
    num_epochs = 200
    NOISE_LEVEL = 0.05
    # NOISE_LEVEL = 0
    LEARNING_RATE = 0.001
    # Define the weight decay factor
    weight_decay = 1e-3  # Adjust this value as needed (L2 regularization)
    # weight_decay = 0  # Adjust this value as needed (L2 regularization)
    # Define early stopping parameters
    # patience = max(num_epochs * 0.15, 50)  # Number of epochs to wait for improvement
    patience = 40  # Number of epochs to wait for improvement
    PCA_COMPONENTS = 0  # 0 means no PCA
    ######################################################################################
    if PCA_COMPONENTS > 0:
        VOLTAGE_VECTOR_LENGTH = PCA_COMPONENTS
    best_val_loss = float('inf')  # Initialize with a very high value
    counter = 0  # Counter to track epochs without improvement
    model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2).to(device)
    #################################
    # path = "../Collectad_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies"
    path = "../Collected_Data/Combined_dataset"
    # path = "../Collected_Data/Dataset_40mm_and_60_mm"
    #################################
    if "multi" in path.lower() and not ABSOLUTE_EIT:
        raise Exception("Are you trying to train a single frequency model on a multi frequency dataset?")
    if not any(x in path.lower() for x in ["multi", "abolute"]) and ABSOLUTE_EIT:
        raise Exception("Are you trying to train a multi frequency model on a single frequency dataset?")
    ####################################
    model_name = "run3"
    ####################################
    # model_name = f"model{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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
        f.write(f"NOISE_LEVEL: {NOISE_LEVEL}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"patience: {patience}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"Augmentations: {ADD_AUGMENTATION}\n")
        f.write(f"Number of augmentations: {NUMBER_OF_NOISE_AUGMENTATIONS}\n")
        f.write(f"Number of rotation augmentations: {NUMBER_OF_ROTATION_AUGMENTATIONS}\n")
        f.write(f"PCA_COMPONENTS: {PCA_COMPONENTS}\n")
        f.write("\n")

    voltage_data_np = np.load(os.path.join(path, "v1_array.npy"))
    image_data_np = np.load(os.path.join(path, "img_array.npy"))

    # reduce the number of images
    # image_data_np = image_data_np[:800]
    # voltage_data_np = voltage_data_np[:800]

    # Highlight: In case of PCA Data v0 is already used for normalization
    if not ABSOLUTE_EIT:
        print("INFO: Single frequency EIT data is used. Normalizing the data with v0")
        v0 = np.load(os.path.join(path, "v0.npy"))
        # v0 = np.load("../ScioSpec_EIT_Device/v0.npy")
        # normalize the voltage data
        voltage_data_np = (voltage_data_np - v0) / v0  # normalized voltage difference
        # Now the model should learn the difference between the voltages and v0 (default state)

    # Highlight: In case of PCA Data v0 is already used for normalization
    voltage_data_np = preprocess(v1=voltage_data_np,
                                 SUBTRACT_MEDIAN=True,
                                 DIVIDE_BY_MEDIAN=True)

    print("Overall data shape: ", voltage_data_np.shape)

    voltage_data_tensor = torch.tensor(voltage_data_np, dtype=torch.float32).to(device)
    image_data_tensor = torch.tensor(image_data_np, dtype=torch.float32).to(device)

    dataset = CustomDataset(voltage_data_tensor, image_data_tensor)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 3: Save the model summary
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

    # Step 4: Split the data into train, test, and validation sets
    print("INFO: Splitting data into train, validation and test sets")
    train_voltage, val_voltage, train_images, val_images = train_test_split(
        voltage_data_tensor, image_data_tensor, test_size=0.2, random_state=42)

    val_voltage, test_voltage, val_images, test_images = train_test_split(
        val_voltage, val_images, test_size=0.2, random_state=42)

    # Step 4.1: Augment the training data
    if ADD_AUGMENTATION:
        # augment the training data
        print("INFO: Adding noise augmentation")
        train_voltage, train_images = add_noise_augmentation(train_voltage, train_images,
                                                             NUMBER_OF_NOISE_AUGMENTATIONS, NOISE_LEVEL, device=device)
        print("INFO: Adding rotation augmentation")
        train_voltage, train_images = add_rotation_augmentation(train_voltage, train_images,
                                                                NUMBER_OF_ROTATION_AUGMENTATIONS, device=device)
    train_voltage_original = train_voltage.clone()
    test_voltage_original = test_voltage.clone()
    # Step4.2 Do PCA to reduce the number of features
    if PCA_COMPONENTS > 0:
        print("INFO: Performing PCA on input data")
        train_voltage, val_voltage, test_voltage = perform_pca_on_input_data(voltage_data_tensor, train_voltage,
                                                                             val_voltage, test_voltage, model_path,
                                                                             device,
                                                                             n_components=PCA_COMPONENTS)

    # Step 5: Create the DataLoader for train, test, and validation sets
    train_dataset = CustomDataset(train_voltage, train_images)
    train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    # number of training samples
    print("Number of training samples: ", len(train_dataset))

    val_dataset = CustomDataset(val_voltage, val_images)
    val_dataloader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("Number of validation samples: ", len(val_dataset))

    test_dataset = CustomDataset(test_voltage, test_images)
    test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Number of test samples: ", len(test_dataset))

    # save number of samples in txt file
    with open(os.path.join(model_path, "settings.txt"), "a") as f:
        f.write(f"Number of training samples: {len(train_dataset)}\n")
        f.write(f"Number of validation samples: {len(val_dataset)}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n")

    # Step 6: Define the loss function and optimizer
    criterion = nn.MSELoss()

    # Initialize the optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    # # add a scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # loss_black_img = calc_average_loss_completly_black(image_data_tensor=image_data_tensor,
    #                                                    criterion=criterion)
    #
    # loss_white_img = calc_average_loss_completly_white(image_data_tensor=image_data_tensor,
    #                                                    criterion=criterion)

    if TRAIN:
        # Step 7: Define the training loop
        if load_model_and_continue_trainig:
            model.load_state_dict(torch.load(
                os.path.join(model_path, "model_2023-09-28_16-06-34_299_300.pth")))
        loss_list = []
        val_loss_list = []
        best_model = model
        for epoch in range(num_epochs):
            # reduce learning rate after 50 epochs
            if epoch == 50:
                print("INFO: Reducing learning rate by factor 2")
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 2, weight_decay=weight_decay)
            model.train()  # Set the model to training mode
            for batch_voltages, batch_images in train_dataloader:
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
                stop = handle_early_stopping()  # Early stopping
                if stop:
                    break

                val_loss_list.append(val_loss)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {round(val_loss, 4)} Training Loss: {round(loss.item(), 4)}")

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
                test_voltage_data = test_voltage[SAMPLE_RECONSTRUCTION_INDEX]
                # plot the voltage data
                # test_voltage_data = test_voltage_data.cpu().numpy()
                # plt.plot(test_voltage_original[SAMPLE_RECONSTRUCTION_INDEX].cpu().numpy())
                # plt.title("Voltage data test")
                # plt.show()

                infer_single_reconstruction(model=model, voltage_data=test_voltage_data,
                                            title=f"Reconstruction after {epoch} epochs",
                                            original_image=test_images[SAMPLE_RECONSTRUCTION_INDEX].cpu())
                # train_voltage_data = train_voltage[SAMPLE_RECONSTRUCTION_INDEX]
                # # plot the voltage data
                # train_voltage_data = train_voltage_data.cpu().numpy()
                # plt.plot(train_voltage_original[SAMPLE_RECONSTRUCTION_INDEX].cpu().numpy())
                # plt.title("Voltage data train")
                # plt.show()
                # infer_single_reconstruction(model=model, voltage_data=train_voltage_data,
                #                             title=f"Reconstruction after {epoch} epochs",
                #                             original_image=train_images[SAMPLE_RECONSTRUCTION_INDEX].cpu())
                # plot the corresponding image
        # save the final model
        if load_model_and_continue_trainig:
            save_path = os.path.join(model_path,
                                     f"continued_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}_{num_epochs}.pth")
        else:
            save_path = os.path.join(model_path,
                                     f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}_{num_epochs}.pth")
        torch.save(model.state_dict(), save_path)
        # plot the final loss
        plot_loss(val_loss_list=val_loss_list, loss_list=loss_list, save_name=os.path.join(model_path, "loss_plot.png"))
    # load the model
    else:  # load the model
        print("Loading the model")
        model.load_state_dict(torch.load(LOADING_PATH))
        model.eval()

    # Step 8: Evaluate the model on the test set
    evaluate_model_and_save_results(model=model, criterion=criterion, test_dataloader=test_dataloader,
                                    train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                                    save_path=model_path)

    # Try inference on test images
    SAVE_TEST_IMAGES = True
    if SAVE_TEST_IMAGES:
        save_path = os.path.join(model_path, "test_images")
    else:
        save_path = None
    plot_sample_reconstructions(test_images, test_voltage, model, criterion, num_images=20,
                                save_path=model_path)
    # plot_difference_images(test_images, test_voltage, model, num_images=20)

    # single_datapoint = voltage_data_np[0]
    # voltage_data_tensor = torch.tensor(single_datapoint, dtype=torch.float32)
    # plot_single_reconstruction(model=model, voltage_data=voltage_data_tensor)
