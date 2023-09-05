import copy
import os.path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from CustomDataset import CustomDataset
from data_augmentation import add_noise_augmentation, add_rotation_augmentation
from Models import LinearModelWithDropout, LinearModelWithDropout2
from model_plot_utils import plot_sample_reconstructions, plot_loss, infer_single_reconstruction

LOSS_SCALE_FACTOR = 1000
# VOLTAGE_VECTOR_LENGTH = 928
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


if __name__ == "__main__":
    TRAIN = True
    ADD_AUGMENTATION = True
    NUMBER_OF_NOISE_AUGMENTATIONS = 2
    NUMBER_OF_ROTATION_AUGMENTATIONS = 2
    LOADING_PATH = "../Collected_Data/Data_24_08_40mm_target/Models/LinearModelDropout/TESTING/model_2023-08-24_16-01-08_epoche_592_of_1000_best_model.pth"
    load_model_and_continue_trainig = False
    SAVE_CHECKPOINTS = False
    LOSS_PLOT_INTERVAL = 10
    # Training parameters
    num_epochs = 200
    NOISE_LEVEL = 0.08
    # NOISE_LEVEL = 0
    LEARNING_RATE = 0.0003
    # Define the weight decay factor
    weight_decay = 1e-6  # Adjust this value as needed (L2 regularization)
    # weight_decay = 0  # Adjust this value as needed (L2 regularization)
    # Define early stopping parameters
    patience = max(num_epochs * 0.15, 50)  # Number of epochs to wait for improvement

    best_val_loss = float('inf')  # Initialize with a very high value
    counter = 0  # Counter to track epochs without improvement

    # path = "Edinburgh mfEIT Dataset"
    path = "../Collected_Data/Combined_dataset"
    # path = "../Own_Simulation_Dataset/1_anomaly_circle"
    # model_name = "Test_1_noise_regularization1e-6"
    model_name = "05_09_all_data_40mm_target_and_augmentation_more_noise"
    # model_name = f"model{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_path = os.path.join(path, "Models", "LinearModelDropout", model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

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
        f.write("\n")

    # voltage_data_np = np.load("../Own_Simulation_Dataset/1_anomaly_circle/v1_array.npy")
    # image_data_np = np.load("../Own_Simulation_Dataset/1_anomaly_circle/img_array.npy")
    # v0 = np.load("../Own_Simulation_Dataset/1_anomaly_circle/v0.npy")
    voltage_data_np = np.load(os.path.join(path, "v1_array.npy"))
    image_data_np = np.load(os.path.join(path, "img_array.npy"))
    v0 = np.load(os.path.join(path, "v0.npy"))
    # subtract v0 from all voltages
    voltage_data_np = (voltage_data_np - v0) / v0  # normalized voltage difference

    # reduce the number of images
    # image_data_np = image_data_np[:100]
    # voltage_data_np = voltage_data_np[:100]

    # Now the model should learn the difference between the voltages and v0 (default state)

    print("Overall data shape: ", voltage_data_np.shape)

    voltage_data_tensor = torch.tensor(voltage_data_np, dtype=torch.float32).to(device)
    image_data_tensor = torch.tensor(image_data_np, dtype=torch.float32).to(device)

    dataset = CustomDataset(voltage_data_tensor, image_data_tensor)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    # # Step 3: Create the model
    model = LinearModelWithDropout(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2).to(device)
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
    # Assuming you have 'voltage_data_tensor' and 'image_data_tensor' as your PyTorch tensors
    # Note: Adjust the test_size and validation_size according to your preference.
    train_voltage, val_voltage, train_images, val_images = train_test_split(
        voltage_data_tensor, image_data_tensor, test_size=0.2, random_state=42)

    val_voltage, test_voltage, val_images, test_images = train_test_split(
        val_voltage, val_images, test_size=0.2, random_state=42)

    # Step 5: Create the DataLoader for train, test, and validation sets
    if ADD_AUGMENTATION:
        # augment the training data
        train_voltage, train_images = add_noise_augmentation(train_voltage, train_images,
                                                             NUMBER_OF_NOISE_AUGMENTATIONS, NOISE_LEVEL)

        train_voltage, train_images = add_rotation_augmentation(train_voltage, train_images,
                                                                NUMBER_OF_ROTATION_AUGMENTATIONS)
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
    # criterion = nn.L1Loss()  # Didnt work well

    # Initialize the optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    # loss_black_img = calc_average_loss_completly_black(image_data_tensor=image_data_tensor,
    #                                                    criterion=criterion)
    #
    # loss_white_img = calc_average_loss_completly_white(image_data_tensor=image_data_tensor,
    #                                                    criterion=criterion)

    if TRAIN:
        # Step 7: Define the training loop
        if load_model_and_continue_trainig:
            model.load_state_dict(torch.load(
                os.path.join(model_path, "MODEL_NAME.pth")))
        loss_list = []
        val_loss_list = []
        best_model = model
        for epoch in range(num_epochs):
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
                infer_single_reconstruction(model=model, voltage_data=test_voltage_data,
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

