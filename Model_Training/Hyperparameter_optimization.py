import pandas as pd
import torch
from torch import nn

from EarlyStoppingHandler import EarlyStoppingHandler
from Model_Training.CustomDataset import get_test_data_loader
from Model_Training.Models import LinearModelWithDropout2
from Model_Training.model_plot_utils import plot_sample_reconstructions
from Model_Training_with_pca_reduction_copy import trainings_loop

LOSS_SCALE_FACTOR = 1000

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

training_data_path = "../Training_Data/3_Freq"
test_data_path = "../Test_Data/Test_Set_Circular_16_10_3_freq"

num_epochs = 20
learning_rate = 0.001
pca_components = 128
add_augmentation = False
noise_level = 0.05
number_of_noise_augmentations = 2
number_of_rotation_augmentations = 2
weight_decay = 1e-3  # Adjust this value as needed (L2 regularization)
VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64

dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

df_complete = pd.DataFrame(columns=["dropout_rate", "loss", "model"])

criterion = nn.MSELoss()

for dropout_rate in dropout_rates:
    print(f"Run {dropout_rate}")
    if pca_components > 0:
        VOLTAGE_VECTOR_LENGTH = pca_components
    model = LinearModelWithDropout2(input_size=VOLTAGE_VECTOR_LENGTH, output_size=OUT_SIZE ** 2,
                                    dropout_prob=dropout_rate).to(device)
    early_stopping_handler = EarlyStoppingHandler(patience=20)
    df_losses, model, pca = trainings_loop(model=model, model_name=f"DROPOUT_{dropout_rate}",
                                           path_to_training_data=training_data_path,
                                           num_epochs=num_epochs, learning_rate=learning_rate,
                                           early_stopping_handler=early_stopping_handler,
                                           pca_components=128, add_augmentation=add_augmentation,
                                           noise_level=noise_level,
                                           number_of_noise_augmentations=number_of_noise_augmentations,
                                           number_of_rotation_augmentations=number_of_rotation_augmentations,
                                           weight_decay=weight_decay, normalize=True,
                                           )
    test_dataloader, test_images, test_voltage = get_test_data_loader(test_data_path=test_data_path,
                                                                      device=device, pca=pca)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_loss = 0.0
        for batch_voltages, batch_images in test_dataloader:
            # convert with pca if needed
            outputs = model(batch_voltages)
            test_loss += criterion(outputs, batch_images.view(-1, OUT_SIZE ** 2)).item() * LOSS_SCALE_FACTOR

        test_loss /= len(test_dataloader)
        print(f"Test loss: {test_loss} on Test Set {test_data_path}")
    plot_sample_reconstructions(test_images, test_voltage, model, criterion, num_images=10)
    # add the final loss to the complete dataframe
    to_add = {"dropout_rate": dropout_rate, "loss": df_losses.iloc[-1]["loss"],
              "test_loss": test_loss, "model": model}
    df_complete = pd.concat([df_complete, pd.DataFrame(to_add, index=[0])], ignore_index=True)

print(df_complete)
df_complete.to_pickle("dropout_comparison.pkl")
