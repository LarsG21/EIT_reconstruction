import pandas as pd

from EarlyStoppingHandler import EarlyStoppingHandler
from Model_Training_with_pca_reduction import trainings_loop
import matplotlib.pyplot as plt

path = "../Collectad_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/3_Frequencies"
num_epochs = 80
learning_rate = 0.001
pca_components = 128
add_augmentation = False
noise_level = 0.05
number_of_noise_augmentations = 2
number_of_rotation_augmentations = 2
weight_decay = 1e-3  # Adjust this value as needed (L2 regularization)
df_complete = pd.DataFrame()
for i in range(1, 3):
    print(f"Run {i}")
    early_stopping_handler = EarlyStoppingHandler(patience=20)
    df_losses, model = trainings_loop(model_name=f"TESTING_{i}", path_to_training_data=path,
                                      num_epochs=num_epochs, learning_rate=learning_rate,
                                      early_stopping_handler=early_stopping_handler,
                                      pca_components=128, add_augmentation=add_augmentation, noise_level=noise_level,
                                      number_of_noise_augmentations=number_of_noise_augmentations,
                                      number_of_rotation_augmentations=number_of_rotation_augmentations,
                                      weight_decay=weight_decay, normalize=True,
                                      )
    print(df_losses)
    # rename the columns
    df_losses = df_losses.rename(columns={"loss": f"loss_{i}", "val_loss": f"val_loss_{i}"})
    # add the dataframe to the side aof the complete dataframe as new columns
    df_complete = pd.concat([df_complete, df_losses], axis=1)
    print(df_complete.shape)

# save the complete dataframe
df_complete.to_pickle("df_complete_no_normalization.pkl")
# get the mean of the complete dataframe
# select the columns with the loss values
df_complete_train_loss = df_complete.filter(regex="loss")
df_complete_train_loss["mean"] = df_complete_train_loss.mean(axis=1)
df_complete_train_loss["std"] = df_complete_train_loss.std(axis=1)
print(df_complete_train_loss)
# select the columns with the val_loss values
df_complete_val_loss = df_complete.filter(regex="val_loss")
df_complete_val_loss["mean"] = df_complete_val_loss.mean(axis=1)
df_complete_val_loss["std"] = df_complete_val_loss.std(axis=1)
print(df_complete_val_loss)

# plot the mean and the std
plt.plot(df_complete_train_loss["mean"])
plt.fill_between(df_complete_train_loss.index, df_complete_train_loss["mean"] - df_complete_train_loss["std"],
                 df_complete_train_loss["mean"] + df_complete_train_loss["std"], alpha=0.2)

plt.plot(df_complete_val_loss["mean"])
plt.fill_between(df_complete_val_loss.index, df_complete_val_loss["mean"] - df_complete_val_loss["std"],
                 df_complete_val_loss["mean"] + df_complete_val_loss["std"], alpha=0.2)
plt.legend(["train_loss", "std_train_loss", "val_loss", "std_val_loss"])

plt.show()
