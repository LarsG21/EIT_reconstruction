import os
import pickle

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


# df = pd.read_pickle("Collected_Data/Combined_dataset_multi_augmented/augmented_data.pkl")
# v0 = np.load("Collected_Data/Combined_dataset/v0.npy")
# #
# MULTI_FREQUENCY_EIT = True
#
# def reduce_voltages_with_pca(df: pd.DataFrame, save_path: str, n_components=1024):
#     """
#     Reduces the voltages with pca.
#     :param df:
#     :return:
#     """
#     df = df[df["voltages"].apply(lambda x: len(x) == 20480)]
#     voltages = df["voltages"].to_list()
#     # print len of the voltages in the list
#     # for v in voltages:
#     #     print(len(v))
#     voltages_array = np.array(voltages)
#     if not MULTI_FREQUENCY_EIT:
#         voltages_array = (voltages_array - v0) / v0  # normalized voltage difference
#         voltages_array = voltages_array - np.mean(voltages_array)
#     # do pca on voltages
#     pca = PCA(n_components=n_components)
#     pca.fit(voltages_array)
#     voltages_pca = pca.transform(voltages_array)
#     # save pca model
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     with open(os.path.join(os.path.dirname(save_path), "pca.pkl"), "wb") as f:
#         pickle.dump(pca, f)
#
#     # plot the main components
#     import matplotlib.pyplot as plt
#     plt.scatter(voltages_pca[:, 0], voltages_pca[:, 1])
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.show()
#     # get variance of pc1 and pc2 ...
#     for i in range(10):
#         print(f"Variance of PC{i}: {np.var(voltages_pca[:, i])}")
#     # copy df
#     df_pca = df.copy()
#     df_pca["voltages"] = voltages_pca.tolist()
#     print(df_pca.head())
#     # save df
#     df_pca.to_pickle(save_path)


# n_components = 128
#
# reduce_voltages_with_pca(df=df,
#                          save_path=f"Collected_Data/Combined_dataset_multi_augmented/PCA_REDUCED{n_components}/combined_pca.pkl",
#                          n_components=n_components)

def get_main_parts_of_principal_components(pca):
    """

    :param pca:
    :return:
    """
    components = pca.components_
    print("Hauptkomponenten:")
    print(components)
    # increase contrast
    minval = np.percentile(components, 2)
    maxval = np.percentile(components, 98)
    pixvals = np.clip(components, minval, maxval)
    components = ((pixvals) / (maxval - minval)) * 255

    # plot the main components as image
    import matplotlib.pyplot as plt
    plt.imshow(components, cmap="viridis")
    plt.ylabel("Principal Component")
    plt.xlabel("Raw Index")
    plt.colorbar()
    plt.savefig('PCA.png', dpi=500)
    plt.show()
    for i in range(10):
        print(f"Variance of PC{i}: {np.var(components[i])}")
        plt.plot(components[i])
        plt.show()


model_pca_path = "Collectad_Data_Experiments/How_many_frequencies_are_needet_for_abolute_EIT/10_Frequencies/Models/LinearModelWithDropout/run_1_with_augmentation_pca_reduced/model_2023-09-21_13-02-00_200_epochs.pth"
# get the pca.okl in the same folder as the model
pca_path = os.path.join(os.path.dirname(model_pca_path), "pca.pkl")
pca = pickle.load(open(pca_path, "rb"))
get_main_parts_of_principal_components(pca)
