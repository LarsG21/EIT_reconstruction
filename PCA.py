import os
import pickle

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_pickle("Collected_Data/Combined_dataset_multi_augmented/combined.pkl")
v0 = np.load("Collected_Data/Combined_dataset/v0.npy")

MULTI_FREQUENCY_EIT = True

def reduce_voltages_with_pca(df: pd.DataFrame, save_path: str, n_components=1024):
    """
    Reduces the voltages with pca.
    :param df:
    :return:
    """
    df = df[df["voltages"].apply(lambda x: len(x) == 20480)]
    voltages = df["voltages"].to_list()
    # print len of the voltages in the list
    # for v in voltages:
    #     print(len(v))
    voltages_array = np.array(voltages)
    if not MULTI_FREQUENCY_EIT:
        voltages_array = (voltages_array - v0) / v0  # normalized voltage difference
        voltages_array = voltages_array - np.mean(voltages_array)
    # do pca on voltages
    pca = PCA(n_components=n_components)
    pca.fit(voltages_array)
    voltages_pca = pca.transform(voltages_array)
    # save pca model
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(os.path.join(os.path.dirname(save_path), "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    # plot the main components
    import matplotlib.pyplot as plt
    plt.scatter(voltages_pca[:, 0], voltages_pca[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    # get variance of pc1 and pc2 ...
    for i in range(10):
        print(f"Variance of PC{i}: {np.var(voltages_pca[:, i])}")
    # copy df
    df_pca = df.copy()
    df_pca["voltages"] = voltages_pca.tolist()
    print(df_pca.head())
    # save df
    df_pca.to_pickle(save_path)


n_components = 128

reduce_voltages_with_pca(df=df,
                         save_path=f"Collected_Data/Combined_dataset_multi_augmented/PCA_REDUCED{n_components}/combined_pca.pkl",
                         n_components=n_components)
