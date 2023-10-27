import numpy as np
import pandas as pd

VOLTAGE_VECTOR_LENGTH = 1024
OUT_SIZE = 64


def generate_noise_figure_data():
    voltages = []
    images = []
    for v in range(VOLTAGE_VECTOR_LENGTH):
        voltage_vector = np.zeros(VOLTAGE_VECTOR_LENGTH)
        voltage_vector[v] = 0.001
        voltages.append(voltage_vector)
        images.append(np.zeros([OUT_SIZE, OUT_SIZE]))
    # save as df
    df = pd.DataFrame()
    df["voltages"] = voltages
    df["images"] = images
    df.to_pickle("noise_figure_data.pkl")


if __name__ == '__main__':
    generate_noise_figure_data()
