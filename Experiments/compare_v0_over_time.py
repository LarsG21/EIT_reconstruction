import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

v0_1 = pd.read_pickle("../Collected_Data/Single_freq_Data/Data_25_08_negative_samples/combined.pkl")
v0_2 = pd.read_pickle("../Collected_Data/Single_freq_Data/Data_05_09_negative_samples/combined.pkl")
v0_3 = pd.read_pickle("../Collected_Data/V0_SAMPLES_13_10_2023/combined.pkl")
v0_4 = pd.read_pickle("../Collected_Data/V0_SAMPLES_19_10_2023/Data_measured2023-10-19 11_33_54.pkl")


def get_std_of_v0(v0):
    voltages = []
    for i, row in v0.iterrows():
        voltages.append(row["voltages"])
    v0_array = np.array(voltages)
    std_of_v0_voltages = v0_array.std(axis=0)
    return std_of_v0_voltages


std_of_v0_1 = get_std_of_v0(v0_1)
std_of_v0_2 = get_std_of_v0(v0_2)
std_of_v0_3 = get_std_of_v0(v0_3)
std_of_v0_4 = get_std_of_v0(v0_4)

plt.plot(std_of_v0_1, label="std_v0_25_08")
plt.plot(std_of_v0_2, label="std_v0_05_09")
plt.plot(std_of_v0_3, label="std_v0_13_10")
plt.plot(std_of_v0_4, label="std_v0_19_10")
plt.legend()
plt.title("Standard deviation of v0 over time")
plt.show()

print("Mean standard deviation of v0_1: ", std_of_v0_1.mean())
print("Mean standard deviation of v0_2: ", std_of_v0_2.mean())
print("Mean standard deviation of v0_3: ", std_of_v0_3.mean())
print("Mean standard deviation of v0_4: ", std_of_v0_4.mean())

print(len(v0_1))

# take average over "voltages"
v0_1_mean = v0_1["voltages"].mean()
v0_2_mean = v0_2["voltages"].mean()
v0_3_mean = v0_3["voltages"].mean()
v0_4_mean = v0_4["voltages"].mean()

# plot the mean voltages
plt.plot(v0_1_mean, label="v0_25_08")
plt.plot(v0_2_mean, label="v0_05_09")
plt.plot(v0_3_mean, label="v0_13_10")
plt.plot(v0_4_mean, label="v0_19_10")
plt.legend()
plt.show()

# subtract the overall mean

v0_1_mean = v0_1_mean - v0_1_mean.mean()
v0_2_mean = v0_2_mean - v0_2_mean.mean()
v0_3_mean = v0_3_mean - v0_3_mean.mean()
v0_4_mean = v0_4_mean - v0_4_mean.mean()

# plot the mean voltages
plt.plot(v0_1_mean, label="v0_25_08")
plt.plot(v0_2_mean, label="v0_05_09")
plt.plot(v0_3_mean, label="v0_13_10")
plt.plot(v0_4_mean, label="v0_19_10")
plt.legend()
plt.show()

# calculate the differences
v0_1_v0_2 = v0_1_mean - v0_2_mean
v0_1_v0_3 = v0_1_mean - v0_3_mean
v0_1_v0_4 = v0_1_mean - v0_4_mean

plt.plot(v0_1_v0_2, label="v0_1 - v0_2")
plt.legend()
plt.show()
plt.plot(v0_1_v0_3, label="v0_1 - v0_3")
plt.legend()
plt.show()
plt.plot(v0_1_v0_4, label="v0_1 - v0_4")
plt.legend()
plt.show()

# calculate the standard deviation of the differences
print("Standard deviation of v0_1 - v0_2: ", v0_1_v0_2.std())
print("Standard deviation of v0_1 - v0_3: ", v0_1_v0_3.std())
print("Standard deviation of v0_1 - v0_4: ", v0_1_v0_4.std())
