import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_pickle("Collected_Data/Even_Orientation_Dataset/samples_negative_orientation_6.pkl")

voltages = df1["voltages"].to_numpy()
print(voltages.shape)

v0 = voltages.mean(axis=0)
print(v0.shape)
plt.plot(v0)
plt.show()
save = np.save("C:\\Users\\lgudjons\\Desktop\\v0.npy", v0)
