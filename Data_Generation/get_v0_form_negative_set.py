import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_pickle("../Trainings_Data_EIT32/1_Freq_More_Orientations/samples_negative_orientation_3.pkl")

voltages = df1["voltages"].to_numpy()
print(voltages.shape)

v0 = voltages.mean(axis=0)

# device in 3 equal parts
# length = len(v0)
# freq1 = v0[0:int(length / 3)]
# freq2 = v0[int(length / 3):int(2 * length / 3)]
# freq3 = v0[int(2 * length / 3):length]
#
# # get mean of each frequency
# mean1 = freq1.mean(axis=0)
# mean2 = freq2.mean(axis=0)
# mean3 = freq3.mean(axis=0)
#
# print(mean1, mean2, mean3)
#
# # plot all 3 frequencies in one plot
# plt.plot(freq1)
# plt.plot(freq2)
# plt.plot(freq3)
# plt.legend(["freq1", "freq2", "freq3"])
# plt.show()
#
# # plot the difference between the frequencies
# plt.plot(freq1 - freq2)
# plt.plot(freq2 - freq3)
# plt.plot(freq3 - freq1)
# plt.legend(["freq1 - freq2", "freq2 - freq3", "freq3 - freq1"])
# plt.show()

print(v0.shape)
plt.plot(v0)
plt.show()
save = np.save("C:\\Users\\lgudjons\\Desktop\\v0.npy", v0)
