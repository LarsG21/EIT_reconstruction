import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Evaluation.Evaluate_Correct_position import plot_amplitude_response, plot_position_error

df = pd.read_pickle(
    "C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\evaluation_2023-09-08_12-27-41.pkl")

# remove outliers from df in amplitude_response and position_error > or < 3 std
df = df[np.abs(df["amplitude_response"] - df["amplitude_response"].mean()) <= (3 * df["amplitude_response"].std())]
df = df[np.abs(df["position_error"] - df["position_error"].mean()) <= (3 * df["position_error"].std())]

# remove constant offset from position_error

errors = df["error_vector"].apply(lambda x: np.array(x))

# find mean error_vector
mean = errors.mean()
print(mean)
# subtract mean from error_vector
df["error_vector"] = df["error_vector"].apply(lambda x: np.array(x) - mean)

plot_amplitude_response(df)
plot_position_error(df)

# convert col error_vector to np.array

# scatter plot of error_vector
plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]))
plt.show()
