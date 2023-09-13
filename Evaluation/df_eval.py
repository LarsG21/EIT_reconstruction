import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Evaluation.eval_plots import plot_amplitude_response, plot_position_error

df = pd.read_pickle(
    "C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\evaluation_2023-09-08_13-08-06.pkl")

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

print("Number of samples", len(df))

plot_amplitude_response(df,
                        save_path="C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\amplitude_response.png")
plot_position_error(df,
                    save_path="C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\position_error.png")

# convert col error_vector to np.array

# scatter plot of error_vector with number at each point represented as a color
# plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]), c=df["position_error"])
# plt.colorbar()
plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]))

plt.xlabel("x error [mm]")
plt.ylabel("y error [mm]")
plt.title("Error vector")
# save plot
plt.savefig("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\error_vector.png")
plt.show()
