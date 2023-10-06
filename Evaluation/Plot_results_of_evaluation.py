import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Evaluation.eval_plots import plot_amplitude_response, plot_position_error, plot_shape_deformation

df = pd.read_pickle(
    "Results/model_2023-10-06_12-15-26_epoche_143_of_300_best_model/evaluation_model_model_2023-10-06_12-15-26_epoche_143_of_300_best_model.pkl")

# remove outliers from df in amplitude_response and position_error > or < N std
N = 3
border_amplitude_response = N * df["amplitude_response"].std()
border_position_error = N * df["position_error"].std()
border_shape_deformation = N * df["shape_deformation"].std()
print(border_amplitude_response)
print(border_position_error)
print(border_shape_deformation)
print("Number of samples", len(df))
df = df[np.abs(df["amplitude_response"] - df["amplitude_response"].mean()) <= border_amplitude_response]
df = df[np.abs(df["position_error"] - df["position_error"].mean()) <= border_position_error]
df = df[np.abs(df["shape_deformation"] - df["shape_deformation"].mean()) <= border_shape_deformation]
# replace outliers with mean
# df["amplitude_response"] = df["amplitude_response"].apply(lambda x: x if np.abs(x - df["amplitude_response"].mean()) <= (1 * df["amplitude_response"].std()) else df["amplitude_response"].mean())
# df["position_error"] = df["position_error"].apply(lambda x: x if np.abs(x - df["position_error"].mean()) <= (1 * df["position_error"].std()) else df["position_error"].mean())
# df["shape_deformation"] = df["shape_deformation"].apply(lambda x: x if np.abs(x - df["shape_deformation"].mean()) <= (1 * df["shape_deformation"].std()) else df["shape_deformation"].mean())

print("Number of samples after outlier removal", len(df))
# remove constant offset from position_error

errors = df["error_vector"].apply(lambda x: np.array(x))

# find mean error_vector
mean = errors.mean()
print(mean)
# subtract mean from error_vector
df["error_vector"] = df["error_vector"].apply(lambda x: np.array(x) - mean)

print("Number of samples", len(df))

plot_amplitude_response(df,
                        # save_path="C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\amplitude_response_evaluation_model_model_2023-09-22_13-48-51_epoche_395_of_400_best_model.png"
                        )
plot_position_error(df,
                    # save_path="C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\position_error_evaluation_model_model_2023-09-22_13-48-51_epoche_395_of_400_best_model.png"
                    )

plot_shape_deformation(df,
                       # save_path="C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\shape_deformation_evaluation_model_model_2023-09-22_13-48-51_epoche_395_of_400_best_model.png"
                       )

# convert col error_vector to np.array

# scatter plot of error_vector with number at each point represented as a color
# plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]), c=df["position_error"])
# plt.colorbar()
plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]))

plt.xlabel("x error [mm]")
plt.ylabel("y error [mm]")
plt.title("Error vector")
# save plot
# plt.savefig("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\error_vector.png")
plt.show()