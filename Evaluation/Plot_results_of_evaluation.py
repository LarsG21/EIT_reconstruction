import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Evaluation.eval_plots import plot_amplitude_response, plot_position_error, plot_shape_deformation

# Path to a pickle file containing the evaluation results created by Evaluate_Test_Set_Dataframe.py
df = pd.read_pickle(
    "Results/evaluation_model_model_2023-10-23_13-50-11_122_150.pkl")
# df = pd.read_pickle(
#     "Results/evaluation_regressor_KNeighborsRegressor.pkl")

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
                        save_path="Results/amplitude_response.png"
                        )
plot_position_error(df,
                    save_path="Results/position_error.png"
                    )

plot_shape_deformation(df,
                       save_path="Results/shape_deformation.png"
                       )

# convert col error_vector to np.array

# scatter plot of error_vector with number at each point represented as a color
# plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]), c=df["position_error"])
# plt.colorbar()
plt.scatter(df["error_vector"].apply(lambda x: x[0]), df["error_vector"].apply(lambda x: x[1]))

plt.xlabel("x error [px]")
plt.ylabel("y error [px]")
plt.title("Error vector")
# save plot
# plt.savefig("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\error_vector.png")
plt.show()

# average amplitude response, position error and shape deformation
avg_ar = df["amplitude_response"].mean()
avg_pe = df["position_error"].mean()
avg_sd = df["shape_deformation"].mean()

# std of amplitude response, position error and shape deformation
std_ar = df["amplitude_response"].std()
std_pe = df["position_error"].std()
std_sd = df["shape_deformation"].std()

results_dict = {"avg_ar": avg_ar,
                "avg_pe": avg_pe,
                "avg_sd": avg_sd,
                "std_ar": std_ar,
                "std_pe": std_pe,
                "std_sd": std_sd}

print(f"Results: \n {results_dict}")

# in one plot
plt.boxplot([df["amplitude_response"], df["shape_deformation"]], labels=["amplitude_response", "shape_deformation"])
plt.title("Boxplot of amplitude_response and shape_deformation")
plt.ylabel("Amplitude response/Shape deformation")
plt.savefig(
    "C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\boxplot_amplitude_response_shape_deformation.png")
plt.show()

plt.boxplot([df["position_error"]], labels=["position_error"])
plt.title("Boxplot of position_error")
plt.ylabel("Error [px]")
plt.savefig("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\boxplot_position_error.png")
plt.show()

# save results dict in json file
# with open("C:\\Users\\lgudjons\\PycharmProjects\\EIT_reconstruction\\Evaluation\\Results\\results_dict.json", "w") as f:
#     json.dump(results_dict, f)
