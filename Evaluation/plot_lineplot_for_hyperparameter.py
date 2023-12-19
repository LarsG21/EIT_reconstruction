import pandas as pd
from matplotlib import pyplot as plt

from tiktzplot_utils import genterate_linepot_with_std
import seaborn as sns

test_set_path = "../Model_Training/df_eval_New_Differnt_PCAS4.pkl"

df_evaluation_results = pd.read_pickle(test_set_path)
df_evaluation_results = df_evaluation_results.reset_index(drop=True)

# load v0 from the same folder as the test set
print(f"INFO: Loaded test set from {test_set_path} with {len(df_evaluation_results)} samples")


def plot_metrics_with_std(df_eval, metric_names):
    num_metrics = len(metric_names)
    for i in range(num_metrics):
        plt.figure()
        sns.lineplot(data=df_eval, x=x_param, y=metric_names[i])
        plt.xlabel("Weight decay")
        # plt.xscale('log')
        plt.ylabel(metric_names[i])
        plt.show()
        # delete figure
        plt.clf()


metric_names_df = ["ar", "sd", "ringing", "pe", "pc"]

x_param = "x"
plot_metrics_with_std(df_evaluation_results, metric_names_df)

# seperate in differnt dataframes
df_ar = df_evaluation_results[[x_param, "ar"]]
df_sd = df_evaluation_results[[x_param, "sd"]]
df_ringing = df_evaluation_results[[x_param, "ringing"]]
df_pe = df_evaluation_results[[x_param, "pe"]]
df_pc = df_evaluation_results[[x_param, "pc"]]

# create an df with mean and std for each metric for each wd
df_ar = df_ar.groupby(x_param).agg(["mean", "std"]).reset_index()
df_sd = df_sd.groupby(x_param).agg(["mean", "std"]).reset_index()
df_ringing = df_ringing.groupby(x_param).agg(["mean", "std"]).reset_index()
df_pe = df_pe.groupby(x_param).agg(["mean", "std"]).reset_index()
df_pc = df_pc.groupby(x_param).agg(["mean", "std"]).reset_index()

# rename columns to mean and std
df_ar.columns = [x_param, "mean", "std"]
df_sd.columns = [x_param, "mean", "std"]
df_ringing.columns = [x_param, "mean", "std"]
df_pe.columns = [x_param, "mean", "std"]
df_pc.columns = [x_param, "mean", "std"]

# plot all metrics in one plot

df_list = [
    # df_ar,
    # df_sd,
    # df_ringing,
    df_pe,
    # df_pc
]

# # normalize all metrics between 0 and 1
# for df in df_list:
#     df["mean"] = (df["mean"] - df["mean"].min()) / (df["mean"].max() - df["mean"].min())
#     df["std"] = (df["std"] - df["std"].min()) / (df["std"].max() - df["std"].min())


labels = [
    # "Amplitude response", "Std Amplitude response"
    # "Shape Deformation", "Std Shape Deformation"
    # "Ringing", "Std Ringing",
    "Position Error", "Std Position Error",
    # "Pearson Correlation", "Std Pearson Correlation"
]

colors = ["blue", "orange", "green", "red", "purple"]

colors = [colors[i] for i in range(len(df_list))]
metric = "pe"

genterate_linepot_with_std(f"{metric}_over_{x_param}_multifreq.tex", df_list, colors, labels,
                           x_label="Number of Blurrs", y_label="A.U.", title="Metrics over Learning Rate",
                           x_ticks=df_ar[x_param].to_list(), log_scale=False)
