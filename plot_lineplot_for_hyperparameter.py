import pandas as pd
from matplotlib import pyplot as plt

from tiktzplot_utils import genterate_linepot_with_std
import seaborn as sns

test_set_path = "Model_Training/df_eval_New.pkl"

df_evaluation_results = pd.read_pickle(test_set_path)
# load v0 from the same folder as the test set
print(f"INFO: Loaded test set from {test_set_path} with {len(df_evaluation_results)} samples")


def plot_metrics_with_std(df_eval, metric_names):
    num_metrics = len(metric_names)
    for i in range(num_metrics):
        plt.figure()
        sns.lineplot(data=df_eval, x="wd", y=metric_names[i])
        plt.xlabel("Weight decay")
        plt.xscale('log')
        plt.ylabel(metric_names[i])
        plt.show()
        # delete figure
        plt.clf()


metric_names_df = ["ar", "sd", "ringing", "pe", "pc"]

# plot_metrics_with_std(df_evaluation_results, metric_names_df)

# seperate in differnt dataframes
df_ar = df_evaluation_results[["wd", "ar"]]
df_sd = df_evaluation_results[["wd", "sd"]]
df_ringing = df_evaluation_results[["wd", "ringing"]]
df_pe = df_evaluation_results[["wd", "pe"]]
df_pc = df_evaluation_results[["wd", "pc"]]

# create an df with mean and std for each metric for each wd
df_ar = df_ar.groupby("wd").agg(["mean", "std"]).reset_index()
df_sd = df_sd.groupby("wd").agg(["mean", "std"]).reset_index()
df_ringing = df_ringing.groupby("wd").agg(["mean", "std"]).reset_index()
df_pe = df_pe.groupby("wd").agg(["mean", "std"]).reset_index()
df_pc = df_pc.groupby("wd").agg(["mean", "std"]).reset_index()

# rename columns to mean and std
df_ar.columns = ["wd", "mean", "std"]
df_sd.columns = ["wd", "mean", "std"]
df_ringing.columns = ["wd", "mean", "std"]
df_pe.columns = ["wd", "mean", "std"]
df_pc.columns = ["wd", "mean", "std"]

# plot all metrics in one plot

df_list = [df_ar,
           # df_sd, df_ringing,
           # df_pe,
           df_pc
           ]
labels = ["Amplitude response", "Std Amplitude response"
# "Shape Deformation",  "Std Shape Deformation"
# "Ringing", "Std Ringing",
# # "Position Error", "Std Position Error",
                                "Pearson Correlation", "Std Pearson Correlation"
          ]

colors = ["blue", "orange"]

genterate_linepot_with_std("parameter_over_wd.txt", df_list, colors, labels,
                           x_label="Weight decay", y_label="Loss", title="Loss plot training and validation",
                           x_ticks=df_ar["wd"].to_list(), log_scale=True)
