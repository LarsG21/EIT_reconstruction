import pandas as pd


def genterate_linepot_with_std(file_name, df_list: [pd.DataFrame], colors, labels):
    """
    Generates a lineplot with the given dataframes and saves it as a tikz file
    :param file_name:
    :param df_list:
    :param colors:
    :param labels:
    :return:
    """
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]
    with open(file_name, "w") as f:
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")
        f.write(f"xmin=-1, xmax={len(df_list[0])},\n")
        f.write(f"ymin={df_list[1]['mean'].min() - 2}, ymax={df_list[0]['mean'].max() + 5},\n")
        f.write("ymajorgrids=true,\n")
        f.write("grid style=dashed,\n")
        f.write("legend pos=north east,\n")
        f.write("width=0.7\\textwidth,\n")
        f.write("height=0.5\\textwidth,\n")
        # x label
        f.write("xlabel={Epochs},\n")
        # y label
        f.write("ylabel={Loss},\n")
        # title
        f.write("title={Loss plot training and validation},\n")
        f.write("]\n")
        for j, dataframe in enumerate(df_list):
            f.write(f"\\addplot[color={colors[j]}] coordinates \n")
            f.write("{\n")
            for i in range(len(dataframe["mean"])):
                f.write(f"({i},{dataframe['mean'][i]})")
            f.write("\n};\n")
            f.write(
                f"\\addplot[name path=us_top,color={colors[j]}!20, forget plot, opacity=0.2] coordinates\n")  # forget plot is needed to not show the fill between in the legend
            f.write("{\n")
            for i in range(len(dataframe["mean"])):
                f.write(f"({i},{dataframe['mean'][i] + dataframe['std'][i]})")
            f.write("\n};\n")
            f.write(f"\\addplot[name path=us_down,color={colors[j]}!20, forget plot, opacity=0.2] coordinates\n")
            f.write("{\n")
            for i in range(len(dataframe["mean"])):
                f.write(f"({i},{dataframe['mean'][i] - dataframe['std'][i]})")
            f.write("\n};\n")
            f.write(f"\\addplot[color={colors[j]}!70, opacity=0.2] fill between[of=us_top and us_down];\n")
        # write legend
        f.write("\\legend{")
        for i, label in enumerate(labels):
            f.write(f"{label}")
            if i != len(labels) - 1:
                f.write(",")
        f.write("}\n")
        f.write("\end{axis}\n")
        f.write("\end{tikzpicture}\n")
