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




def generate_boxplot(file_name, boxplot_dict_list, labels, colors):
    with open(file_name, "w") as f:
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")
        f.write("boxplot/draw direction = y,\n")
        f.write("enlarge y limits,\n")
        f.write("ymajorgrids,\n")
        f.write("grid style = {dashed, gray!50}, % Add grid style\n")
        f.write("xtick = {1, 2, 3, 4},\n")
        f.write("xticklabel style = {align=center, font=\\samll},\n")
        f.write("xticklabels = {")
        for i, label in enumerate(labels):
            f.write(f"{label}")
            if i != len(labels) - 1:
                f.write(",")
        f.write("},\n")
        f.write("xtick style = {draw=none},\n")
        f.write("ylabel = {Relative metric},\n")
        f.write("]\n")
        for i, boxplot_dict in enumerate(boxplot_dict_list):
            f.write("\\addplot+[boxplot prepared={\n")
            for key, value in boxplot_dict.items():
                f.write(f"{key}={value}, \n")
            f.write(f"}},fill = {colors[i]}!20, draw = black] coordinates {{}};\n")
        f.write("\\end{axis}\n")
        f.write("\\end{tikzpicture}\n")


if __name__ == '__main__':
    # test boxplot
    boxplot_dict_ar = {
        "lower whisker": 0.2,
        "lower quartile": 0.4,
        "median": 1,
        "upper quartile": 1.2,
        "upper whisker": 1.5
    }
    boxplot_dict_sd = {
        "lower whisker": 1,
        "lower quartile": 1.5,
        "median": 2,
        "upper quartile": 2.3,
        "upper whisker": 2.7
    }
    boxplot_dict_ringing = {
        "lower whisker": 0.1,
        "lower quartile": 0.5,
        "median": 0.7,
        "upper quartile": 1.4,
        "upper whisker": 1.9
    }
    boxplot_dict_list = [boxplot_dict_ar, boxplot_dict_sd, boxplot_dict_ringing]
    labels = ["AR", "SD", "Ringing"]
    colors = ["cyan", "orange", "green"]
    generate_boxplot("boxplot.tex", boxplot_dict_list, labels, colors)
