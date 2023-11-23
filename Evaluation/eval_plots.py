import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib import pyplot as plt
from plotly import express as px
from scipy import ndimage
from scipy.interpolate import griddata
import seaborn as sns

RESOLUTION_PLOT = 100
nr_of_blurs = 1


def plot_amplitude_response(df: pd.DataFrame, save_path: str = None):
    df["x"] = [x[0] for x in df["positions"]]
    df["y"] = [x[1] for x in df["positions"]]

    # create col radius from the center
    df["radius"] = np.sqrt((df["x"] - 32) ** 2 + (df["y"] - 32) ** 2)
    # round radius
    df["radius"] = df["radius"].apply(lambda x: round(x, 2))
    # get average amplitude response for each radius
    df_ar_per_radius = df.groupby("radius").mean().reset_index()
    print(df_ar_per_radius)
    # plot AR over radius with seaborn
    # rolling average over 5 points
    df_ar_per_radius["amplitude_response"] = df_ar_per_radius["amplitude_response"].rolling(20).mean()
    sns.lineplot(data=df_ar_per_radius, x="radius", y="amplitude_response")
    plt.xlabel("Radius [px]")
    plt.ylabel("Amplitude response")
    plt.title("Amplitude response over radius")
    plt.show()

    # scatter plot with matplotlib and viridis color coding
    plt.scatter(df["x"], df["y"], c=df["amplitude_response"], cmap="viridis")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.title("Amplitude response over space")
    plt.colorbar()
    # save as tikz
    tikzplotlib.save("amplitude_response_over_space.tex")
    plt.show()







    # interpolate between points
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["amplitude_response"].to_numpy()
    # plot scatter 2d with color coding

    # Create a grid of points for interpolation
    xi, yi = np.linspace(x.min(), x.max(), RESOLUTION_PLOT), np.linspace(y.min(), y.max(), RESOLUTION_PLOT)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the data using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi = np.abs(zi)
    # clip zi to 0.0 - 1.0
    zi = np.clip(zi, 0.0, 1.0)

    # Create a heatmap with color coding
    for i in range(nr_of_blurs):
        zi = ndimage.gaussian_filter(zi, sigma=1, radius=1)
    fig = px.imshow(zi, x=xi[0, :], y=yi[:, 0], color_continuous_scale='Viridis')
    fig.update_layout(title='Amplitude response over space', xaxis_title="x [px]", yaxis_title="y [px]")
    # text to colorbar
    fig.update_layout(coloraxis_colorbar=dict(
        title="Amplitude response",
        thicknessmode="pixels", thickness=50,
        lenmode="pixels",
        yanchor="top", y=1,
        ticks="outside", ticksuffix="",
        dtick=0.1
    ))
    if save_path is not None:
        fig.write_image(save_path)
    # Show the plot
    fig.show()


def plot_position_error(df: pd.DataFrame, save_path: str = None):
    df["x"] = [x[0] for x in df["positions"]]
    df["y"] = [x[1] for x in df["positions"]]

    # create col radius from the center
    df["radius"] = np.sqrt((df["x"] - 32) ** 2 + (df["y"] - 32) ** 2)
    # round radius
    df["radius"] = df["radius"].apply(lambda x: round(x, 2))
    # get average amplitude response for each radius
    df_ar_per_radius = df.groupby("radius").mean().reset_index()
    print(df_ar_per_radius)
    # plot AR over radius with seaborn
    # rolling average over 5 points
    df_ar_per_radius["position_error"] = df_ar_per_radius["position_error"].rolling(20).mean()
    sns.lineplot(data=df_ar_per_radius, x="radius", y="position_error")
    plt.xlabel("Radius [px]")
    plt.ylabel("PE")
    plt.title("Position error over radius")
    plt.show()

    # interpolate between points
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["position_error"].to_numpy()
    # interpolate
    # Create a grid of points for interpolation
    xi, yi = np.linspace(x.min(), x.max(), RESOLUTION_PLOT), np.linspace(y.min(), y.max(), RESOLUTION_PLOT)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate the data using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    # get absolute value of position error
    zi = np.abs(zi)
    # Create a heatmap with color coding
    for i in range(nr_of_blurs):
        zi = ndimage.gaussian_filter(zi, sigma=1, radius=1)
    fig = px.imshow(zi, x=xi[0, :], y=yi[:, 0], color_continuous_scale='Viridis')
    # title
    fig.update_layout(title="Position error over space", xaxis_title="x [px]", yaxis_title="y [px]")
    # text to colorbar
    fig.update_layout(coloraxis_colorbar=dict(
        title="Position error",
        thicknessmode="pixels", thickness=50,
        lenmode="pixels",
        yanchor="top", y=1,
        ticks="outside", ticksuffix="",
        dtick=1
    ))
    if save_path is not None:
        fig.write_image(save_path)
    # Show the plot
    fig.show()


def plot_shape_deformation(df: pd.DataFrame, save_path: str = None):
    # remove rows with nan
    df = df.dropna()
    df["x"] = [x[0] for x in df["positions"]]
    df["y"] = [x[1] for x in df["positions"]]
    # interpolate between points
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["shape_deformation"].to_numpy()
    # interpolate
    # Create a grid of points for interpolation
    xi, yi = np.linspace(x.min(), x.max(), RESOLUTION_PLOT), np.linspace(y.min(), y.max(), RESOLUTION_PLOT)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate the data using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    # get absolute value of position error
    zi = np.abs(zi)
    # clip zi to 0.0 - 1.0
    zi = np.clip(zi, 0.0, 1.0)
    # Create a heatmap with color coding
    for i in range(nr_of_blurs):
        zi = ndimage.gaussian_filter(zi, sigma=1, radius=1)
    fig = px.imshow(zi, x=xi[0, :], y=yi[:, 0], color_continuous_scale='Viridis')
    # title
    fig.update_layout(title="Shape deformation over space", xaxis_title="x [px]", yaxis_title="y [px]")
    # text to colorbar
    fig.update_layout(coloraxis_colorbar=dict(
        title="Shape deformation",
        thicknessmode="pixels", thickness=50,
        lenmode="pixels",
        yanchor="top", y=1,
        ticks="outside", ticksuffix="",
        dtick=0.1
    ))
    if save_path is not None:
        fig.write_image(save_path)
    # Show the plot
    fig.show()


def plot_ringing(df, save_path=None):
    # remove rows with nan
    df = df.dropna()
    df["x"] = [x[0] for x in df["positions"]]
    df["y"] = [x[1] for x in df["positions"]]
    # interpolate between points
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["ringing"].to_numpy()
    # interpolate
    # Create a grid of points for interpolation
    xi, yi = np.linspace(x.min(), x.max(), RESOLUTION_PLOT), np.linspace(y.min(), y.max(), RESOLUTION_PLOT)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate the data using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    # get absolute value of position error
    zi = np.abs(zi)
    # Create a heatmap with color coding
    for i in range(nr_of_blurs):
        zi = ndimage.gaussian_filter(zi, sigma=1, radius=1)
    fig = px.imshow(zi, x=xi[0, :], y=yi[:, 0], color_continuous_scale='Viridis')
    # title
    fig.update_layout(title="Ringing over space", xaxis_title="x [mm]", yaxis_title="y [mm]")
    # text to colorbar
    fig.update_layout(coloraxis_colorbar=dict(
        title="Ringing",
        thicknessmode="pixels", thickness=50,
        lenmode="pixels",
        yanchor="top", y=1,
        ticks="outside", ticksuffix="",
        dtick=0.1
    ))
    if save_path is not None:
        fig.write_image(save_path)
    # Show the plot
    fig.show()

# def plot_metric(df: pd.DataFrame, metric_name: str, save_path: str = None):
#     # Remove rows with NaN
#     df = df.dropna()
#     df["x"] = [x[0] for x in df["positions"]]
#     df["y"] = [x[1] for x in df["positions"]]
#
#     # Interpolate between points
#     x = df["x"].to_numpy()
#     y = df["y"].to_numpy()
#     z = df[metric_name].to_numpy()
#
#     # Create a grid of points for interpolation
#     RESOLUTION_PLOT = 100  # Define your resolution value
#     xi, yi = np.linspace(x.min(), x.max(), RESOLUTION_PLOT), np.linspace(y.min(), y.max(), RESOLUTION_PLOT)
#     xi, yi = np.meshgrid(xi, yi)
#
#     # Interpolate the data using griddata
#     zi = griddata((x, y), z, (xi, yi), method='cubic')
#     zi = np.abs(zi)
#
#     # Create a heatmap with color coding
#     nr_of_blurs = 3  # Define the number of blurs
#     for i in range(nr_of_blurs):
#         zi = ndimage.gaussian_filter(zi, sigma=1, radius=1)
#
#     fig = px.imshow(zi, x=xi[0, :], y=yi[:, 0], color_continuous_scale='Viridis')
#
#     # Set plot title and axis labels
#     fig.update_layout(title=f"{metric_name} over space", xaxis_title="x [mm]", yaxis_title="y [mm]")
#
#     # Customize the colorbar
#     fig.update_layout(coloraxis_colorbar=dict(
#         title=metric_name,
#         thicknessmode="pixels", thickness=50,
#         lenmode="pixels",
#         yanchor="top", y=1,
#         ticks="outside", ticksuffix="",
#         dtick=0.1 if metric_name == "Amplitude response" else 1  # Customize dtick for different metrics
#     ))
#
#     if save_path is not None:
#         fig.write_image(save_path)
#
#     # Show the plot
#     fig.show()
