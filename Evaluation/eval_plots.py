import numpy as np
import pandas as pd
from plotly import express as px
from scipy import ndimage
from scipy.interpolate import griddata

RESOLUTION_PLOT = 80
nr_of_blurs = 1


def plot_amplitude_response(df: pd.DataFrame, save_path: str = None):
    df["x"] = [x[0] for x in df["positions"]]
    df["y"] = [x[1] for x in df["positions"]]
    # fig = px.scatter(df, x="x", y="y", color="amplitude_response")
    # fig.show()
    # interpolate between points
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["amplitude_response"].to_numpy()
    # plot scatter 2d with color coding
    # fig = px.scatter(df, x="x", y="y", color="amplitude_response")
    # # show
    # fig.show()
    # interpolate
    # Create a grid of points for interpolation
    xi, yi = np.linspace(x.min(), x.max(), RESOLUTION_PLOT), np.linspace(y.min(), y.max(), RESOLUTION_PLOT)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the data using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi = np.abs(zi)

    # Create a heatmap with color coding
    for i in range(nr_of_blurs):
        zi = ndimage.gaussian_filter(zi, sigma=1, radius=1)
    fig = px.imshow(zi, x=xi[0, :], y=yi[:, 0], color_continuous_scale='Viridis')
    fig.update_layout(title='Amplitude response over space', xaxis_title="x [mm]", yaxis_title="y [mm]")
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
    # Create a new DataFrame for interpolated data
    interpolated_df = pd.DataFrame({'x': xi.flatten(), 'y': yi.flatten(), 'amplitude_response': zi.flatten()})
    # Create a scatter plot with interpolated data
    # fig = px.scatter(interpolated_df, x="x", y="y", color="amplitude_response")
    # fig.update_layout(title='Amplitude response over space', xaxis_title="x", yaxis_title="y")
    # # Show the plot
    # fig.show()


def plot_position_error(df: pd.DataFrame, save_path: str = None):
    df["x"] = [x[0] for x in df["positions"]]
    df["y"] = [x[1] for x in df["positions"]]
    # fig = px.scatter(df, x="x", y="y", color="position_error")
    # fig.show()
    # interpolate between points
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["position_error"].to_numpy()
    # fig = px.scatter(df, x="x", y="y", color="position_error")
    # # show
    # fig.show()
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
    fig.update_layout(title="Position error over space", xaxis_title="x [mm]", yaxis_title="y [mm]")
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
    # Create a new DataFrame for interpolated data
    interpolated_df = pd.DataFrame({'x': xi.flatten(), 'y': yi.flatten(), 'position_error': zi.flatten()})
    # Create a scatter plot with interpolated data
    # fig = px.scatter(interpolated_df, x="x", y="y", color="position_error")
    # # title
    # fig.update_layout(title="Position error over space", xaxis_title="x", yaxis_title="y")
    # # Show the plot
    # fig.show()
