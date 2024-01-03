import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib


def plot_bode(path_amplitude, path_phase):
    """Plot the bode diagram of a given file.

    Args:
        path (str): Path to the file.
    """
    # read the data
    df = pd.read_csv(path_amplitude, sep=",", skiprows=3, header=None)
    # set column names
    df.columns = ["frequency", "amplitude"]
    # plot the data
    df.plot(x="frequency", y="amplitude")
    # log scale
    plt.yscale("log")
    plt.xscale("log")
    # grid
    plt.grid()
    plt.show()
    df_phase = pd.read_csv(path_phase, sep=",", skiprows=3, header=None)
    # set column names
    df_phase.columns = ["frequency", "phase"]
    # plot the data
    df_phase.plot(x="frequency", y="phase")
    plt.grid()
    plt.show()


# plot in the same figure
def plot_bode_same(path_amplitude, path_phase, title="Bode Diagram", save_path="bode_diagram.tex"):
    """Plot the bode diagram of a given file.

    Args:
        path (str): Path to the file.
    """
    # read the data
    df = pd.read_csv(path_amplitude, sep=",", skiprows=3, header=None)
    # set column names
    df.columns = ["frequency", "amplitude"]

    # read the data
    df_phase = pd.read_csv(path_phase, sep=",", skiprows=3, header=None)
    # set column names
    df_phase.columns = ["frequency", "phase"]
    # plot the data
    # differnt y scales for amplitude and phase
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("frequency (Hz)")
    # set x scale to log
    ax1.set_xscale("log")
    ax1.set_ylabel("amplitude ($\Omega$)", color=color)
    # set y scale to log
    ax1.set_yscale("log")
    ax1.plot(df["frequency"], df["amplitude"], color=color)
    # plot as scatter plot with curve fitting
    # ax1.scatter(df["frequency"], df["amplitude"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:blue"
    ax2.set_ylabel("phase ($\degree$)", color=color)  # we already handled the x-label with ax1
    ax2.plot(df_phase["frequency"], df_phase["phase"], color=color)
    # ax2.scatter(df_phase["frequency"], df_phase["phase"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid()
    plt.title(title)
    tikzplotlib.save(save_path)

    plt.show()


if __name__ == "__main__":
    title = "Bode Diagram Katroffel"
    path_amplitude = "C:/Users/lgudjons/Desktop/kartoffel_amplitude.sig"
    path_phase = "C:/Users/lgudjons/Desktop/kartoffel_phase.sig"
    plot_bode_same(path_amplitude, path_phase, title=title
                   , save_path="bode_diagram_unknown.tex")
    title = "Bode Diagram Wurst"
    path_amplitude = "C:/Users/lgudjons/Desktop/wurst_amplitude.sig"
    path_phase = "C:/Users/lgudjons/Desktop/wurst_phase.sig"
    plot_bode_same(path_amplitude, path_phase, title=title
                   , save_path="bode_diagram_kartoffel.tex")
