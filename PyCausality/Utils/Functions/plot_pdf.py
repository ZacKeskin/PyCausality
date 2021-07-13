import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .pdf import get_pdf
from .sanitise import sanitise


def plot_pdf(
    df,
    estimator="kernel",
    gridpoints=None,
    bandwidth=None,
    covar=None,
    bins=None,
    show=False,
    cmap="inferno",
    label_fontsize=7,
):
    """Wrapper function to plot the pdf of a pandas dataframe.

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        estimator   -       (string)    'kernel' or 'histogram'
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df.
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'
        show        -       (Boolean)   whether or not to plot direclty, or simply return axes for later use
        cmap        -       (string)    Colour map (see: https://matplotlib.org/examples/color/colormaps_reference.html)
        label_fontsize -    (float)     Defines the fontsize for the axes labels

    Returns:
        ax          -       AxesSubplot object. Can be added to figures to allow multiple plots.

    """
    DF = sanitise(df)
    if len(DF.columns) != 2:
        print(
            "DataFrame has "
            + str(len(DF.columns))
            + " dimensions. Only 2D or less can be plotted"
        )
        axes = None
    else:
        # Plot data in Histogram or Kernel form
        if estimator == "histogram":

            if bins is None:
                bins = {
                    axis: np.linspace(DF[axis].min(), DF[axis].max(), 9)
                    for axis in DF.columns.values
                }
            fig, axes = plot_pdf_histogram(df, bins, cmap)
        else:
            fig, axes = plot_pdf_kernel(df, gridpoints, bandwidth, covar, cmap)

        # Format plot
        axes.set_xlabel(DF.columns.values[0], labelpad=20)
        axes.set_ylabel(DF.columns.values[1], labelpad=20)

        for label in axes.xaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)

        for label in axes.yaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)

        for label in axes.zaxis.get_majorticklabels():
            label.set_fontsize(label_fontsize)

        axes.view_init(10, 45)

        if show is True:
            plt.show()
        plt.close(fig)

        axes.remove()

    return axes


def plot_pdf_histogram(df, bins, cmap="inferno"):
    """Function to plot the pdf of a dataset, estimated via histogram.

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator = 'histogram'

    Returns:
        ax          -       AxesSubplot object, passed back via to plot_pdf() function

    """
    DF = sanitise(df)  # in case function called directly

    # Calculate PDF
    PDF = get_pdf(df=DF, estimator="histogram", bins=bins)

    # Get x-coords, y-coords for each bar
    (x_edges, y_edges) = bins.values()
    X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])

    # Get dx, dy for each bar
    dxs, dys = np.meshgrid(np.diff(x_edges), np.diff(y_edges))

    # Colourmap
    cmap = cm.get_cmap(cmap)
    rgba = [
        cmap((p - PDF.flatten().min()) / PDF.flatten().max()) for p in PDF.flatten()
    ]

    # Create subplots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(
        x=X.flatten(),  # x coordinates of each bar
        y=Y.flatten(),  # y coordinates of each bar
        z=0,  # z coordinates of each bar
        dx=dxs.flatten(),  # width of each bar
        dy=dys.flatten(),  # depth of each bar
        dz=PDF.flatten(),  # height of each bar
        alpha=1,  # transparency
        color=rgba,
    )
    ax.set_title("Histogram Probability Distribution", fontsize=10)

    return fig, ax


def plot_pdf_kernel(df, gridpoints=None, bandwidth=None, covar=None, cmap="inferno"):
    """Function to plot the pdf, calculated by KDE, of a dataset.

    Args:
        df          -       (DataFrame) Samples over which to estimate density
        gridpoints  -       (int)       Number of gridpoints when integrating KDE over
                                        the domain. Used if estimator='kernel'
        bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                        matrix). Used if estimator='kernel'
        covar       -       (Numpy ndarray) Covariance matrix between dimensions of df.

    Returns:
        ax          -       AxesSubplot object, passed back via to plot_pdf() function

    """
    DF = sanitise(df)

    # Estimate the PDF from the data
    if gridpoints is None:
        gridpoints = 20

    pdf = get_pdf(DF, gridpoints=gridpoints, bandwidth=bandwidth)
    N = complex(gridpoints)
    slices = [slice(dim.min(), dim.max(), N) for dimname, dim in DF.iteritems()]  # noqa
    X, Y = np.mgrid[slices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, pdf, cmap=cmap)

    ax.set_title("KDE Probability Distribution", fontsize=10)

    return fig, ax
