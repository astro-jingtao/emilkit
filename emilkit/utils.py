import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def gaussian(x, amp, cen, sig):
    return amp * np.exp(-(x - cen)**2 / sig**2 / 2)


def lorentzian(x, amp, cen, sig, p0):
    return amp * sig / (sig**2 + (x - cen)**2) + p0


def to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x


def imshow(data,
           cmap='BuPu',
           fig=None,
           ax=None,
           cb_position='right',
           cb_size='5%',
           cb_pad=0.1,
           extend='both',
           *arg,
           **kwarg):
    """
    It takes an image and a colorbar position, and returns the image with a colorbar
    
    :param data: the data to be plotted
    :param cmap: the color map to use, defaults to BuPu (optional)
    :param fig: the figure to add the colorbar to
    :param ax: the axis to plot on
    :param cb_position: The position of the colorbar. Can be one of 'left', 'right', 'top', 'bottom',
    defaults to right (optional)
    :param cb_size: the size of the colorbar as a fraction of the axes, defaults to 5% (optional)
    :param cb_pad: the padding between the colorbar and the image
    """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(cb_position, size=cb_size, pad=cb_pad)
    im = ax.imshow(data, origin='lower', cmap=cmap, *arg, **kwarg)
    fig.colorbar(im, ax=ax, cax=cax, extend=extend)
    fig.sca(ax)  # 把当前 axis 设为 ax


def get_seg_cmap(bins, under=None, over=None, colors=None):
    """
    Get a colormap and norm for a given set of bins
    if colors is None, then the colors are linearly interpolated between under and over
    """

    nbin = len(bins) - 1
    norm = mcolors.BoundaryNorm(bins, nbin)

    # Calculate colors using linear interpolation
    # sourcery skip: merge-nested-ifs
    if colors is None:
        if under is not None and over is not None:
            num_colors = len(bins) + 1
            # colors = [mcolors.to_rgb(under)]
            colors = []
            for i in range(1, num_colors - 1):
                t = i / (num_colors - 1)
                color = [
                    t * c1 + (1 - t) * c2 for c1, c2 in zip(
                        mcolors.to_rgb(over), mcolors.to_rgb(under))
                ]
                colors.append(color)
            # colors.append(mcolors.to_rgb(over))

    cmap = mcolors.ListedColormap(colors, name='Nsigma')
    cmap.set_under(under)
    cmap.set_over(over)

    return cmap, norm

# TODO: support both ij and xy
def plot_polygon(pts, ax=None, *arg, **kwarg):
    '''
    the order is j, i or x, y
    '''
    if ax is None:
        ax = plt.gca()
    pts = np.vstack((pts, pts[:1]))
    codes = np.ones(pts.shape[0]) * Path.LINETO
    codes[0] = Path.MOVETO
    path = Path(pts, codes)
    patch = patches.PathPatch(path, *arg, **kwarg)
    ax.add_patch(patch)


def is_in_polygon(x, y, pts):
    '''
    the order is i, j or y, x
    '''
    x_shape = x.shape
    # y_shape = y.shape
    x = x.flatten()
    y = y.flatten()
    return Path(pts).contains_points(np.c_[x, y], radius=-0.1).reshape(x_shape)
