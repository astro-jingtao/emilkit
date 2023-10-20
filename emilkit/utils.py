import numpy as np
import matplotlib.pyplot as plt
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
