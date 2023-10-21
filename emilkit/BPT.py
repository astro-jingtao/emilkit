# https://sites.google.com/site/agndiagnostics/agn-optical-line-diagnostics/bpt-diagrams?authuser=0

import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-6


class NIIBPT:

    def __init__(self):
        pass

    def draw(self,
             ax=None,
             x_min=-1.2,
             x_max=0.5,
             y_min=-1.2,
             y_max=1,
             xlabel=r'$\log$ [NII] 6584 / H$\alpha$',
             ylabel=r'$\log$ [OIII] 5007 / H$\beta$'):
        if ax is None:
            ax = plt.gca()

        xx = np.linspace(-1.2, 0.05 - EPS)
        ax.plot(xx, self.K03(xx), '--k', label='Kauffmann+03')
        xx = np.linspace(-1.2, 0.47 - EPS)
        ax.plot(xx, self.K01(xx), '-.k', label='Kewley+01')
        ax.legend()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

    def judge(self):
        pass

    def K03(self, X):
        return 0.61 / (X - 0.05) + 1.3

    def K01(self, X):
        return 0.61 / (X - 0.47) + 1.19


class SIIBPT:

    def __init__(self):
        pass

    def draw(self,
             ax=None,
             x_min=-1.2,
             x_max=0.5,
             y_min=-1.2,
             y_max=1,
             xlabel=r'$\log$ [SII] 6717,6731 / H$\alpha$',
             ylabel=r'$\log$ [OIII] 5007 / H$\beta$'):
        if ax is None:
            ax = plt.gca()

        xx = np.linspace(-1.2, 0.32 - EPS)
        ax.plot(xx, self.main_AGN_line(xx), '-.k', label='main AGN line')
        xx = np.linspace(-0.3, 1)
        ax.plot(xx, self.LINER_Sy2_line(xx), '--k', label='LINER/Sy2 line')
        ax.legend()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def judge(self):
        pass

    def main_AGN_line(self, x):
        return 0.72 / (x - 0.32) + 1.3

    def LINER_Sy2_line(self, x):
        return 1.89 * x + 0.76


class OIBPT:
    '''[OI] 6300'''

    def __init__(self):
        pass

    def draw(self,
             ax=None,
             x_min=-2,
             x_max=0,
             y_min=-1.2,
             y_max=1,
             xlabel=r'$\log$ [OI] 6300 / H$\alpha$',
             ylabel=r'$\log$ [OIII] 5007 / H$\beta$'):
        if ax is None:
            ax = plt.gca()

        xx = np.linspace(-2, -0.59 - EPS)
        ax.plot(xx, self.main_AGN_line(xx), '-.k', label='main AGN line')
        xx = np.linspace(-1.11, 1)
        ax.plot(xx, self.LINER_Sy2_line(xx), '--k', label='LINER/Sy2 line')
        ax.legend()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def judge(self):
        pass

    def main_AGN_line(self, x):
        return 0.73 / (x + 0.59) + 1.33

    def LINER_Sy2_line(self, x):
        return 1.18 * x + 1.30
