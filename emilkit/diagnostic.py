from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import plot_polygon

EPS = 1e-6
FILE_PATH = str(Path(__file__).parent)


class NIIBPT:
    # https://sites.google.com/site/agndiagnostics/agn-optical-line-diagnostics/bpt-diagrams?authuser=0
    def __init__(self):
        pass

    def draw(self,
             ax=None,
             x_min=-1.2,
             x_max=0.5,
             y_min=-1.2,
             y_max=1,
             xlabel=r'$\log$ [NII] 6583 / H$\alpha$',
             ylabel=r'$\log$ [OIII] 5007 / H$\beta$'):
        if ax is None:
            ax = plt.gca()

        xx = np.linspace(x_min, 0.05 - EPS)
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
    # https://sites.google.com/site/agndiagnostics/agn-optical-line-diagnostics/bpt-diagrams?authuser=0
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

        xx = np.linspace(x_min, 0.32 - EPS)
        ax.plot(xx, self.main_AGN_line(xx), '-.k', label='main AGN line')
        xx = np.linspace(-0.3, y_max)
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

    # https://sites.google.com/site/agndiagnostics/agn-optical-line-diagnostics/bpt-diagrams?authuser=0
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

        xx = np.linspace(x_min, -0.59 - EPS)
        ax.plot(xx, self.main_AGN_line(xx), '-.k', label='main AGN line')
        xx = np.linspace(-1.11, x_max)
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


class SMB:
    # Zanin+1999
    # https://articles.adsabs.harvard.edu/full/1999ASPC..188..231Z Figure 4

    def __init__(self):
        data_path = f'{FILE_PATH}/data/Zanin+1999_fig4'
        self.data = {}
        for n in ['HH', 'HII', 'PN_left', 'PN_R', 'SNR']:
            self.data[n] = pd.read_csv(f'{data_path}/{n}.csv', index_col=0)

    def draw(self,
             ax=None,
             x_min=-0.6,
             x_max=2.4,
             y_min=-1.0,
             y_max=1.3,
             xlabel=r'$\log$ H$\alpha$ / [SII] 6717,6731',
             ylabel=r'$\log$ H$\alpha$ / [NII] 6548,6583'):
        if ax is None:
            ax = plt.gca()

        plot_polygon(self.data['HH'][['X', 'Y']], ax=ax, facecolor='none')
        ax.text(-0.2, 0.5, 'HH')

        plot_polygon(self.data['HII'][['X', 'Y']], ax=ax, facecolor='none')
        ax.text(0.65, 0.45, 'HII')

        plot_polygon(self.data['SNR'][['X', 'Y']], ax=ax, facecolor='none')
        ax.text(-0.05, -0.4, '\n'.join('SNR'))

        ax.plot(self.data['PN_left']['X'], self.data['PN_left']['Y'], '-k')
        ax.plot(self.data['PN_R']['X'], self.data['PN_R']['Y'], '-k')
        ax.text(1.3, 0.45, 'PN')


        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def judge(self):
        pass
