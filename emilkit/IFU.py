import itertools

import numpy as np
import matplotlib.pyplot as plt

from .utils import imshow, to_list

c = 299792.458  # Speed of light in km/s

# https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion


def vacuum_to_air(wavelength_vac):
    """Converts a wavelength from vacuum to air."""
    s = 10**4 / wavelength_vac
    n = 1.0 + 0.0000834254 + 0.02406147 / (130.0 - s**2) + 0.00015998 / (38.9 -
                                                                         s**2)
    return wavelength_vac / n


def air_to_vacuum(wavelength_air):
    """Converts a wavelength from air to vacuum."""
    s = 10**4 / wavelength_air
    n = 1.0 + 0.00008336624212083 + 0.02408926869968 / (
        130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return wavelength_air * n


def get_line_info_single(line_info, name, vac_wave=False):
    # NI5200 看起来也是多条线
    # OII7320/OII7330 实际上是四条线 7319.0 + 7320.0 / 7329.7 + 7330.7
    lines_center = {
        'OII3726': 3726.0,
        'OII3729': 3728.8,
        'Heta3836': 3835.4,
        'NeIII3869': 3868.8,
        'Hzeta3889': 3889.1,
        'CaII3969': 3968.5,
        'Hepsilon3970': 3970.1,
        'SII4069': 4068.6,
        'SII4076': 4076.4,
        'Hdelta4102': 4101.7,
        'Hgamma4340': 4340.5,
        'OIII4363': 4363.2,
        'Hb4861': 4861.4,
        'OIII4959': 4958.9,
        'OIII5007': 5006.9,
        'NI5200': 5200,
        'NII5755': 5754.6,
        'HeI5876': 5875.6,
        'OI6300': 6300.3,
        'OI6366': 6366.3,
        'NII6548': 6548.1,
        'Ha6563': 6562.8,
        'NII6583': 6583.5,
        'SII6717': 6716.4,
        'SII6731': 6730.8,
        'ArIII7135': 7135.8,
        'CaII7291': 7291.5,
        'OII7320': 7320,
        'CaII7324': 7323.9,
        'OII7330': 7330,
        'CI8729': 8727.1,
        'SIII9069': 9068.9,
        'SIII9530': 9532.1,
        'CI9826': 9824.3,
        'CI9852': 9850.3
    }

    if vac_wave:
        lines_center = {k: air_to_vacuum(v) for k, v in lines_center.items()}

    if line_info[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    center_raw = lines_center.get(name, int(name[-4:]))
    v = (line_info[0] - center_raw) / center_raw * c
    v_err = line_info[4] / center_raw * c
    vd = line_info[1] / center_raw * c
    vd_err = line_info[5] / center_raw * c
    flux = line_info[2]
    flux_err = line_info[6]
    return v, v_err, vd, vd_err, flux, flux_err


def get_line_info(lines_info, vac_wave=False):
    return {
        name: get_line_info_single(lines_info[name], name, vac_wave=vac_wave)
        for name in lines_info.keys()
    }


def get_line_info_all(sps, vac_wave=False):
    res = {
        name: {
            data_type: np.zeros((42, 42))
            for data_type in ['v', 'v_err', 'vd', 'vd_err', 'flux', 'flux_err']
        }
        for name in sps[10][10].keys()
    }
    for i, j in itertools.product(range(42), range(42)):
        if sps[i][j] is not None:
            data = get_line_info(sps[i][j], vac_wave=vac_wave)
            for name, (k, data_type) in itertools.product(
                    sps[10][10].keys(),
                    enumerate(
                        ['v', 'v_err', 'vd', 'vd_err', 'flux', 'flux_err'])):
                res[name][data_type][i, j] = data[name][k]
        else:
            for name in sps[10][10].keys():
                for data_type in [
                        'v', 'v_err', 'vd', 'vd_err', 'flux', 'flux_err'
                ]:
                    res[name][data_type][i, j] = np.nan
    return res


class MaNGALinesResult:

    def __init__(self, res):
        self.res = res

    def get_line_ratio(self, l1, l2, log=False, with_error=False):
        l1 = to_list(l1)
        l2 = to_list(l2)
        l1_flux = sum(self.res[l]['flux'] for l in l1)
        l2_flux = sum(self.res[l]['flux'] for l in l2)
        if with_error:
            l1_var = sum(np.square(self.res[l]['flux_err']) for l in l1)
            l2_var = sum(np.square(self.res[l]['flux_err']) for l in l2)
        lr = np.log10(l1_flux / l2_flux) if log else l1_flux / l2_flux
        if with_error:
            lr_err = np.sqrt(l1_var / np.square(l2_flux) + l2_var /
                             np.power(l2_flux, 4) * np.square(l1_flux))
            if log:
                lr_err = lr_err / (np.log(10) * np.power(10, lr))
            return lr, lr_err
        else:
            return lr

    def get_surface_brightness(self, l):
        # in erg/s/kpc^2
        return self.res[l]['flux'] * 4 * np.pi / (np.deg2rad(
            0.5 / 3600)**2) * (3.0856776e21)**2 * 1e-17

    def draw_line_ratio(self, l1, l2, log=False, *arg, **kwarg):
        l1 = to_list(l1)
        l2 = to_list(l2)
        lr = self.get_line_ratio(l1, l2, log=log)
        imshow(lr, *arg, **kwarg)
        p1 = '(' + '+'.join(l1) + ')' if len(l1) > 1 else l1[0]
        p2 = '(' + '+'.join(l2) + ')' if len(l2) > 1 else l2[0]
        if log:
            plt.title(f'log {p1}/{p2}')
        else:
            plt.title(f'{p1}/{p2}')

    def get_snr(self, l, v):
        return np.abs(self.res[l][v] / self.res[l][f'{v}_err'])

    def draw_surface_brightness(self, l, log=False, *arg, **kwarg):
        sb = self.get_surface_brightness(l)
        if log:
            sb = np.log10(sb)
        imshow(sb, *arg, **kwarg)
        if log:
            plt.title(f'{l} log surface brightness (erg/s/kpc^2)')
        else:
            plt.title(f'{l} surface brightness (erg/s/kpc^2)')
        plt.tight_layout()

    def draw_velocity(self, l, *arg, **kwarg):
        imshow(self.res[l]['v'], *arg, **kwarg)
        plt.title(f'{l} velocity')

    def draw_velocity_dispersion(self, l, *arg, **kwarg):
        imshow(self.res[l]['vd'], *arg, **kwarg)
        plt.title(f'{l} velocity dispersion')
