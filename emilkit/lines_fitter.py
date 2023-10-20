import itertools
import numpy as np
import lmfit
from scipy.signal import savgol_filter
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt

from .utils import gaussian, to_list, imshow

c = 299792.458  # Speed of light in km/s

sigclip = SigmaClip(sigma=3, maxiters=5)
sigclip_err = SigmaClip(sigma=2, maxiters=5)


def use_side_windows(wave,
                     left_window,
                     right_window,
                     residual,
                     error,
                     use_error=False):
    left_wave = wave[left_window].mean()
    right_wave = wave[right_window].mean()
    residual_left_masked = sigclip(residual[left_window])
    residual_right_masked = sigclip(residual[right_window])
    if use_error:
        residual_left_masked.mask = sigclip_err(
            error[left_window]).mask | residual_left_masked.mask
        residual_right_masked.mask = sigclip_err(
            error[right_window]).mask | residual_right_masked.mask
    left_flux = residual_left_masked.mean()
    right_flux = residual_right_masked.mean()
    return left_wave, left_flux, right_wave, right_flux


def lines_prepocess(nl, zero_points, wave, residual, error, lines_wave,
                    lines_name, use_weight, size, n_cut, use_error):
    wide_lines = [
        "NII5755", "NI5200", "OI6300", "ArIII7135", "HeI5876",
        "CaII7291&OII7320&CaII7324&OII7330", 'SIII9069', 'SIII9530',
        'CI9826&CI9852'
    ]
    red_end_line = ['CI8729', 'SIII9069', 'SIII9530']
    use_err_lines = ['CI8729', 'SIII9069', 'SIII9530', 'CI9826&CI9852']

    center_window_width = 10
    if lines_name in wide_lines:
        size = 50
    if lines_name in red_end_line:
        n_cut = 5
    if lines_name in use_err_lines:
        use_error = True
    if (wave[0] > lines_wave[0] - size) or (wave[-1] < lines_wave[-1] + size):
        raise ValueError("Lines is out of fitting window")

    left_window = (wave > lines_wave[0] - size) & (wave <= lines_wave[0] -
                                                   center_window_width)
    right_window = (wave >= lines_wave[-1] +
                    center_window_width) & (wave <= lines_wave[-1] + size)
    indx = (wave > lines_wave[0] - center_window_width) & (
        wave < lines_wave[-1] + center_window_width)
    if (indx.sum() <= n_cut) or (left_window.sum()
                                 <= n_cut) or (right_window.sum() <= n_cut):
        raise ValueError("Too few data in left/fitting/right window")

    left_wave, left_flux, right_wave, right_flux = use_side_windows(
        wave, left_window, right_window, residual, error, use_error=use_error)

    for i in range(nl):
        zero_points[i] = np.array(
            [left_wave, left_flux, right_wave, right_flux])

    lam = wave[indx]
    wei = 1 / error[indx] if use_weight else np.ones_like(lam)
    flux_zero = (lam - left_wave) * (right_flux - left_flux) / \
        (right_wave - left_wave) + left_flux
    emi = residual[indx] - flux_zero

    return emi, lam, wei


def lines(wave,
          residual,
          error,
          lines_wave,
          lines_name=None,
          size=30,
          n_cut=10,
          use_error=False,
          use_weight=True):

    nl = len(lines_wave)
    res = np.zeros((nl, 3))
    res_err = np.zeros((nl, 3))
    zero_points = np.zeros((nl, 4))

    try:
        lines_regions = lines_name.split('+')
        len_lines_regions = len(lines_regions)
        if len_lines_regions == 1:
            emi, lam, wei = lines_prepocess(nl, zero_points, wave, residual,
                                            error, lines_wave, lines_name,
                                            use_weight, size, n_cut, use_error)
        else:
            head = 0
            emi = []
            lam = []
            wei = []
            for i in range(len_lines_regions):
                n_lines_in_region = len(lines_regions[i].split(
                    '&'))  # how many lines in this region
                head_new = head + n_lines_in_region
                emi_, lam_, wei_ = lines_prepocess(
                    n_lines_in_region, zero_points[head:head_new], wave,
                    residual, error, lines_wave[head:head_new],
                    lines_regions[i], use_weight, size, n_cut, use_error)

                emi.append(emi_)
                lam.append(lam_)
                wei.append(wei_)

                head = head_new
            emi = np.hstack(emi)
            lam = np.hstack(lam)
            wei = np.hstack(wei)

    except Exception:
        return res, res_err, (0., 0., 0., 0., 0., 0.), zero_points

    params = lmfit.Parameters()
    for i in range(nl):
        if i == 0:
            gmodel = lmfit.Model(gaussian, prefix=f'g{i}_')
        else:
            gmodel += lmfit.Model(gaussian, prefix=f'g{i}_')
        params.add(f'g{i}_cen',
                   lines_wave[i],
                   min=lines_wave[i] - 5,
                   max=lines_wave[i] + 5)

        params.add(f'g{i}_sig', lines_wave[i] * 100 / c, min=0.5, max=10)
        params.add(f'g{i}_amp', 1., min=0.)
        params.add(f'g{i}_area', expr=f'g{i}_amp*g{i}_sig*sqrt(2*pi)')

    # sourcery skip: switch
    if lines_name == 'OII3726&OII3729':
        params['g1_sig'].expr = 'g0_sig'
    if lines_name == 'SII4069&SII4076&Hdelta4102':
        params['g1_sig'].expr = 'g0_sig'
    if lines_name == 'CI9826&CI9852':
        params['g1_sig'].expr = 'g0_sig'
    if lines_name == 'CaII7291&OII7320&CaII7324&OII7330':
        # 全放开意义不大，结果基本上是差不多的
        # OII 的线不该绑在一起，因为 7320 和 7330 的两根线相对强度是不同的
        # 合起来造成的线宽也就是不同的
        # params['g3_sig'].expr = 'g1_sig'
        # params['g3_area'].expr = 'g1_area*0.805'

        params.add('O2_width_ratio', 1, min=0.8, max=1.25)
        params['g3_sig'].expr = 'g1_sig*O2_width_ratio'

        # 距离太远，仪器展宽不同
        # params['g0_sig'].expr = 'g2_sig/7291*3969'

        params['g2_sig'].expr = 'g0_sig/7291.5*7323.9'
        params['g2_cen'].expr = '(g0_cen-7291.5)/7291.5*7323.9+7323.9'

        # params.add('Ca_ratio', 0.69, min=0.65, max=0.75)
        # params['g4_area'].expr = 'g2_area*Ca_ratio'  # Yan 2018 P12
    mires = gmodel.fit(emi, params, x=lam, weights=wei)
    for i in range(nl):
        keys = [f'g{i}_cen', f'g{i}_sig', f'g{i}_area']
        for j, key in enumerate(keys):
            res[i, j] = mires.params[key].value
            res_err[i, j] = mires.params[key].stderr
    return res, res_err, (0., 0., 0., 0., 0., 0), zero_points


def save_lines(line_wave,
               wave,
               scont,
               zero_points,
               res=None,
               res_err=None,
               res_b=None,
               broad=False,
               rest_center=True):
    result = np.zeros(12)
    if rest_center:
        line_scont = np.interp([line_wave], wave, scont)[0]
    elif broad:
        line_scont = np.interp([res_b[4]], wave, scont)[0]
    else:
        line_scont = np.interp([res[0]], wave, scont)[0]
    if broad:
        if res_b[3] is None:
            res_b[3] = 0
        vals = [
            res_b[4], res_b[0], res_b[2], res_b[2] / line_scont, res_b[5],
            res_b[1], res_b[3], res_b[3] / line_scont
        ]
        for i in range(8):
            result[i] = vals[i]
    else:
        for i in range(3):
            result[i] = res[i]
            result[i + 4] = res_err[i]
        result[3] = res[2] / line_scont
        result[7] = res_err[2] / line_scont
    result[8:] = zero_points
    return result


def emlines(wave, residual, error_gal, line_add=None, use_weight=True):
    # FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated
    scont = savgol_filter(residual, 201, 3)

    result = {}
    line = {
        'OII3726&OII3729': [3726, 3729],
        'Heta3836': 3836,
        'NeIII3869&Hzeta3889': [3869, 3889],
        'Hepsilon3970': 3970,
        'SII4069&SII4076&Hdelta4102': [4069., 4076., 4102],
        'Hgamma4340&OIII4363': [4340, 4363],
        'Hb4861': 4861,
        'OIII4959': 4959,
        'OIII5007': 5007,
        'NI5200': 5200,
        'NII5755': 5755,
        'HeI5876': 5876,
        'OI6300': 6300.,
        'OI6366': 6366,
        'NII6548&Ha6563&NII6583': [6548., 6563., 6583.],
        'SII6717&SII6731': [6717., 6731.],
        'ArIII7135': 7135.,
        'CaII7291&OII7320&CaII7324&OII7330': [7291, 7320, 7323, 7330],
        'CI8729': 8729,
        'SIII9069': 9069,
        'SIII9530': 9530,
        'CI9826&CI9852': [9826, 9852]
    }
    if line_add is not None:
        line |= line_add

    for line_name, line_wave in line.items():
        if not isinstance(line_wave, list):
            line_wave = [line_wave]
        tmp_res, tmp_res_err, _, zero_points = lines(wave,
                                                     residual,
                                                     error_gal,
                                                     line_wave,
                                                     line_name,
                                                     use_weight=use_weight)
        line_name_ = line_name.replace('+', '&')  # replace + by &
        line_names = line_name_.split('&')
        for l in range(len(line_names)):
            result[line_names[l]] = save_lines(line_wave[l],
                                               wave,
                                               scont,
                                               zero_points[l],
                                               res=tmp_res[l],
                                               res_err=tmp_res_err[l])
    return result


def all_lines_plot(wave,
                   flux,
                   error,
                   emlines,
                   x_shape,
                   y_shape,
                   shadow_kwargs={}):
    fig, axes = plt.subplots(x_shape, y_shape, dpi=200, figsize=(16, 16))
    keys = list(emlines.keys())
    model = get_lines_emission(wave, emlines)
    k = 0
    for i in range(x_shape):
        for j in range(y_shape):
            if k >= len(keys):
                continue
            to_show = (wave > emlines[keys[k]][0] -
                       15 * emlines[keys[k]][1]) & (wave < emlines[keys[k]][0]
                                                    + 15 * emlines[keys[k]][1])
            if to_show.sum() < 10:
                to_show = (wave > int(keys[k][-4:]) -
                           20) & (wave < int(keys[k][-4:]) + 20)
            left_wave, left_flux, right_wave, right_flux = emlines[keys[k]][8:]
            zeros = (wave[to_show] - left_wave) * (right_flux - left_flux) / \
                (right_wave - left_wave) + left_flux
            flux_to_show = flux[to_show] - zeros
            error_to_show = error[to_show]
            axes[i, j].plot(wave[to_show], flux_to_show)
            axes[i,
                 j].fill_between(wave[to_show], flux_to_show - error_to_show,
                                 flux_to_show + error_to_show, **shadow_kwargs)
            axes[i, j].plot(wave[to_show], model[to_show])
            vlines_max = 1 if flux_to_show.size == 0 else 1.1 * np.max(
                flux_to_show)
            if emlines[keys[k]][0] == 0:
                axes[i, j].vlines(int(keys[k][-4:]),
                                  0,
                                  vlines_max,
                                  color='red',
                                  linestyle='--')
                postfix = '(NF)'
            else:
                axes[i, j].vlines(emlines[keys[k]][0],
                                  0,
                                  vlines_max,
                                  color='red',
                                  linestyle='--')
                postfix = ''
            axes[i, j].set_title(keys[k] + postfix)
            k += 1
    fig.tight_layout()


def get_lines_emission(wave, emlines):
    emlines_fit = np.zeros_like(wave)
    # sourcery skip: move-assign, remove-pass-body
    for key in emlines.keys():
        cen = emlines[key][0]
        sig = emlines[key][1]
        area = emlines[key][2]
        if (sig <= 0):
            continue
        if (key == 'Ha6563_b') | (key == 'Hb4861_b'):
            pass
        else:
            amp = area / sig / np.sqrt(2 * np.pi)

            emlines_fit += gaussian(wave, amp, cen, sig)
    return emlines_fit
