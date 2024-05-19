import numpy as np


def calz_unred(wave, ebv, flux=None, Rv=4.05):
    '''
    Convert 'calz_unred.pro' from IDL to python.
    ############################################
    purpose:
    --------
    Deredden a galaxy spectrum using the Calzetti et al. (2000) recipe.
    Calzetti et al. (2000, ApJ 533, 682) developed a recipe for dereddening
    the spectra of galaxies where massive stars dominate the radiation output,
    valid between 0.12 to 2.2 microns. (CALZ_UNRED extrapolates between
    0.12 and 0.0912 microns.)

    parameters:
    ----------
    wave : wavelength vector (Angstroms).
    flux : calibrated flux vector, same number of elements as WAVE.
    ebv : color excess E(B-V), scalar.  If a negative EBV is supplied,
          then fluxes will be reddened rather than deredenned.
          Note that the supplied color excess should be that derived for
          the stellar  continuum, EBV(stars), which is related to the
          reddening derived from the gas, EBV(gas), via the Balmer
          decrement by EBV(stars) = 0.44*EBV(gas)
    Rv : Ratio of total to selective extinction, default = 4.05.
         Calzetti et al. (2000) estimate R_V = 4.05 +/- 0.80 from optical
         -IR observations of 4 starbursts.

    return
    ------
    the flux after redden or deredden.

    example:
    --------
    wave = np.arange(6000) + 3000
    flux = wave * 0. + 1
    funred = calz_unred(wave, flux, 0.1)
    '''
    wave = np.asarray(wave, dtype=float)
    w1 = (wave >= 6300) & (wave <= 22000)
    w2 = (wave >= 912) & (wave < 6300)
    x = 10000. / wave

    # if w1.sum() + w2.sum() != wave.size:
    #     print('Warning - some elements of wavelength vector outside valid domain.')
    klam = np.zeros_like(x)
    klam[w1] = 2.659 * (-1.857 + 1.040 * x[w1]) + Rv
    c2 = np.array([-2.156, 1.509, -0.198, 0.011])
    p2 = np.poly1d(c2[::-1])
    klam[w2] = 2.659 * p2(x[w2]) + Rv

    unred_factor = np.power(10, 0.4 * klam * ebv)

    if flux is None:
        return unred_factor
    flux = np.asarray(flux, dtype=float)
    return flux * unred_factor


def ccm_unred(wave, ebv, flux=None, Rv=3.1):
    '''
    Convert 'ccm_unred.pro' from IDL to python.
    ###########################################
    purpose:
    --------
    Deredden a galaxy spectrum using the CCM 1989 parameterization.
    The reddening curve is that of Cardelli, Clayton, and Mathis (1989 ApJ.
    345, 245), including the update for the near-UV given by O'Donnell
    (1994, ApJ, 422, 158). Parameterization is valid from the IR to the
    far-UV (3.5 microns to 0.1 microns).

    parameters:
    ----------
    wave : wavelength vector (Angstroms).
    flux : calibrated flux vector, same number of elements as WAVE.
    ebv : color excess E(B-V), scalar.  If a negative EBV is supplied,
          then fluxes will be reddened rather than deredenned.
    Rv : scalar specifying the ratio of total selective extinction
         R(V) = A(V) / E(B - V). If not specified, then R_V = 3.1
         Extreme values of R(V) range from 2.75 to 5.3

    return
    ------
    the flux after redden or deredden.
    '''
    wave = np.asarray(wave, dtype=float)
    x = 10000. / wave
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    # ########## Infrared ##########
    is_in_this_band = (x > 0.3) & (x < 1.1)
    a[is_in_this_band] = 0.574 * x[is_in_this_band]**1.61
    b[is_in_this_band] = -0.527 * x[is_in_this_band]**1.61
    # ########## Optical/NIR ##########
    is_in_this_band = (x >= 1.1) & (x < 3.3)
    y = x[is_in_this_band] - 1.82
    # Original coefficients from CCM89
    # c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085, 0.01979, -0.77530,  0.32999 ]
    # c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434, -0.62251,  5.30260, -2.09002 ]
    # New coefficients from O'Donnell (1994)
    c1 = np.array(
        [1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505])
    c2 = np.array(
        [0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])
    p1 = np.poly1d(c1[::-1])
    p2 = np.poly1d(c2[::-1])
    a[is_in_this_band] = p1(y)
    b[is_in_this_band] = p2(y)
    # ########## Mid-UV ##########
    is_in_this_band = (x >= 3.3) & (x < 8)
    y = x[is_in_this_band]
    F_a = np.zeros_like(y)
    F_b = np.zeros_like(y)
    good1 = y > 5.9
    y1 = y[good1] - 5.9
    F_a[good1] = -0.04473 * y1**2 - 0.009779 * y1**3
    F_b[good1] = 0.2130 * y1**2 + 0.1207 * y1**3
    a[is_in_this_band] = 1.752 - 0.316 * y - (0.104 /
                                              ((y - 4.67)**2 + 0.341)) + F_a
    b[is_in_this_band] = -3.090 + 1.825 * y + (1.206 /
                                               ((y - 4.62)**2 + 0.263)) + F_b
    # ########## Far-UV ##########
    is_in_this_band = (x >= 8) & (x <= 11)
    y = x[is_in_this_band] - 8.
    c1 = np.array([-1.073, -0.628, 0.137, -0.070])
    c2 = np.array([13.670, 4.257, -0.420, 0.374])
    p1 = np.poly1d(c1[::-1])
    p2 = np.poly1d(c2[::-1])
    a[is_in_this_band] = p1(y)
    b[is_in_this_band] = p2(y)
    # Now apply extinction correction to input flux vector
    Av = Rv * ebv
    Alambda = Av * (a + b / Rv)

    unred_factor = np.power(10, 0.4 * Alambda)

    if flux is None:
        return unred_factor
    flux = np.asarray(flux, dtype=float)
    return flux * unred_factor


def get_ebv_by_HaHb(ha_hb_ratio_obs):
    """
    Calculate the color excess E(B-V) given the observed Halpha/Hbeta ratio.

    Parameters:
    ha_hb_ratio_obs (float): The observed ratio of Halpha to Hbeta.

    Returns:
    float: The color excess E(B-V).
    """
    intrinsic_ratio = 2.86

    # Calculate E(B-V)
    e_b_v = 1.97 * np.log10(ha_hb_ratio_obs / intrinsic_ratio)

    return e_b_v
