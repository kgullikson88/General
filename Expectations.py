import pandas as pd
import numpy as np
import pysynphot
from scipy.optimize import leastsq
from astropy import units as u

import SpectralTypeRelations
import Mamajek_Table


MS = SpectralTypeRelations.MainSequence()
MT = Mamajek_Table.MamajekTable()
MT.mam_df['radius'] = 10 ** (0.5 * MT.mam_df.logL - 2.0 * MT.mam_df.logT + 2.0 * 3.762)
teff2radius = MT.get_interpolator('Teff', 'radius')
spt2teff = MT.get_interpolator('SpTNum', 'Teff')
sptnum2mass = MT.get_interpolator('SpTNum', 'Msun')
mass2teff = MT.get_interpolator('Msun', 'Teff')

mam_Tmin = MT.mam_df.Teff.min()
mam_Tmax = MT.mam_df.Teff.max()


def get_info(row, N_sigma=2, N_pts=1e4):
    if pd.notnull(row['sec_teff']):
        return row['sec_teff']

    # Get information from the dataframe
    delta_m = row['sec_mag'] - row['pri_mag']
    if pd.isnull(delta_m):
        print 'delta mag is null!'
        print row['sec_mag'], row['pri_mag']
        return np.nan

    delta_m_err = np.sqrt(row['sec_mag_err'] ** 2 + row['pri_mag_err'] ** 2)
    if pd.isnull(delta_m_err) or delta_m_err < 1e-4:
        delta_m_err = 0.1
    teff = row['pri_teff']
    filt_lam = row['filt_lam_normalized']
    filt_fwhm = max(10.0, row['filt_fwhm_normalized'])

    # Make the filter
    filt_sigma = filt_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    N_sigma = int(min(N_sigma, (filt_lam - 100) / filt_sigma))
    x = np.linspace(filt_lam - filt_sigma * N_sigma, filt_lam + filt_sigma * N_sigma, N_pts)
    T = np.exp(-(x - filt_lam) ** 2 / (2.0 * filt_sigma ** 2))
    filt = pysynphot.ArrayBandpass(x * u.nm.to(u.angstrom), T, name='Gaussian filter')

    # Fit the temperature of the secondary
    T_guess = 6000.0
    errfcn = lambda T, args: lnlike(T, *args)
    try:
        result, success = leastsq(errfcn, T_guess, args=[teff, delta_m, delta_m_err, filt, False])
    except:
        print(delta_m, delta_m_err, filt_lam, filt_fwhm)
        result, success = leastsq(errfcn, T_guess, args=[teff, delta_m, delta_m_err, filt, True])
    sec_teff = np.abs(result[0])
    if success > 3:
        return np.nan

    return sec_teff


def safe_fcn(row, N_sigma=5, N_pts=1e4):
    # T = get_info(row, N_sigma, N_pts)
    try:
        print row['pri_teff'].item()
        T = get_info(row, N_sigma, N_pts)
        return T
    except KeyError, e:
        print e
        return np.nan


def get_radius(T):
    if T > mam_Tmax:
        return teff2radius(mam_Tmax)
    elif T < mam_Tmin:
        return teff2radius(mam_Tmin)
    else:
        return teff2radius(T)


def lnlike(Teff, Teff_prim, delta_mag, delta_mag_err, bandpass, debug):
    """
    A log-likelihood function for secondary star temperature.
    Assumes we know what the primary star temperature is, and the delta-magnitude in this band!
    """
    if debug:
        print('{}\t{}'.format(Teff, Teff_prim))
    penalty = 0
    if Teff < 500:
        Teff = 500
        penalty = 100

    R1 = float(get_radius(Teff_prim))
    R2 = float(get_radius(Teff))
    # R1, R2 = 1.0, 1.0
    bb_prim = pysynphot.BlackBody(Teff_prim) * R1 ** 2
    obs_prim = pysynphot.Observation(bb_prim, bandpass)
    bb_sec = pysynphot.BlackBody(Teff) * R2 ** 2
    obs_sec = pysynphot.Observation(bb_sec, bandpass)
    dm = obs_sec.effstim('abmag') - obs_prim.effstim('abmag')
    return (dm - delta_mag) ** 2 / delta_mag_err ** 2 + penalty


def get_teff(pri_spt, magdiff, magdiff_err, filt_lam, filt_fwhm, pri_spt_err=1.0, N=300):
    """
    Get a probability distribution for the companion temperature from the magnitude difference.
    """
    sptnum = MS.SpT_To_Number(pri_spt)
    sptnum_arr = np.random.normal(loc=sptnum, scale=pri_spt_err, size=N)

    df = pd.DataFrame(data={'pri_teff': spt2teff(sptnum_arr),
                            'sec_teff': np.nan,
                            'pri_mag': 0.0, 'pri_mag_err': 0.0,
                            'sec_mag': magdiff, 'sec_mag_err': magdiff_err,
                            'filt_lam_normalized': filt_lam,
                            'filt_fwhm_normalized': filt_fwhm})
    df['sec_teff'] = df.apply(lambda r: safe_fcn(r, N_sigma=2), axis=1)
    return df[['pri_teff', 'sec_teff']]


"""
=============================================
   SB9 Processing
=============================================
"""


def get_primary_mass(spt):
    # if pd.isnull(spt):
    #    return np.nan
    return sptnum2mass(spt)


def get_secondary_mass(M1, q):
    M2 = float(M1) * q
    return M2


def get_teff_sb9(pri_spt, q, pri_spt_err=1.0, N=300):
    """
    Get a probability distribution for the companion temperature from the mass ratio
    :param pri_spt:
    :param q:
    :param pri_spt_err:
    :param N:
    :return:
    """
    sptnum = MS.SpT_To_Number(pri_spt)
    sptnum_arr = np.random.normal(loc=sptnum, scale=pri_spt_err, size=N)
    primary_mass = get_primary_mass(sptnum_arr)
    secondary_mass = primary_mass * q
    secondary_teff = mass2teff(secondary_mass)

    return pd.DataFrame(data={'M1': primary_mass, 'M2': secondary_mass, 'sec_teff': secondary_teff})
