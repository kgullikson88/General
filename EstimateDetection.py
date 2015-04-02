"""
This module is generally for estimating the latest detectable spectral type.
For now, it has a function to get the expected vsini PDF of a given companion, given the age of the system.
"""

import os
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import InterpolatedUnivariateSpline as spline, griddata
from astropy import units as u, constants

import StarData
import SpectralTypeRelations
import Sensitivity
import Mamajek_Table


# Read in the Barnes & Kim (2010) table (it is in LaTex format)
home = os.environ['HOME']
bk_filename = '{}/Dropbox/School/Research/Databases/SpT_Relations/Barnes_and_Kim2010_table1.txt'.format(home)
bk_data = pd.read_csv(bk_filename, sep='&', usecols=(0,1,2,3,4), dtype={'Mass': np.float,
                                                                      'logT': np.float,
                                                                      'logL': np.float,
                                                                      'Age': np.float,
                                                                      'global_tc': np.float})
bk_data = bk_data.assign(Teff = lambda x: 10**x.logT)
bk_data = bk_data.assign(Radius = lambda x: 10**(0.5*x.logL) * (x.Teff/5680)**(-2))

# Interpolate
teff2tau_int = spline(bk_data.Teff, bk_data.global_tc)
teff2radius_int = spline(bk_data.Teff, bk_data.Radius)

# Make functions that treat extrapolation in a slightly more sane way
Tmin = bk_data.Teff.min()
Tmax = bk_data.Teff.max()
def teff2tau(T):
    if T < Tmin:
        return teff2tau(Tmin)
    elif T > Tmax:
        return teff2tau(Tmax)
    return max(0.1, teff2tau_int(T))
def teff2radius(T):
    if T < Tmin:
        return teff2radius_int(Tmin)
    elif T > Tmax:
        return teff2radius_int(Tmax)
    return teff2radius_int(T)

# Make a mainsequence instance
MS = SpectralTypeRelations.MainSequence()


def lnlike(P, P_0, t, tau, k_C, k_I):
    """
    This is the likelihood function for getting the period out of Equation 19 in Barnes (2010)
    :param P: Period (what we will be solving for)
    :param P_0: Initial period
    :param t: age
    :param tau: convection turnover timescale
    :param k_C: parameter (constant)
    :param k_I: parameter (constant)
    :return:
    """
    retval = (k_C*t/tau - np.log(P/P_0) - k_I*k_C/(2.0*tau**2) * (P**2 - P_0**2))**2
    return retval


def get_period_dist(ages, P0_min, P0_max, T_star, N_P0=1000, k_C=0.646, k_I=452):
    """
    All-important function to get the period distribution out of stuff that I know
    :param ages: Random samples for the age of the system (Myr)
    :param P0_min, P0_max: The minimum and maximum values of P0, the initial period. (days)
                           We will choose random values in equal log-spacing.
    :param T_star: The temperature of the star, in Kelvin
    :keyword N_age, N_P0: The number of samples to take in age and initial period
    :keyword k_C, k_I: The parameters fit in Barnes 2010
    """

    # Convert temperature to convection turnover timescale
    tau = teff2tau(T_star)
    period_list = []
    for age in ages:
        P0_vals = 10**np.random.uniform(np.log10(P0_min), np.log10(P0_max), size=N_P0)
        for P0 in P0_vals:
            out = minimize_scalar(lnlike, bracket=[1.0, 5.0], bounds=[0.1, 100], method='golden',
                                  args=(P0, age, tau, k_C, k_I))
            period_list.append(out.x)

    return np.array(period_list)


def get_vsini_pdf(T_sec, age, age_err=None, P0_min=0.1, P0_max=5, N_age=1000, N_P0=1000, k_C=0.646, k_I=452):
    """
    Get the probability distribution function of vsini for a star of the given temperature, at the given age
    :param T_sec: float - the temperature of the companion
    :param age: float, or numpy array: If a numpy array, should be a bunch of random samples of the age using
                whatever PDF you want for the age. Could be MCMC samples, for instance. If a float, this
                function will generate N_age samples from a gaussian distribution with mean age and
                standard deviation age_err. Units: Myr
    :param age_err: float - Only used if age is a float or length-1 array. Gives the standard deviation
                    of the gaussian with which we will draw ages.
    :param P0_min: float - The minimum inital period (days). Should be near the breakup velocity of the star.
    :param P0_max: float - The maximum initial period, in days. Generally, should be of order 1-10.
    :param N_age: The number of age samples to draw. Ignored if age is an numpy array with size > 1
    :param N_P0: The number of initial period samples to draw.
    :param k_C: The parameter fit from Barnes (2010). Probably shouldn't change this...
    :param k_I: The parameter fit from Barnes (2010). Probably shouldn't change this...
    :return: A numpy array with samples of the vsini for the star
    """

    # Figure out what to do with age
    if isinstance(age, float) or (isinstance(age, np.ndarray) and age.size == 1):
        if age_err is None:
            raise ValueError('Must either give several samples of age, or an age error!')
        age = np.random.normal(loc=age, scale=age_err, size=N_age)

    # Get the period distribution
    periods = get_period_dist(age, P0_min, P0_max, T_sec, N_P0=N_P0, k_C=k_C, k_I=k_I)

    # Convert to an equatorial velocity distribution by using the radius
    R = teff2radius(T_sec)
    v_eq = 2.0*np.pi*R*constants.R_sun/(periods*u.day)

    # Finally, sample random inclinations to get a distribution of vsini
    vsini = v_eq.to(u.km/u.s) * np.sin(np.random.uniform(0, np.pi/2.0, size=periods.size))

    return vsini


def read_detection_rate(infilename):
    """
    Read in the detection rate information for a given star/date combination.
    :param infilename: The file to read in. It can be one of two things:
                       1. A csv file (must have the extension .csv) containing the information for all the stars.
                          This is the file output as Sensitivity_Dataframe.csv in Sensitivity.analyze_sensitivity.
                       2. An hdf5 file containing the raw sensitivity information.
    :return: a pandas dataframe with the detection rate and average significance
             for all star/date combinations the user chooses.
    """

    # Read in the data, however it was stored.
    if infilename.endswith('csv'):
        df = pd.read_csv(infilename)
    else:
        # Assume an HDF5 file. Eventually, I should have it throw an informative error...
        df = Sensitivity.read_hdf5(infilename)
        df.to_csv('temp.csv', index=False)

    # Group by primary star, date observed, and the way the CCFs were added.
    groups = df.groupby(('star', 'date', 'addmode', 'primary SpT'))

    # Have the user choose which groups to analyze
    for i, key in enumerate(groups.groups.keys()):
        print('[{}]: {}'.format(i + 1, key))
    inp = raw_input('Enter the numbers of the keys you want to plot (, or - delimited): ')
    chosen = Sensitivity.parse_input(inp)
    keys = [k for i, k in enumerate(groups.groups.keys()) if i + 1 in chosen]

    # Compile dataframes for each star
    dataframes = defaultdict(lambda: defaultdict(pd.DataFrame))
    for key in keys:
        g = groups.get_group(key)
        detrate = g.groupby(('temperature', 'vsini', 'logL', 'contrast')).apply(
            lambda df: float(sum(df.significance.notnull())) / float(len(df)))
        significance = g.groupby(('temperature', 'vsini', 'logL', 'contrast')).apply(
            lambda df: np.nanmean(df.significance))
        dataframes['detrate'][key] = detrate.reset_index().rename(columns={0: 'detection rate'})
        dataframes['significance'][key] = significance.reset_index().rename(columns={0: 'significance'})

    return dataframes


def marginalize_vsini(df, age, age_err=None, P0_min=0.1, P0_max=5, N_age=100, N_P0=100, k_C=0.646, k_I=452):
    """
    Get the detection rate as a function of temperature by marginalizing over vsini.
    :param df: A dataframe with keys 'temperature', 'vsini', and 'detection rate' (at least)
    :param age: float, or numpy array: If a numpy array, should be a bunch of random samples of the age using
                whatever PDF you want for the age. Could be MCMC samples, for instance. If a float, this
                function will generate N_age samples from a gaussian distribution with mean age and
                standard deviation age_err. Units: Myr
    :param age_err: float - Only used if age is a float or length-1 array. Gives the standard deviation
                    of the gaussian with which we will draw ages.
    :param P0_min: float - The minimum inital period (days). Should be near the breakup velocity of the star.
    :param P0_max: float - The maximum initial period, in days. Generally, should be of order 1-10.
    :param N_age: The number of age samples to draw. Ignored if age is an numpy array with size > 1
    :param N_P0: The number of initial period samples to draw.
    :param k_C: The parameter fit from Barnes (2010). Probably shouldn't change this...
    :param k_I: The parameter fit from Barnes (2010). Probably shouldn't change this...
    :return: A pandas dataframe with keys 'temperature' and 'detection rate'
    """

    # Make sure the dataframe is all floats
    df = df.astype(float)

    # Get the vsini pdf for each temperature
    T_vals = df['temperature'].unique()
    vsini_pdfs = defaultdict(np.ndarray)
    T_grid_list = []
    vsini_grid_list = []
    for T in T_vals:
        logging.info('Generating vsini pdf for T = {} K'.format(T))
        vsini_pdfs[T] = get_vsini_pdf(T, age, age_err, P0_min, P0_max, N_age, N_P0, k_C, k_I)
        T_grid_list.append(np.ones(vsini_pdfs[T].size) * T)
        vsini_grid_list.append(vsini_pdfs[T])

    # Interpolate the detection grid at all temperatures and vsini values in the posterior
    X = (df.temperature.values, df.vsini.values)
    T_vals = np.hstack(T_grid_list)
    vsini_vals = np.hstack(vsini_grid_list)
    X_predict = np.array((T_vals, vsini_vals)).T
    detprob = griddata(X, df['detection rate'].values, X_predict)

    # Turn the predicted stuff into a dataframe
    predicted = pd.DataFrame(data={'temperature': X_predict[:, 0],
                                   'vsini': X_predict[:, 1],
                                   'detection rate': detprob})

    # Marginalize over vsini
    marginalized = predicted.groupby('temperature').apply(np.mean)

    return marginalized.rename(columns={'vsini': 'mean vsini'})[['detection rate', 'mean vsini']].reset_index()



def get_ages(starname, N_age=100):
    """
    Get age samples for the given star. This function will first try to sample from
    the posterior PDFs that Trevor David gave me (from the paper Trevor & Hillenbrand (2015)).

    If my star is not in there, it will try to uniformly sample from the main sequence lifetime of the star.
    :param starname: The name of the star. To find in the posteriod data, it must be of the form HIPNNNNN
                     (giving the hipparchos number)
    :param N_age: The number of age samples to take
    :return: a numpy array of random samples from the age of the star
    """
    post_dir = '{}/Dropbox/School/Research/AstarStuff/TargetLists/David_and_Hillenbrand2015/'.format(home)
    file_dict = {s.split('-')[0]: '{}{}'.format(post_dir, s) for s in os.listdir(post_dir) if s.endswith('age.txt')}
    if starname in file_dict:
        logging.info('Found posterior age distribution: {}'.format(file_dict[starname]))
        # Read in the PDF, cut out the very small values, and interpolate
        t, P = np.loadtxt(file_dict[starname], unpack=True)
        t = t[P>P.max()/1e3]
        P = P[P>P.max()/1e3]
        pdf = spline(t, P)

        # Make an array to sample from. It should be about 100x more points than N_age
        # to ensure the discretization doesn't do bad things.
        t_arr = np.linspace(t.min(), t.max(), N_age*100)
        dens = pdf(t_arr)
        dens[dens<0] = 0.0

        # Sample N_age samples from the posterior distribution
        samples = 10**np.random.choice(t_arr, size=N_age, p=dens/dens.sum()) / 1e6

    else:
        logging.info('Using main sequence age')
        # This star was not in the DH2015 sample. Just use the main sequence lifetime
        data = StarData.GetData(starname)
        spt = data.spectype
        sptnum = MS.SpT_To_Number(spt)

        # Get the age from the Mamajek table
        mt = Mamajek_Table.MamajekTable()
        fcn = mt.get_interpolator('SpTNum', 'logAge')
        ms_age = 10**fcn(sptnum) / 1e6
        logging.info('Main Sequence age for {} is {:.0f} Myr'.format(starname, ms_age))

        # Sample from 0 to the ms_age
        samples = np.random.uniform(0, ms_age, size=N_age)

    return samples
