import os
import re
from collections import defaultdict
from operator import itemgetter
import logging

import pandas
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from george import kernels
import matplotlib.pyplot as plt
import numpy as np
import george
import emcee
import StarData
import SpectralTypeRelations


def classify_filename(fname, type='bright'):
    """
    Given a CCF filename, it classifies the star combination, temperature, metallicity, and vsini
    :param fname:
    :return:
    """
    # First, remove any leading directories
    fname = fname.split('/')[-1]

    # Star combination
    m1 = re.search('\.[0-9]+kps', fname)
    stars = fname[:m1.start()]
    star1 = stars.split('+')[0].replace('_', ' ')
    star2 = stars.split('+')[1].split('_{}'.format(type))[0].replace('_', ' ')

    # secondary star vsini
    vsini = float(fname[m1.start() + 1:].split('kps')[0])

    # Temperature
    m2 = re.search('[0-9]+\.0K', fname)
    temp = float(m2.group()[:-1])

    # logg
    m3 = re.search('K\+[0-9]\.[0-9]', fname)
    logg = float(m3.group()[1:])

    # metallicity
    metal = float(fname.split(str(logg))[-1])

    return star1, star2, vsini, temp, logg, metal


def get_ccf_data(basedir, primary_name=None, secondary_name=None, vel_arr=np.arange(-900.0, 900.0, 0.1), type='bright'):
    """
    Searches the given directory for CCF files, and classifies
    by star, temperature, metallicity, and vsini
    :param basedir: The directory to search for CCF files
    :keyword primary_name: Optional keyword. If given, it will only get the requested primary star data
    :keyword secondary_name: Same as primary_name, but only reads ccfs for the given secondary
    :keyword vel_arr: The velocities to interpolate each ccf at
    :return: pandas DataFrame
    """
    if not basedir.endswith('/'):
        basedir += '/'
    all_files = ['{}{}'.format(basedir, f) for f in os.listdir(basedir) if type in f.lower()]
    primary = []
    secondary = []
    vsini_values = []
    temperature = []
    gravity = []
    metallicity = []
    ccf = []
    for fname in all_files:
        star1, star2, vsini, temp, logg, metal = classify_filename(fname, type=type)
        if primary_name is not None and star1.lower() != primary_name.lower():
            continue
        if secondary_name is not None and star2.lower() != secondary_name.lower():
            continue
        vel, corr = np.loadtxt(fname, unpack=True)
        fcn = spline(vel, corr)
        ccf.append(fcn(vel_arr))
        primary.append(star1)
        secondary.append(star2)
        vsini_values.append(vsini)
        temperature.append(temp)
        gravity.append(logg)
        metallicity.append(metal)

    # Make a pandas dataframe with all this data
    df = pandas.DataFrame(data={'Primary': primary, 'Secondary': secondary, 'Temperature': temperature,
                                'vsini': vsini_values, 'logg': gravity, '[Fe/H]': metallicity, 'CCF': ccf})
    return df


def get_ccf_summary(basedir, vel_arr=np.arange(-900.0, 900.0, 0.1), velocity='highest', type='bright'):
    """
    Very similar to get_ccf_data, but does it in a way that is more memory efficient
    :param basedir: The directory to search for CCF files
    :keyword velocity: The velocity to measure the CCF at. The default is 'highest', and uses the maximum of the ccf
    :keyword vel_arr: The velocities to interpolate each ccf at
    :return: pandas DataFrame
    """
    if not basedir.endswith('/'):
        basedir += '/'
    all_files = ['{}{}'.format(basedir, f) for f in os.listdir(basedir) if type in f.lower()]
    file_dict = defaultdict(lambda: defaultdict(list))
    for fname in all_files:
        star1, star2, vsini, temp, logg, metal = classify_filename(fname, type=type)
        file_dict[star1][star2].append(fname)

    # Now, read the ccfs for each primary/secondary combo, and find the best combination
    summary_dfs = []
    for primary in file_dict.keys():
        for secondary in file_dict[primary].keys():
            data = get_ccf_data(basedir, primary_name=primary, secondary_name=secondary,
                                vel_arr=vel_arr, type=type)
            summary_dfs.append(find_best_pars(data, velocity=velocity, vel_arr=vel_arr))

    return pandas.concat(summary_dfs, ignore_index=True)


def find_best_pars(df, velocity='highest', vel_arr=np.arange(-900.0, 900.0, 0.1)):
    """
    Find the 'best-fit' parameters for each combination of primary and secondary star
    :param df: the dataframe to search in
    :keyword velocity: The velocity to measure the CCF at. The default is 'highest', and uses the maximum of the ccf
    :keyword vel_arr: The velocities to interpolate each ccf at
    :return: a dataframe with keys of primary, secondary, and the parameters
    """
    # Get the names of the primary and secondary stars
    primary_names = pandas.unique(df.Primary)
    secondary_names = pandas.unique(df.Secondary)

    # Find the ccf value at the given velocity
    if velocity == 'highest':
        fcn = lambda row: (np.max(row), vel_arr[np.argmax(row)])
        vals = df['CCF'].map(fcn)
        df['ccf_max'] = vals.map(lambda l: l[0])
        df['rv'] = vals.map(lambda l: l[1])
        # df['ccf_max'] = df['CCF'].map(np.max)
    else:
        df['ccf_max'] = df['CCF'].map(lambda arr: arr[np.argmin(np.abs(vel_arr - velocity))])

    # Find the best parameter for each combination
    d = defaultdict(list)
    for primary in primary_names:
        for secondary in secondary_names:
            good = df.loc[(df.Primary == primary) & (df.Secondary == secondary)]
            best = good.loc[good.ccf_max == good.ccf_max.max()]
            d['Primary'].append(primary)
            d['Secondary'].append(secondary)
            d['Temperature'].append(best['Temperature'].item())
            d['vsini'].append(best['vsini'].item())
            d['logg'].append(best['logg'].item())
            d['[Fe/H]'].append(best['[Fe/H]'].item())
            d['rv'].append(best['rv'].item())

    return pandas.DataFrame(data=d)


def get_detected_objects(df, tol=1.0):
    """
    Takes a summary dataframe with RV information. Finds the median rv for each star,
      and removes objects that are 'tol' km/s from the median value
    :param df: A summary dataframe, such as created by find_best_pars
    :param tol: The tolerance, in km/s, to accept an observation as detected
    :return: a dataframe containing only detected companions
    """
    secondary_names = pandas.unique(df.Secondary)
    secondary_to_rv = defaultdict(float)
    for secondary in secondary_names:
        rv = df.loc[df.Secondary == secondary]['rv'].median()
        secondary_to_rv[secondary] = rv
        print secondary, rv

    keys = df.Secondary.values
    good = df.loc[abs(df.rv.values - np.array(itemgetter(*keys)(secondary_to_rv))) < tol]
    return good


def add_actual_temperature(df, method='spt'):
    """
    Add the actual temperature to a given summary dataframe
    :param df: The dataframe to which we will add the actual secondary star temperature
    :param method: How to get the actual temperature. Options are:
                   - 'spt': Use main-sequence relationships to go from spectral type --> temperature
                   - 'excel': Use tabulated data, available in the file 'SecondaryStar_Temperatures.xls'
    :return: copy of the original dataframe, with an extra column for the secondary star temperature
    """
    # First, get a list of the secondary stars in the data
    secondary_names = pandas.unique(df.Secondary)
    secondary_to_temperature = defaultdict(float)
    secondary_to_error = defaultdict(float)

    if method.lower() == 'spt':
        MS = SpectralTypeRelations.MainSequence()
        for secondary in secondary_names:
            star_data = StarData.GetData(secondary)
            spt = star_data.spectype[0] + re.search('[0-9]\.*[0-9]*', star_data.spectype).group()
            T_sec = MS.Interpolate(MS.Temperature, spt)
            secondary_to_temperature[secondary] = T_sec

    elif method.lower() == 'excel':
        table = pandas.read_excel('SecondaryStar_Temperatures.xls', 0)
        for secondary in secondary_names:
            T_sec = table.loc[table.Star.str.lower().str.contains(secondary.strip().lower())]['Literature_Temp'].item()
            T_error = table.loc[table.Star.str.lower().str.contains(secondary.strip().lower())][
                'Literature_error'].item()
            secondary_to_temperature[secondary] = T_sec
            secondary_to_error[secondary] = T_error

    df['Tactual'] = df['Secondary'].map(lambda s: secondary_to_temperature[s])
    df['Tact_err'] = df['Secondary'].map(lambda s: secondary_to_error[s])
    return


def make_gaussian_process_samples(df):
    """
    Make a gaussian process fitting the Tactual-Tmeasured relationship
    :param df: pandas DataFrame with columns 'Temperature' (with the measured temperature)
                 and 'Tactual' (for the actual temperature)
    :return: emcee sampler instance
    """
    # First, find the uncertainties at each actual temperature
    # Tactual = df['Tactual'].values
    #Tmeasured = df['Temperature'].values
    #error = df['Tact_err'].values
    temp = df.groupby('Temperature').mean()['Tactual']
    Tmeasured = temp.keys().values
    Tactual = temp.values
    error = np.nan_to_num(df.groupby('Temperature').std(ddof=1)['Tactual'].values)
    default = np.median(error[error > 1])
    error = np.maximum(error, np.ones(error.size) * default)
    for Tm, Ta, e in zip(Tmeasured, Tactual, error):
        print Tm, Ta, e
    plt.figure(1)
    plt.errorbar(Tmeasured, Tactual, yerr=error, fmt='.k', capsize=0)
    plt.plot(Tmeasured, Tmeasured, 'r--')
    plt.xlim((min(Tmeasured) - 100, max(Tmeasured) + 100))
    plt.xlabel('Measured Temperature')
    plt.ylabel('Actual Temperature')
    plt.show(block=False)

    # Define some functions to use in the GP fit
    def model(pars, T):
        #polypars = pars[2:]
        #return np.poly1d(polypars)(T)
        return T

    def lnlike(pars, Tact, Tmeas, Terr):
        a, tau = np.exp(pars[:2])
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(Tmeas, Terr)
        return gp.lnlikelihood(Tact - model(pars, Tmeas))

    def lnprior(pars):
        lna, lntau = pars[:2]
        polypars = pars[2:]
        if -20 < lna < 20 and 4 < lntau < 20:
            return 0.0
        return -np.inf

    def lnprob(pars, x, y, yerr):
        lp = lnprior(pars)
        return lp + lnlike(pars, x, y, yerr) if np.isfinite(lp) else -np.inf

    # Set up the emcee fitter
    initial = np.array([0, 6])#, 1.0, 0.0])
    ndim = len(initial)
    nwalkers = 100
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Tactual, Tmeasured, error))

    print 'Running first burn-in'
    p1, lnp, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print "Running second burn-in..."
    p_best = p1[np.argmax(lnp)]
    p2 = [p_best + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p3, _, _ = sampler.run_mcmc(p2, 250)
    sampler.reset()

    print "Running production..."
    sampler.run_mcmc(p3, 1000)

    # Plot a bunch of the fits
    print "Plotting..."
    N = 100
    Tvalues = np.arange(3300, 7000, 20)
    idx = np.argsort(-sampler.lnprobability.flatten())[:N]  # Get N 'best' curves
    par_vals = sampler.flatchain[idx]
    for i, pars in enumerate(par_vals):
        a, tau = np.exp(pars[:2])
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(Tmeasured, error)
        s = gp.sample_conditional(Tactual - model(pars, Tmeasured), Tvalues) + model(pars, Tvalues)
        plt.plot(Tvalues, s, 'b-', alpha=0.1)
    plt.draw()

    # Finally, get posterior samples at all the possibly measured temperatures
    print 'Generating posterior samples at all temperatures...'
    N = 10000  # This is 1/10th of the total number of samples!
    idx = np.argsort(-sampler.lnprobability.flatten())[:N]  # Get N 'best' curves
    par_vals = sampler.flatchain[idx]
    Tvalues = np.arange(3000, 6900, 100)
    gp_posterior = []
    for pars in par_vals:
        a, tau = np.exp(pars[:2])
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(Tmeasured, error)
        s = gp.sample_conditional(Tactual - model(pars, Tmeasured), Tvalues) + model(pars, Tvalues)
        gp_posterior.append(s)

    # Finally, make confidence intervals for the actual temperatures
    gp_posterior = np.array(gp_posterior)
    l, m, h = np.percentile(gp_posterior, [16.0, 50.0, 84.0], axis=0)
    conf = pandas.DataFrame(data={'Measured Temperature': Tvalues, 'Actual Temperature': m,
                                  'Lower Bound': l, 'Upper bound': h})
    conf.to_csv('Confidence_Intervals.csv', index=False)

    return sampler, np.array(gp_posterior)


def check_posterior(df, posterior, Tvalues):
    """
    Checks the posterior samples: Are 95% of the measurements within 2-sigma of the prediction?
    :param df: The summary dataframe
    :param posterior: The MCMC predicted values
    :param Tvalues: The measured temperatures the posterior was made with
    :return: boolean, as well as some warning messages if applicable
    """
    # First, make 2-sigma confidence intervals
    l, m, h = np.percentile(posterior, [5.0, 50.0, 95.0], axis=0)

    # Save the confidence intervals
    # conf = pandas.DataFrame(data={'Measured Temperature': Tvalues, 'Actual Temperature': m,
    #                              'Lower Bound': l, 'Upper bound': h})
    #conf.to_csv('Confidence_Intervals.csv', index=False)

    Ntot = []  # The total number of observations with the given measured temperature
    Nacc = []  # The number that have actual temperatures within the confidence interval

    g = df.groupby('Temperature')
    for i, T in enumerate(Tvalues):
        if T in g.groups.keys():
            Ta = g.get_group(T)['Tactual']
            low, high = l[i], h[i]
            Ntot.append(len(Ta))
            Nacc.append(len(Ta.loc[(Ta >= low) & (Ta <= high)]))
            p = float(Nacc[-1]) / float(Ntot[-1])
            if p < 0.95:
                logging.warn(
                    'Only {}/{} of the samples ({:.2f}%) were accepted for T = {} K'.format(Nacc[-1], Ntot[-1], p * 100,
                                                                                            T))
                print low, high
                print sorted(Ta)
        else:
            Ntot.append(0)
            Nacc.append(0)

    p = float(sum(Nacc)) / float(sum(Ntot))
    if p < 0.95:
        logging.warn('Only {:.2f}% of the total samples were accepted!'.format(p * 100))
        return False
    return True


if __name__ == '__main__':
    pass

