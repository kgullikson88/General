import os
import re
from collections import defaultdict
from operator import itemgetter
import logging
import sys

import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from george import kernels
import matplotlib.pyplot as plt
import numpy as np
import george
import emcee
import h5py
from scipy.integrate import quad

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
    df = pd.DataFrame(data={'Primary': primary, 'Secondary': secondary, 'Temperature': temperature,
                                'vsini': vsini_values, 'logg': gravity, '[Fe/H]': metallicity, 'CCF': ccf})
    return df


def get_ccf_summary(hdf5_filename, vel_arr=np.arange(-900.0, 900.0, 0.1),
                    velocity='highest', addmode='simple', debug=False):
    """
    Goes through the given HDF5 file, and finds the best set of parameters for each combination of primary/secondary star
    :param hdf5_filename: The HDF5 file containing the CCF data
    :keyword velocity: The velocity to measure the CCF at. The default is 'highest', and uses the maximum of the ccf
    :keyword vel_arr: The velocities to interpolate each ccf at
    :keyword addmode: The way the CCF orders were added while generating the ccfs
    :keyword debug: If True, it prints the progress. Otherwise, does its work silently and takes a while
    :return: pandas DataFrame summarizing the best parameters.
             This is the type of dataframe to give to the other function here
    """
    summary_dfs = []
    with h5py.File(hdf5_filename, 'r') as f:
        primaries = f.keys()
        for p in primaries:
            if debug:
                print(p)
            secondaries = f[p].keys()
            for s in secondaries:
                if debug:
                    print('\t{}'.format(s))
                datasets = f[p][s][addmode].keys()
                vsini_values = []
                temperature = []
                gravity = []
                metallicity = []
                ccf = []
                for i, d in enumerate(datasets):
                    if debug:
                        sys.stdout.write('\r\t\tDataset {}/{}'.format(i+1, len(datasets)))
                        sys.stdout.flush()
                    ds = f[p][s][addmode][d]
                    vel, corr = ds.attrs['velocity'], ds.value
                    fcn = spline(vel, corr)
                    vsini_values.append(ds.attrs['vsini'])
                    temperature.append(ds.attrs['T'])
                    gravity.append(ds.attrs['logg'])
                    metallicity.append(ds.attrs['[Fe/H]'])
                    ccf.append(fcn(vel_arr))
                if debug:
                    print()
                data = pd.DataFrame(data={'Primary': [p]*len(ccf), 'Secondary': [s]*len(ccf),
                                          'Temperature': temperature, 'vsini': vsini_values,
                                          'logg': gravity, '[Fe/H]': metallicity, 'CCF': ccf})
                summary_dfs.append(find_best_pars(data, velocity=velocity, vel_arr=vel_arr))

    return pd.concat(summary_dfs, ignore_index=True)



def find_best_pars(df, velocity='highest', vel_arr=np.arange(-900.0, 900.0, 0.1)):
    """
    Find the 'best-fit' parameters for each combination of primary and secondary star
    :param df: the dataframe to search in
    :keyword velocity: The velocity to measure the CCF at. The default is 'highest', and uses the maximum of the ccf
    :keyword vel_arr: The velocities to interpolate each ccf at
    :return: a dataframe with keys of primary, secondary, and the parameters
    """
    # Get the names of the primary and secondary stars
    primary_names = pd.unique(df.Primary)
    secondary_names = pd.unique(df.Secondary)

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

    return pd.DataFrame(data=d)


def get_detected_objects(df, tol=1.0):
    """
    Takes a summary dataframe with RV information. Finds the median rv for each star,
      and removes objects that are more than 'tol' km/s from the median value
    :param df: A summary dataframe, such as created by get_ccf_summary or find_best_pars
    :param tol: The tolerance, in km/s, to accept an observation as detected
    :return: a dataframe containing only detected companions
    """
    secondary_names = pd.unique(df.Secondary)
    secondary_to_rv = defaultdict(float)
    for secondary in secondary_names:
        rv = df.loc[df.Secondary == secondary]['rv'].median()
        secondary_to_rv[secondary] = rv
        print secondary, rv

    keys = df.Secondary.values
    good = df.loc[abs(df.rv.values - np.array(itemgetter(*keys)(secondary_to_rv))) < tol]
    return good


def add_actual_temperature(df, method='excel', filename='SecondaryStar_Temperatures.xls'):
    """
    Add the actual temperature to a given summary dataframe
    :param df: The dataframe to which we will add the actual secondary star temperature
    :keyword method: How to get the actual temperature. Options are:
                   - 'spt': Use main-sequence relationships to go from spectral type --> temperature
                   - 'excel': Use tabulated data, available in the file 'SecondaryStar_Temperatures.xls'
    :keyword filename: The filename of the excel spreadsheet containing the literature temperatures.
                       Needs to have the right format! Ignored if method='spt'
    :return: copy of the original dataframe, with an extra column for the secondary star temperature
    """
    # First, get a list of the secondary stars in the data
    secondary_names = pd.unique(df.Secondary)
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
        table = pd.read_excel(filename, 0)
        #print(secondary_names)
        #print(table)
        print(table.keys())
        for secondary in secondary_names:
            print(secondary)
            print(table.Star)
            #print table.loc[table.Star.str.lower().str.contains(secondary.strip().lower())]
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
    Tmeasured, Tactual, error, lit_err = get_values(df)
    for i, e in enumerate(error):
        if e < 1:
            e = fit_sigma(df, i)
        error[i] = np.sqrt(e**2 + lit_err[i]**2)
    for Tm, Ta, e in zip(Tmeasured, Tactual, error):
        print Tm, Ta, e
    plt.figure(1)
    limits = [3000, 7000]
    plt.errorbar(Tmeasured, Tactual, yerr=error, fmt='.k', capsize=0)
    plt.plot(limits, limits, 'r--')
    #plt.xlim((min(Tmeasured) - 100, max(Tmeasured) + 100))
    plt.xlabel('Measured Temperature')
    plt.ylabel('Actual Temperature')
    plt.xlim(limits)
    plt.ylim(limits)
    #plt.show(block=False)

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
        if -20 < lna < 20 and 12 < lntau < 20:
            return 0.0
        return -np.inf

    def lnprob(pars, x, y, yerr):
        lp = lnprior(pars)
        return lp + lnlike(pars, x, y, yerr) if np.isfinite(lp) else -np.inf

    # Set up the emcee fitter
    initial = np.array([0, 14])#, 1.0, 0.0])
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

    # We now need to increase the spread of the posterior distribution so that it encompasses the right number of data points
    # This is because the way we have been treating error bars here is kind of funky...
    # First, generate a posterior distribution of Tactual for every possible Tmeasured
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

    # Get the median and spread in the pdf
    gp_posterior = np.array(gp_posterior)
    medians = np.median(gp_posterior, axis=0)
    sigma_pdf = np.std(gp_posterior, axis=0)

    # Correct the data and get the residual spread
    df['Corrected_Temperature'] = df['Temperature'].map(lambda T: medians[np.argmin(abs(T - Tvalues))])
    sigma_spread = np.std(df.Tactual - df.Corrected_Temperature)

    # Increase the spread in the pdf to reflect the residual spread
    ratio = np.maximum(np.ones(sigma_pdf.size), sigma_spread / sigma_pdf)
    gp_corrected = (gp_posterior - medians) * ratio + medians

    # Make confidence intervals
    l, m, h = np.percentile(gp_corrected, [16.0, 50.0, 84.0], axis=0)
    conf = pd.DataFrame(data={'Measured Temperature': Tvalues, 'Actual Temperature': m,
                              'Lower Bound': l, 'Upper bound': h})
    conf.to_csv('Confidence_Intervals.csv', index=False)


    # Finally, plot a bunch of the fits
    print "Plotting..."
    N = 300
    Tvalues = np.arange(3000, 7000, 20)
    idx = np.argsort(-sampler.lnprobability.flatten())[:N]  # Get N 'best' curves
    par_vals = sampler.flatchain[idx]
    plot_posterior = []
    for i, pars in enumerate(par_vals):
        a, tau = np.exp(pars[:2])
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(Tmeasured, error)
        s = gp.sample_conditional(Tactual - model(pars, Tmeasured), Tvalues) + model(pars, Tvalues)
        plot_posterior.append(s)
    plot_posterior = np.array(plot_posterior)
    medians = np.median(plot_posterior, axis=0)
    sigma_pdf = np.std(plot_posterior, axis=0)

    # Increase the spread in the pdf to reflect the residual spread
    ratio = np.maximum(np.ones(sigma_pdf.size), sigma_spread / sigma_pdf)
    plot_posterior = (plot_posterior - medians) * ratio + medians
    plt.plot(Tvalues, plot_posterior.T, 'b-', alpha=0.05)

    plt.draw()
    plt.savefig('Temperature_Correspondence.pdf')

    return sampler, gp_corrected


def check_posterior(df, posterior, Tvalues=np.arange(3000, 6900, 100)):
    """
    Checks the posterior samples: Are 95% of the measurements within 2-sigma of the prediction?
    :param df: The summary dataframe
    :param posterior: The MCMC predicted values
    :param Tvalues: The measured temperatures the posterior was made with
    :return: boolean, as well as some warning messages if applicable
    """
    # First, make 2-sigma confidence intervals
    l, m, h = np.percentile(posterior, [5.0, 50.0, 95.0], axis=0)

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


def get_values(df):
    temp = df.groupby('Temperature')
    Tmeasured = temp.groups.keys()
    Tactual_values = [temp.get_group(Tm)['Tactual'].values for Tm in Tmeasured]
    Tactual = np.array([np.mean(Ta) for Ta in Tactual_values])
    spread = np.nan_to_num([np.std(Ta, ddof=1) for Ta in Tactual_values])
    literr_values = [temp.get_group(Tm)['Tact_err'].values for Tm in Tmeasured]
    lit_err = np.array([np.sqrt(np.sum(literr**2)) for literr in literr_values])
    return Tmeasured, Tactual, spread, lit_err


def integrate_gauss(x1, x2, amp, mean, sigma):
    """
    Integrate a gaussian between the points x1 and x2
    """
    gauss = lambda x, A, mu, sig: A*np.exp(-(x-mu)**2 / (2.0*sig**2))
    if x1 < -1e6:
        x1 = -np.inf
    if x2 > 1e6:
        x2 = np.inf
    result = quad(gauss, x1, x2, args=(amp, mean, sigma))
    return result[0]


def get_probability(x1, x2, x3, x4, N, mean, sigma):
    """
    Get the probability of the given value of sigma
    x1-x4 are the four limits, which are the bin edges of the possible values Tactual can take
    N is the number of entries in the single bin, and mean what it sounds like
    """
    if x2 < 100:
        x2 = x3 - (x4-x3)
    if x4 > 1e6:
        x4 = x3 + (x3-x2)
    int1 = integrate_gauss(x2, x3, 1.0, mean, sigma)
    A = float(N) / int1
    int2 = 0 if x1 < 100 else integrate_gauss(x1, x2, A, mean, sigma)
    int3 = 0 if x4 > 1e6 else integrate_gauss(x3, x4, A, mean, sigma)
    if int2 > 1 or int3 > 1:
        return 0
    return 1


def fit_sigma(df, i):
    """
    Find the largest allowable standard deviation, given the possible values Tactual can take.
    """
    Tmeasured, Tactual, _, _ = get_values(df)
    Tm = Tmeasured[i]
    
    # Get the possible values, and bin those with this measured value
    possible_values = sorted(pd.unique(df.Tactual))
    edges = [(possible_values[i] + possible_values[i+1])/2 for i in range(len(possible_values)-1)]
    bins = [0] + edges + [9e9]
    good = df.loc[df.Temperature == Tm]
    values, _= np.histogram(good.Tactual.values, bins=bins)
    
    mean = np.mean(good.Tactual.values)
    std = np.std(good.Tactual.values, ddof=1)
    if std > 0:
        return std
    
    sigma_test = np.arange(500, 10, -10) #Just test a bunch of values
    idx = np.searchsorted(bins, mean)
    x1 = bins[idx-2] if idx > 2 else -1
    x2 = bins[idx-1]
    x3 = bins[idx]
    x4 = bins[idx+1] if idx < len(bins)-2 else np.inf
    N = len(good)
    probs = [get_probability(x1, x2, x3, x4, N, mean, s) for s in sigma_test]
    for s, p in zip(sigma_test, probs):
        if p > 0.5:
            return s
    
    raise ValueError('No probability > 0!')


if __name__ == '__main__':
    pass

