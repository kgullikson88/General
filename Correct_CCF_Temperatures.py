"""
This file contains a couple functions for taking a dataframe of measured temperatures
and associated starnames/instruments, and returning a corrected temperature with error bar.
The correction needed is determined in the iPython notebook 'CheckCCFSystematics'.
"""

import os

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity


# Make a cache to speed things up
home = os.environ['HOME']
rootdir = {'TS23': '{0:s}/School/Research/McDonaldData/SyntheticData'.format(home),
           'HET': '{0:s}/School/Research/HET_data/SyntheticData'.format(home),
           'CHIRON': '{0:s}/School/Research/CHIRON_data/SyntheticData'.format(home),
           'IGRINS': '{0:s}/School/Research/IGRINS_data/SyntheticData'.format(home)}
addmode = 'simple'
kernel_cache = {'TS23': {}, 'HET': {}, 'CHIRON': {}, 'IGRINS': {}}
posterior_cache = {'TS23': None, 'HET': None, 'CHIRON': None, 'IGRINS': None}


def get_kernel(instrument, Tmeasured, Tvalues=np.arange(3000, 6900, 100)):
    """
    Return the estimate of the kernel density for the given instrument and measured temperature
    :param instrument: One of 'TS23', 'HRS', 'CHIRON', or 'IGRINS'
    :param Tmeasured: The measured temperature
    :param Tvalues: All the values used in making the posterior distribution
    :return: the kernel density estimator, as well as the smallest and
               largest actual temperatures for the given measured temperature.
    """
    if instrument == 'HRS':
        instrument = 'HET'

    # Check if we already got this kernel
    if Tmeasured in kernel_cache[instrument]:
        return kernel_cache[instrument][Tmeasured]

    # Get the posterior samples
    if posterior_cache[instrument] is None:
        filename = '{}/{}_{}_Posterior_Samples.dat'.format(rootdir[instrument], instrument, addmode)
        posterior = np.loadtxt(filename).T
        posterior_cache[instrument] = posterior
    else:
        posterior = posterior_cache[instrument]

    # Get the index of the measured temperature
    idx = np.argmin(abs(Tmeasured - Tvalues))

    # Do the kernel density estimation
    X = posterior[idx][:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(X)
    kernel_cache[instrument][Tmeasured] = (kde, np.min(posterior[idx]), np.max(posterior[idx]))

    return kde, np.min(posterior[idx]), np.max(posterior[idx])


def get_real_temperature(df, addmode='simple'):
    """
    Given a dataframe of observations, find the actual temperature and uncertainty for each star
    :param df: The input dataframe. Must have the following keys:
        - 'Star'
        - '[Fe/H]'
        - 'vsini'  (which is the vsini of the secondary star)
        - 'Instrument'
        - 'Temperature' (which is the measured temperature)
    :param addmode:
    :return:
    """
    # Group by the star name. We will do error propagation with kernel density estimation
    star_groups = df.groupby('Star')
    starnames = star_groups.groups.keys()
    metal = []
    vsini = []
    temperature = []
    for starname in starnames:
        star_df = star_groups.get_group(starname)
        star_df = star_df.loc[star_df.Temperature.notnull()]
        if len(star_df) == 0:
            metal.append(np.nan)
            vsini.append(np.nan)
            temperature.append(np.nan)
            continue

        kernels = []
        low = np.inf
        high = -np.inf
        metal_values = []
        vsini_values = []
        for i, row in star_df.iterrows():
            metal_values.append(row['[Fe/H]'])
            vsini_values.append(row['vsini'])
            kde, minimum, maximum = get_kernel(row['Instrument'], row['Temperature'])
            kernels.append(kde)
            low = min(low, minimum)
            high = max(high, maximum)

        # Make some samples from the combined kernel
        x = np.linspace(low, high, 1e3)
        N = 1e6
        dens = np.ones(x.size)
        for kde in kernels:
            d = np.exp(kde.score_samples(x[:, np.newaxis]))
            # plt.plot(x, d, lw=2, label='input')
            dens *= d
        # plt.plot(x, dens, lw=2, label='final')
        #plt.legend(loc='best')
        #plt.show()
        values = np.random.choice(x, size=N, p=dens / dens.sum())

        l, m, h = np.percentile(values, [16.0, 50.0, 84.0])

        # Store in lists
        metal.append(np.mean(metal_values))
        vsini.append(np.mean(vsini_values))
        temperature.append(r'${:d}^{{+{:d}}}_{{-{:d}}}$'.format(int(m), int(h - m), int(m - l)))

    return pd.DataFrame(data={'Star': starnames, 'Temperature': temperature, '[Fe/H]': metal, 'vsini': vsini})