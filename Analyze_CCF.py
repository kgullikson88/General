"""
This is a module to read in an HDF5 file with CCFs.
Use this to determine the best parameters, and plot the best CCF for each star/date
"""
import h5py
# import DataStructures
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import InterpolatedUnivariateSpline as spline


class CCF_Interface(object):
    def __init__(self, filename, vel=np.arange(-900, 900, 1)):
        self.hdf5 = h5py.File(filename, 'r')
        self.velocities = vel


    def list_stars(self, print2screen=False):
        """
        List the stars available in the HDF5 file, and the dates available for each
        :return: A list of the stars
        """
        if print2screen:
            for star in sorted(self.hdf5.keys()):
                print(star)
                for date in sorted(self.hdf5[star].keys()):
                    print('\t{}'.format(date))
        return sorted(self.hdf5.keys())


    def list_dates(self, star, print2screen=False):
        """
        List the dates available for the given star
        :param star: The name of the star
        :return: A list of dates the star was observed
        """
        if print2screen:
            for date in sorted(self.hdf5[star].keys()):
                print(date)
        return sorted(self.hdf5[star].keys())


    def _compile_data(self, starname, date, addmode='simple'):
        """
        Private function. This reads in all the datasets for the given star and date
        :param starname: the name of the star. Must be in self.hdf5
        :param date: The date to search. Must be in self.hdf5[star]
        :keyword addmode: The way the individual CCFs were added. Options are:
                          - 'simple'
                          - 'ml'
                          - 'all'  (saves all addmodes)
        :return: a pandas DataFrame with the columns:
                  - star
                  - date
                  - temperature
                  - log(g)
                  - [Fe/H]
                  - vsini
                  - addmode
                  - rv (at maximum CCF value)
                  - CCF height (maximum)
        """
        datasets = self.hdf5[starname][date].keys()
        data = defaultdict(list)
        for ds_name in datasets:
            ds = self.hdf5[starname][date][ds_name]
            am = ds.attrs['addmode']
            if addmode == 'all' or addmode == am:
                data['T'].append(ds.attrs['T'])
                data['logg'].append(ds.attrs['logg'])
                data['[Fe/H]'].append(ds.attrs['[Fe/H]'])
                data['vsini'].append(ds.attrs['vsini'])
                data['addmode'].append(am)
                v = ds.value
                vel, corr = v[0], v[1]
                fcn = spline(vel, corr)
                data['ccf'].append(fcn(self.velocities))

        #data['Star'] = [starname] * len(data['T'])
        #data['Date'] = [date] * len(data['T'])
        df = pd.DataFrame(data=data)
        return df

    def get_temperature_run(self, starname=None, date=None, df=None):
        """
        Return the maximum ccf height for each temperature. Either starname AND date, or df must be given
        :param starname: The name of the star
        :param date: The date of the observation
        :param df: Input dataframe, such as from _compile_data. Overrides starname and date, if given
        :return: a pandas DataFrame with all the best parameters for each temperature
        """
        # Get the dataframe if it isn't given
        if df is None:
            if starname is None or date is None:
                raise ValueError('Must give either starname or date to get_temperature_run!')
            df = self._compile_data(starname, date)

        # Find the maximum CCF for each set of parameters
        fcn = lambda row: (np.max(row), self.velocities[np.argmax(row)])
        vals = df['ccf'].map(fcn)
        df['ccf_max'] = vals.map(lambda l: l[0])
        df['rv'] = vals.map(lambda l: l[1])

        # Find the best parameters for each temperature
        d = defaultdict(list)
        temperatures = pd.unique(df['T'])
        for T in temperatures:
            good = df.loc[df['T'] == T]
            best = good.loc[good.ccf_max == good.ccf_max.max()]
            d['vsini'].append(best['vsini'].item())
            d['logg'].append(best['logg'].item())
            d['[Fe/H]'].append(best['[Fe/H]'].item())
            d['rv'].append(best['rv'].item())
            d['ccf_value'].append(best.ccf_max.item())
            d['T'].append(T)
            d['metal'].append(best['[Fe/H]'].item())

        return pd.DataFrame(data=d)

    def get_ccf(self, params, df=None):
        """
        Get the ccf with the given parameters. A dataframe can be given to speed things up
        :param params: All the parameters necessary to define a single ccf
        :param starname:
        :param date:
        :param df: a pandas DataFrame such as outputted by _compile_data
        :return: ??
        """
        if df is None:
            try:
                df = self._compile_data(params['starname'], params['date'])
            except KeyError:
                raise KeyError('Must give get_ccf params with starname and date keywords, if df is not given!')

        good = df.loc[(df['T'] == params['T']) & (df.logg == params['logg']) & (df.vsini == params['vsini']) \
                      & (df['[Fe/H]'] == params['[Fe/H]']) & (df.addmode == params['addmode'])]


        return pd.DataFrame(data={'velocity': self.velocities, 'CCF': good['ccf'].item()})


