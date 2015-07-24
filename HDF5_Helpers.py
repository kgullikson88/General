import logging
import os
import warnings
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

import Analyze_CCF
from GenericSmooth import roundodd
from HelperFunctions import mad, integral
import CCF_Systematics
import Fitters


home = os.environ['HOME']


def create_group(current, name, attrs, overwrite):
    if name in current:
        if not overwrite:
            return current[name]

        # Update the attributes
        for k in attrs:
            current[name].attrs[k] = attrs[k]
        return current[name]

    group = current.create_group(name)
    for k in attrs:
        group.attrs[k] = attrs[k]
    return group


def create_dataset(group, name, attrs, data, overwrite, **kwargs):
    if name in group:
        new_ds = group[name]
        if not overwrite:
            return new_ds

        new_ds.resize(data.shape)
        new_ds[:] = data

        # Update the attributes
        for k in attrs:
            new_ds.attrs[k] = attrs[k]
        return new_ds

    new_ds = group.create_dataset(data=data, name=name, **kwargs)
    for k in attrs:
        new_ds.attrs[k] = attrs[k]
    return new_ds


def combine_hdf5_synthetic(file_list, output_file, overwrite=True):
    """
    Combine several hdf5 files into one. The structure is assumed to be that of the synthetic binary search
    :param file_list: A list containing the filenames of the hdf5 files to combine
    :param output_file: The name of the file to output with the combined data
    :param overwrite: If True, it overwrites any duplicated datasets.
                      The last hdf5 file in the file_list will not be overwritten.
    :return: None
    """
    with h5py.File(output_file, 'w') as output:
        # Loop over the files in file_list
        for fname in file_list:
            with h5py.File(fname, 'r') as f:
                logging.debug('\n\nFile {}'.format(fname))
                # Primary star
                for p_name, primary in f.iteritems():
                    logging.debug('Primary {}'.format(p_name))
                    p = create_group(output, p_name, primary.attrs, overwrite)

                    # Secondary star
                    for s_name, secondary in primary.iteritems():
                        if 'bright' in s_name:
                            logging.warn('Ignoring entry {}!'.format(s_name))
                            continue
                        logging.debug('\tSecondary {}'.format(s_name))
                        s = create_group(p, s_name, secondary.attrs, overwrite)

                        # Add mode
                        for mode, mode_group in secondary.iteritems():
                            m = create_group(s, mode, mode_group.attrs, overwrite)

                            # Loop over datasets
                            for ds_name, ds in mode_group.iteritems():
                                # Make a more informative dataset name
                                ds_name = 'T{}_logg{}_metal{:+.1f}_vsini{}'.format(ds.attrs['T'],
                                                                                   ds.attrs['logg'],
                                                                                   ds.attrs['[Fe/H]'],
                                                                                   ds.attrs['vsini'])

                                # Dataset attributes should not be big things like arrays.
                                if 'velocity' in ds.attrs:
                                    data = np.array((ds.attrs['velocity'], ds.value))
                                else:
                                    data = ds.value

                                # Make attributes dictionary
                                attrs = {k: ds.attrs[k] for k in ['T', 'logg', '[Fe/H]', 'vsini']}

                                new_ds = create_dataset(m, ds_name, attrs, data, overwrite,
                                                        chunks=True, maxshape=(2, None))

                f.flush()


def combine_hdf5_real(file_list, output_file, overwrite=True):
    """
    Combine several hdf5 files into one. The structure is assumed to be that of the normal binary search
    :param file_list: A list containing the filenames of the hdf5 files to combine
    :param output_file: The name of the file to output with the combined data
    :param overwrite: If True, it overwrites any duplicated datasets.
                      The last hdf5 file in the file_list will not be overwritten.
    :return: None
    """
    with h5py.File(output_file, 'w') as output:
        # Loop over the files in file_list
        for fname in file_list:
            with h5py.File(fname, 'r') as f:
                logging.debug('\n\nFile {}'.format(fname))
                # Star name
                for star_name, star in f.iteritems():
                    logging.debug('Star {}'.format(star_name))
                    s = create_group(output, star_name, star.attrs, overwrite)

                    # Date
                    for date_str, date in star.iteritems():
                        logging.debug('\tDate {}'.format(date_str))
                        d = create_group(s, date_str, date.attrs, overwrite)

                        # Loop over datasets
                        for ds_name, ds in date.iteritems():
                            # Make a more informative dataset name
                            ds_name = 'T{}_logg{}_metal{:+.1f}_vsini{}_mode-{}'.format(ds.attrs['T'],
                                                                                       ds.attrs['logg'],
                                                                                       ds.attrs['[Fe/H]'],
                                                                                       ds.attrs['vsini'],
                                                                                       ds.attrs['addmode'])

                            # Dataset attributes should not be big things like arrays.
                            if 'velocity' in ds.attrs:
                                data = np.array((ds.attrs['velocity'], ds.value))
                            else:
                                data = ds.value

                            # Make attributes dictionary
                            attrs = {k: ds.attrs[k] for k in ['T', 'logg', '[Fe/H]', 'vsini', 'addmode']}

                            new_ds = create_dataset(d, ds_name, attrs, data, overwrite,
                                                        chunks=True, maxshape=(2, None))

                f.flush()


def combine_hdf5_sensitivity(file_list, output_file='tmp.fits', overwrite=True):
    """
    Combine several hdf5 files into one. The structure is assumed to be that of the sensitivity analysis
    :param file_list: A list containing the filenames of the hdf5 files to combine
    :param output_file: The name of the file to output with the combined data
    :param overwrite: If True, it overwrites any duplicated datasets.
                      The last hdf5 file in the file_list will not be overwritten.
    :return: None
    """
    with h5py.File(output_file, 'w') as output:
        # Loop over the files in file_list
        for fname in file_list:
            with h5py.File(fname, 'r') as f:
                logging.debug('\n\nFile {}'.format(fname))
                # Star name
                for star_name, star in f.iteritems():
                    logging.debug('Star {}'.format(star_name))
                    s = create_group(output, star_name, star.attrs, overwrite)

                    # Date
                    for date_str, date in star.iteritems():
                        logging.debug('\tDate {}'.format(date_str))
                        d = create_group(s, date_str, date.attrs, overwrite)

                        # Temperature
                        for T_string, Teff in date.iteritems():
                            logging.debug('\t\tT = {}'.format(T_string))
                            T = create_group(d, T_string, Teff.attrs, overwrite)

                            # Loop over datasets
                            for ds_name, ds in Teff.iteritems():
                                # Make a more informative dataset name
                                ds_name = 'logg{}_metal{:+.1f}_vsini{}_rv{:+.0f}_mode-{}'.format(ds.attrs['logg'],
                                                                                                 ds.attrs['[Fe/H]'],
                                                                                                 ds.attrs['vsini'],
                                                                                                 ds.attrs['rv'],
                                                                                                 ds.attrs['addmode'])

                                # Dataset attributes should not be big things like arrays.
                                if 'velocity' in ds.attrs:
                                    data = np.array((ds.attrs['velocity'], ds.value))
                                else:
                                    data = ds.value

                                # Make attributes dictionary
                                attrs = {k: ds.attrs[k] for k in ['logg', '[Fe/H]', 'vsini', 'rv', 'addmode',
                                                                  'detected', 'significance']}

                                new_ds = create_dataset(T, ds_name, attrs, data, overwrite,
                                                        chunks=True, maxshape=(2, None))

                f.flush()



class Full_CCF_Interface(object):
    """
    Interface to all of my cross-correlation functions in one class!
    """

    def __init__(self):
        # Instance variables to hold the ccf interfaces
        self._ccf_files = {'TS23': '{}/School/Research/McDonaldData/Cross_correlations/CCF.hdf5'.format(home),
                           'HET': '{}/School/Research/HET_data/Cross_correlations/CCF.hdf5'.format(home),
                           'CHIRON': '{}/School/Research/CHIRON_data/Cross_correlations/CCF.hdf5'.format(home),
                           'IGRINS': '{}/School/Research/IGRINS_data/Cross_correlations/CCF.hdf5'.format(home)}
        self._interfaces = {inst: Analyze_CCF.CCF_Interface(self._ccf_files[inst]) for inst in self._ccf_files.keys()}

        # Variables for correcting measured --> actual temperatures
        self._caldir = {'TS23': '{}/School/Research/McDonaldData/SyntheticData/'.format(home),
                       'HET': '{}/School/Research/HET_data/SyntheticData/'.format(home),
                       'CHIRON': '{}/School/Research/CHIRON_data/SyntheticData/'.format(home),
                       'IGRINS': '{}/School/Research/IGRINS_data/SyntheticData/'.format(home)}
        self._fitters = {'TS23': CCF_Systematics.Bayesian_LS,
                         'HET': CCF_Systematics.Bayesian_LS,
                         'CHIRON': CCF_Systematics.Bayesian_LS,
                         'IGRINS': Fitters.Bayesian_LS}
        self._flatchain_format = '{directory}{instrument}_{addmode}_flatchain.npy'
        self._flatlnprob_format = '{directory}{instrument}_{addmode}_flatlnprob.npy'
        self._uncertainty_scale = '{directory}{instrument}_{addmode}uncertainty_scalefactor.txt'

        # Make a couple data caches to speed things up
        self._chainCache = {}
        self._predictionCache = {}

        return

    def list_stars(self, print2screen=False):
        """
        List all of the stars in all of the CCF interfaces
        """
        stars = []
        for inst in self._interfaces.keys():
            if print2screen:
                print('Stars observed with {}: \n============================\n\n'.format(inst))
            stars.extend(self._interfaces[inst].list_stars(print2screen=print2screen))

        return list(pd.unique(stars))



    def get_observations(self, starname, print2screen=False):
        """
        Return a list of all observations of the given star
        :param starname:
        :return:
        """
        observations = []
        for instrument in self._interfaces.keys():
            interface = self._interfaces[instrument]
            if starname in interface.list_stars():
                for date in interface.list_dates(starname):
                    observations.append((instrument, date))
                    if print2screen:
                        print('{}   /   {}'.format(instrument, date))
        return observations

    def get_ccfs(self, instrument, starname, date, addmode='simple'):
        """
        Get a pandas dataframe with all the cross-correlation functions for the given instrument, star, and date
        :param instrument:
        :param starname:
        :param date:
        :return:
        """
        interface = self._interfaces[instrument]
        data = interface._compile_data(starname, date, addmode=addmode)
        data['maxidx'] = data.ccf.map(np.argmax)
        data['ccf_max'] = data.apply(lambda r: r.ccf[r.maxidx], axis=1)
        data['vel_max'] = interface.velocities[data.maxidx]
        data['vel'] = [interface.velocities] * len(data)
        #data.rename(columns={'[Fe/H]': 'feh'}, inplace=True)

        return data




    def get_measured_temperature(self, starname, date, Tmax, instrument=None, N=7, addmode='simple'):
        """
        Get the measured temperature by doing a weighted sum over temperatures near the given one (which I find by hand)
        :param starname: The name of the star
        :param Tmax: The temperature to search near
        :param date: The date the observation was taken
        :param instrument: The instrument used (this function automatically finds it if not given)
        :param N:  The number of temperature points to take
        :param addmode: The way the individual order CCFs were co-added.
        :return: A pandas DataFrame with the starname, date, instrument, and model parameters for the
                 temperatures near the requested one
        """
        if instrument is None:
            # Find this star/date in all of the interfaces
            found = False
            df_list = []
            for inst in self._interfaces.keys():
                interface = self._interfaces[inst]
                if starname in interface.list_stars() and date in interface.list_dates(starname):
                    found = True
                    df = self.get_measured_temperature(starname, date, Tmax, instrument=inst, N=N)
                    df_list.append(df)
            if not found:
                warnings.warn('Star ({}) not found for date ({}) in any CCF interfaces!'.format(starname, date))
                return None
            return pd.concat(df_list, ignore_index=True)

        # Check that the star/date combination are in the requested interface
        if starname not in self._interfaces[instrument].list_stars():
            raise KeyError('Star ({}) not in instrument ({})'.format(starname, instrument))
        if date not in self._interfaces[instrument].list_dates(starname):
            # Try date +/- 1 before failing (in case of civil/UT date mismatch or something)
            year, month, day = date.split('-')
            day = int(day)
            for inc in [-1, 1]:
                test_date = '{}-{}-{:02d}'.format(year, month, day + inc)
                if test_date in self._interfaces[instrument].list_dates(starname):
                    return self.get_measured_temperature(starname, test_date, Tmax,
                                                         instrument=instrument, N=N, addmode=addmode)
            raise KeyError(
                'Date ({}) not in CCF interface for star {} and instrument {}'.format(date, starname, instrument))

        # Get CCF information from the requested instrument/star/date combo
        interface = self._interfaces[instrument]
        logging.info(starname, date, instrument, addmode)
        df = interface._compile_data(starname=starname, date=date, addmode=addmode)
        df['ccf_max'] = df.ccf.map(np.max)

        # Get the parameters and RV of the CCF with the highest peak (which has temperature = Tmax)
        requested = df.loc[df['T'] == Tmax]
        best = requested.loc[requested.ccf_max == requested.ccf_max.max()]
        vsini = best['vsini'].item()
        metal = best['[Fe/H]'].item()
        logg = best['logg'].item()
        idx = np.argmax(best['ccf'].item())
        rv = interface.velocities[idx]

        # Now, get the CCF height for the N/2 temperatures on either side of Tmax
        N = roundodd(N)
        d = defaultdict(list)
        for T in np.arange(Tmax - 100 * (N - 1) / 2, Tmax + 100 * (N - 1) / 2 + 1, 100):
            requested = df.loc[(df['T'] == T) & (df.vsini == vsini) &
                               (df['[Fe/H]'] == metal) & (df.logg == logg)]
            if len(requested) == 0:
                warnings.warn('No matches for T = {} with star/date = {}/{}!'.format(T, starname, date))
                d['Star'].append(starname)
                d['Date'].append(date)
                d['Instrument'].append(instrument)
                d['Temperature'].append(T)
                d['vsini'].append(vsini)
                d['logg'].append(logg)
                d['[Fe/H]'].append(metal)
                d['rv'].append(rv)
                d['CCF'].append(np.nan)
                d['significance'].append(np.nan)
                continue

            # Save the best parameters for this temperature
            d['Star'].append(starname)
            d['Date'].append(date)
            d['Instrument'].append(instrument)
            d['Temperature'].append(T)
            d['vsini'].append(requested['vsini'].item())
            d['logg'].append(requested['logg'].item())
            d['[Fe/H]'].append(requested['[Fe/H]'].item())
            idx = np.argmin(np.abs(interface.velocities - rv))
            d['rv'].append(rv)
            d['CCF'].append(requested['ccf'].item()[idx])

            # Measure the detection significance
            std = mad(requested.ccf.item())
            mean = np.median(requested.ccf.item())
            d['significance'].append((d['CCF'][-1] - mean) / std)

        # Do the weighted sum.
        summary = CCF_Systematics.get_Tmeas(pd.DataFrame(data=d), include_actual=False)

        # Put the star, date, and instrument back in the dataframe before returning
        summary['Star'] = starname
        summary['Date'] = date
        summary['Instrument'] = instrument
        summary['addmode'] = addmode

        return summary


    def _correct(self, df, cache=True):
        """
        This function is called by convert_measured_to_actual and is NOT meant to be called directly!
        It takes a pandas dataframe that all have the same star
        """

        # Group by instrument and addmode, and get the PDF for the actual temperature for each
        groups = df.groupby(('Instrument', 'addmode'))
        Probabilities = []
        for (inst, addmode), group in groups:
            # Make a fitter instance
            d = {'instrument': inst,
                 'directory': self._caldir[inst],
                 'addmode': addmode}
            key = (inst, addmode)

            fitter = self._fitters[inst]()

            # get/set the cache
            if key in self._chainCache:
                chain, probs = self._chainCache[key]
                Tpredictions = self._predictionCache[key]
                fitter.spoof_sampler(chain, probs)
            else:
                chain = np.loadtxt(self._flatchain_format.format(**d))
                probs = np.loadtxt(self._flatlnprob_format.format(**d))
                fitter.spoof_sampler(chain, probs)

                Ta_arr = np.arange(2000, 10000, 1.0)
                Tmeas_pred = fitter.predict(Ta_arr, N=10000)
                Tpredictions = pd.DataFrame(Tmeas_pred, columns=Ta_arr)

                if cache:
                    self._chainCache[key] = (chain, probs)
                    self._predictionCache[key] = Tpredictions

            # Get the PDF (probability distribution function)
            Tmeas = group['Tmeas'].values
            Tmeas_err = group['Tmeas_err'].values
            for Tm, Tm_err in zip(Tmeas, Tmeas_err):
                temperature, probability = CCF_Systematics.get_actual_temperature(fitter, Tm, Tm_err,
                                                                                  cache=Tpredictions,
                                                                                  summarize=False)
                Probabilities.append(probability / probability.sum())

        # Multiply all the PDFs
        Prob = np.array(Probabilities).prod(axis=0)

        # Summarize the PDF (similar interface to np.percentile)
        l, m, h = integral(temperature, Prob, [0.16, 0.5, 0.84], k=0)

        # Save in a Pandas DataFrame and return
        return pd.DataFrame(data={'Star': df['Star'].values[0], 'Corrected_Temperature': m,
                                  'T_uperr': h - m, 'T_lowerr': m - l}, index=[0])


    def convert_measured_to_actual(self, df, cache=True):
        """
        Convert a dataframe with measured values into actual temperatures using the MCMC sample calibrations
        """

        # Correct for the systematics
        corrected = df.groupby(('Star')).apply(lambda d: self._correct(d, cache=cache))

        """
        # Correct the uncertainty
        corrections = {}
        for inst in df['Instrument'].unique():
            for addmode in df['addmode'].unique():
                d = {'instrument': inst,
                     'directory': self._caldir[inst],
                     'addmode': addmode}
                corrections[(inst, addmode)] = float(np.loadtxt(self._uncertainty_scale.format(*d)))
        corrected['Scaled_T_uperr'] = corrected.apply(lambda r: r['T_uperr'] * corrections[(r['Instrument'], r['addmode'])], axis=1)
        corrected['Scaled_T_lowerr'] = corrected.apply(lambda r: r['T_lowerr'] * corrections[(r['Instrument'], r['addmode'])], axis=1)
        """

        return corrected

