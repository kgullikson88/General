import logging

import h5py
import numpy as np


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


import os
import Analyze_CCF
import warnings
import pandas as pd
from GenericSmooth import roundodd
from collections import defaultdict
from HelperFunctions import mad
import CCF_Systematics

home = os.environ['HOME']


class Full_CCF_Interface(object):
    """
    Interface to all of my cross-correlation functions in one class!
    """
    _ccf_files = {'TS23': '{}/School/Research/McDonaldData/Cross_correlations/CCF.hdf5'.format(home),
                  'HET': '{}/School/Research/HET_data/Cross_correlations/CCF.hdf5'.format(home),
                  'CHIRON': '{}/School/Research/CHIRON_data/Cross_correlations/CCF.hdf5'.format(home),
                  'IGRINS': '{}/School/Research/IGRINS_data/Cross_correlations/CCF.hdf5'.format(home)}

    def __init__(self):
        self._interfaces = {inst: Analyze_CCF.CCF_Interface(self._ccf_files[inst]) for inst in self._ccf_files.keys()}


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


        # Get CCF information from the requested instrument/star/date combo
        interface = self._interfaces[instrument]
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

        summary = pd.DataFrame(data=d)

        # Finally, do the weighted sum.
        return CCF_Systematics.get_Tmeas(summary, include_actual=False)



