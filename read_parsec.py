import pandas
import numpy as np
import os
import warnings


DEFAULT_GRID_LOC = "{}/Dropbox/School/Research/Stellar_Evolution/Padova_Tracks.dat".format(os.environ["HOME"])


def read_full_iso(grid_loc=DEFAULT_GRID_LOC):
    names=['junk', 'Z', 'logAge', 'M_ini', 'Mass', 'logL', 'logT', 'logg',
           'Mbol', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'int_IMF', 'stage']
    data = pandas.read_csv(grid_loc, sep="\t", skipinitialspace=True, comment="#", header=None, names=names)
    return data


def get_isochrone(iso, age, warn=True):
    """
    Given a full isochrone (such as read by 'read_full_iso'), pull the requested age out.
    If the requested age is not available, a warning will be issued and the closest age will
    be used.
    :param iso: a pandas DataFrame containing the full isochrone
    :param age: The age (in Myrs) of the isochrone you want
    :param warn: If True (the default), it will warn the user if the requested age does not exist
    :return: A pandas DataFrame containing only the requested age
    """
    age = np.log10(age)
    ages = iso['logAge'].drop_duplicates()
    idx = np.argmin(abs(ages - age))
    if warn and abs(ages[idx] - age) > 1e-5:
        warnings.warn("The requested age ({:.1f} Myr) is not available. "
                      "Returning the next-closest age ({:.1f} Myr)".format(10**age / 1e6,
                                                                           10**ages[idx] / 1e6))

    return iso[iso['logAge'] == ages[idx]]


