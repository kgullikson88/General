
import numpy as np
import pandas as pd
import os
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import SpectralTypeRelations

home = os.environ['HOME']
TABLE_FILENAME = '{}/Dropbox/School/Research/Databases/SpT_Relations/Mamajek_Table.txt'.format(home)

class MamajekTable(object):
    """
    Class to interact with the table that Eric mamajek has online at
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    """
    def __init__(self, filename=TABLE_FILENAME):
        MS = SpectralTypeRelations.MainSequence()

        # Read in the table.
        colspecs=[[0,7], [7,14], [14,21], [21,28], [28,34], [34,40], [40,47], [47,55],
                  [55,63], [63,70], [70,78], [78,86], [86,94], [94,103], [103,110],
                  [110,116], [116,122], [122,130], [130,137], [137,144], [144,151],
                  [151,158]]
        mam_df = pd.read_fwf(filename, header=20, colspecs=colspecs, na_values=['...'])[:92]

        # Strip the * from the logAge column. Probably should but...
        mam_df['logAge'] = mam_df['logAge'].map(lambda s: s.strip('*') if isinstance(s, basestring) else s)

        # Convert everything to floats
        self.mam_df = mam_df.convert_objects(convert_numeric=True)

        # Add the spectral type number for interpolation
        self.mam_df['SpTNum'] = mam_df['SpT'].map(MS.SpT_To_Number)


    def get_columns(self, print_keys=True):
        """
        Get the column names in a list, and optionally print them to the screen.
        :param print_keys: bool variable to decide if the keys are printed.
        :return:
        """
        if print_keys:
            for k in self.mam_df.keys():
                print k
        return list(self.mam_df.keys())

    def get_interpolator(self, xcolumn, ycolumn, extrap='nearest'):
        """
        Get an interpolator instance between the two columns
        :param xcolumn: The name of the x column to interpolate between
        :param ycolumn: The name of the value you want to interpolate
        :param extrap: How to treat extrapolation. Options are:
                       'nearest': Default behavior. It will return the nearest match to the given 'x' value
                       'extrapolate': Extrapolate the spline. This is probably only safe for very small extrapolations
        :return: an interpolator function
        """
        # Make sure the column names are correct
        assert xcolumn in self.mam_df.keys() and ycolumn in self.mam_df.keys()

        # Sort the dataframe by the x column, and drop any duplicates or nans it might have
        sorted = self.mam_df.sort(xcolumn).dropna(subset=[xcolumn, ycolumn], how='any').drop_duplicates(xcolumn)

        # Make an interpolator
        ext_value = {'nearest': 3, 'extrapolate': 0}
        fcn = spline(sorted[xcolumn].values, sorted[ycolumn].values, ext=ext_value[extrap])
        return fcn

