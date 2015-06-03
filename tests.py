"""
This module contains a few tests for my core code. Far from complete!
"""
from __future__ import print_function
import FittingUtilities
import logging

import numpy as np
from astropy import units as u, constants
from scipy.interpolate import InterpolatedUnivariateSpline as spline

import StellarModel
import DataStructures
import Correlate


# Get a bunch of stellar models
def get_models(Tlist, x0=635., x1=675.):
    hdf_file = StellarModel.HDF5_FILE.replace('Search_Grid', 'CHIRON_Grid')
    model_list = StellarModel.GetModelList(metal=[0],
                                           logg=[4.5],
                                           temperature=Tlist,
                                           type='hdf5',
                                           hdf5_file=hdf_file)
    model_dict, _ = StellarModel.MakeModelDicts(model_list, type='hdf5', vsini_values=[1], hdf5_file=hdf_file)

    # Get the xgrid from the first model
    model = model_dict[Tlist[0]][4.5][0.0][0.0][1]
    left = np.searchsorted(model.x, x0)
    right = np.searchsorted(model.x, x1)
    xgrid = model.x[left:right]

    retdict = {}
    for T in Tlist:
        logging.info('Processing model with T = {}'.format(T))
        model = FittingUtilities.RebinData(model_dict[T][4.5][0.0][0.0][1], xgrid)
        model.cont = FittingUtilities.Continuum(model.x, model.y, fitorder=2, lowreject=1.5, highreject=10)
        retdict[T] = model.copy()
    return retdict


def make_synthetic_observation(model, rv, snr, x0, x1):
    c = constants.c.to(u.km / u.s).value
    fcn = spline(model.x, model.y / model.cont)
    left = np.searchsorted(model.x, x0)
    right = np.searchsorted(model.x, x1)
    x = model.x[left:right]
    data = DataStructures.xypoint(x=x, y=fcn(x * (1 + rv / c)))
    data.y += np.random.normal(loc=c, scale=1.0 / snr, size=data.size())
    return data


def test_rv():
    """
    Make a few synthetic observations with different RV shifts,
    and check to make sure that Correlate gives the right answer
    :return:
    """
    T = 5000
    x0, x1 = 610., 680.
    models = get_models([T], x0=x0, x1=x1)

    # Set up the "true" values
    N_rv = 10.0
    RV_list = np.random.uniform(-100, 100, size=N_rv)
    snr = 100  # signal-to-noise ratio

    # Make the "data" with a bit of noise
    for RV in RV_list:
        # Get the model and synthetic data
        model = models[T].copy()
        data = make_synthetic_observation(model=model.copy(), rv=RV, snr=snr, x0=x0 + 5, x1=x1 - 5)

        # make the cross-correlation function
        ccf = Correlate.Correlate([data], [model], addmode='simple')

        # Check the radial velocity
        idx = np.argmax(ccf.y)
        RV_measured = ccf.x[idx]
        print('True RV =\t{:.2f} km/s\nMeasured RV =\t{:.2f} km/s'.format(RV, RV_measured))
        print('Difference = {:.2f} km/s\n'.format(RV - RV_measured))
