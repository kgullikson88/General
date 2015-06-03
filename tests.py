"""
This module contains a few tests for my core code. Far from complete!
"""
from __future__ import print_function
import FittingUtilities
import logging

import numpy as np
from astropy import units as u, constants
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import lmfit

import StellarModel
import DataStructures
import Correlate
import HelperFunctions


logging.basicConfig(level=logging.DEBUG)


# Get a bunch of stellar models
def get_models(Tlist, x0=635., x1=675., logspace=True):
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


def make_synthetic_observation(model, rv, snr, x0, x1, dx=0.0):
    c = constants.c.to(u.km / u.s).value
    fcn = spline(model.x, model.y / model.cont)
    left = np.searchsorted(model.x, x0)
    right = np.searchsorted(model.x, x1)
    x = model.x[left:right] + dx
    d_logx = np.log(model.x[1] / model.x[0])
    x = np.exp(np.arange(np.log(model.x[left] + dx), np.log(model.x[right] + dx), d_logx))
    # x = np.logspace(np.log(model.x[left]+dx), np.log(model.x[right]+dx), right - left + 1, base=np.e)
    print((model.x[left:right] + dx) - x)
    data = DataStructures.xypoint(x=x, y=fcn(x * (1 + rv / c)))
    data.y += np.random.normal(loc=0, scale=1.0 / snr, size=data.size())
    return data


def test_rv(tol=0.1, T=5000, x0=610., x1=680., N_rv=10, dx=0.0):
    """
    Make a few synthetic observations with different RV shifts,
    and check to make sure that Correlate gives the right answer
    :return:
    """
    models = get_models([T], x0=x0, x1=x1)

    # Set up the "true" values
    RV_list = np.random.uniform(-100, 100, size=N_rv)
    snr = 100.0  # signal-to-noise ratio

    # Generate an lmfit modeler
    mod = lmfit.Model(HelperFunctions.Gauss)

    offsets = []
    for RV in RV_list:
        # Get the model and synthetic data
        model = models[T].copy()
        data = make_synthetic_observation(model=model.copy(), rv=RV, snr=snr, x0=x0 + 5, x1=x1 - 5, dx=dx)

        # make the cross-correlation function
        ccf = Correlate.Correlate([data], [model], addmode='simple')

        # Check the radial velocity
        idx = np.argmax(ccf.y)
        fitresult = mod.fit(ccf.y, x=ccf.x, mu=ccf.x[idx], amp=ccf.y[idx], sigma=4)
        RV_measured = fitresult.best_values['mu']
        logging.info('True RV =\t{:.2f} km/s\nMeasured RV =\t{:.2f} km/s'.format(RV, RV_measured))
        logging.info('Difference = {:.2f} km/s\n'.format(RV - RV_measured))
        offsets.append(RV - RV_measured)

    logging.info('Mean offset = {:.3g} +/- {:.3g} km/s'.format(np.mean(offsets), np.std(offsets)))

    assert abs(np.mean(offsets)) < tol
