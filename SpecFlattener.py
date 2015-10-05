import logging
import FittingUtilities

import numpy as np
from statsmodels.robust import norms
import statsmodels.api as sm
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd

import HelperFunctions
import StellarModel
import DataStructures
from Fitters import RobustFit
import Broaden
import Correlate


clight = 3e5


class ModelContinuumFitter(object):
    """
    A class to flatten an echelle spectrum by first dividing by a model spectrum.
    The algorithm used is:

      1. Get model spectrum with the requested parameters
      2. Divide data by (normalized) model
      3. Perform a robust 2D fit to the residuals to get continuum as a function of wavelength and aperture number
      4. (optionally) Iterate on the above steps until the best model spectrum is found
    """

    def __init__(self, data_orders, model_library, x_degree=4, y_degree=6, wave_spacing=0.003,
                 T=9000, logg=4.0, feh=0.0, initialize=True, order_numbers=None, **kwargs):
        """
        Initialize the ModelContinuumFitter class.

        Parameters:
        ===========
         - data_orders:        An iterable of DataStructures.xypoint instances
                               Each entry in the iterable should contain one of the echelle apertures.
                                They should be ordered by increasing wavelength.

         - model_library:      string
                               The path (either full or relative) to an HDF5 file containing model
                               spectra of various parameters. The HDF5 file should be processed through
                               Starfish's grid utilities for a specific instrument.

         - x_degree, y_degree: floats -- defaults = (4,6)
                               The degree of the chebyshev polynomial to use when fitting the continuum.
                               x is the wavelength, and y is the aperture number.

         - wave_spacing:       float
                               The wavelength spacing, in nm, to interpolate each order onto. The input data will
                               not be interpolated, only the internal wavelength and flux arrays used in fitting.
                               Appropriate values are:

                                 - CHIRON: 0.003
                                 - TS23: ?
                                 - HRS: ?
                                 - IGRINS: ?

         - T:                  float
                               The temperature of the model to use. This can be updated with the 'update_model' method.

         - logg:               float
                               The surface gravity of the model to use.
                               This can be updated with the 'update_model' method.

         - feh:                float
                               The metallicity of the model to use. This can be updated with the 'update_model' method.

         - initialize:         boolean
                               Whether to estimate the radial velocity and vsini of the model through
                               cross-correlation functions. This MUST be run before the 'flatten_orders' method.

         - order_numbers:      iterable
                               A list of the echelle order numbers. This is only useful if using non-consecutive
                               orders for flattening (I am for HET data).

        Returns:
        ========
        None
        """

        wave_arr = np.arange(data_orders[0].x[0], data_orders[-1].x[-1], wave_spacing)
        if order_numbers is None:
            order_numbers = range(len(data_orders))
        wave, aperture, flux, flux_err = [], [], [], []

        # combine all the orders into a few different arrays
        for i, order in enumerate(data_orders):
            idx = (wave_arr >= order.x[0]) & (wave_arr <= order.x[-1])
            wave.append(wave_arr[idx])
            aperture.append(np.ones(idx.sum()) * order_numbers[i])
            rebinned = FittingUtilities.RebinData(order, wave_arr[idx])
            flux.append(rebinned.y)
            flux_err.append(rebinned.err)

        # Stack all the indices
        wave = np.hstack(wave)
        aperture = np.hstack(aperture)
        flux = np.hstack(flux)
        flux_err = np.hstack(flux_err)

        self.low_wave, self.high_wave = np.min(wave), np.max(wave)
        self.low_ap, self.high_ap = np.min(aperture), np.max(aperture)
        self.low_flux, self.high_flux = np.min(flux), np.max(flux)

        self.x = dict(wave=wave, order=aperture)
        self.y = flux
        self.yerr = flux_err
        self.order_domain = (-1, 1)
        self.wave_domain = (-1, 1)
        self.echelle_orders = data_orders
        self.x_degree = x_degree
        self.y_degree = y_degree
        self.order_numbers = order_numbers

        # Concatenate the echelle orders
        N = min([o.size() for o in data_orders])
        wl = [o.x[:N] * 10 for o in data_orders]
        fl = [o.y[:N] for o in data_orders]
        sig = [o.err[:N] for o in data_orders]

        # Initialize the model spectrum
        self._T = None
        self._logg = None
        self._feh = None
        hdf5_int = StellarModel.HDF5Interface(model_library)
        dataspec = StellarModel.DataSpectrum(wls=wl, fls=fl, sigmas=sig)
        self.interpolator = StellarModel.Interpolator(hdf5_int, dataspec)
        self.update_model(Teff=T, logg=logg, feh=feh, **kwargs)

        # Guess the rv and vsini
        if initialize:
            self.rv_guess, self.vsini_guess = self.initialize_rv()

        return


    def update_model(self, Teff=9000, logg=4.5, feh=0.0, norm=True, **kwargs):
        """
        Update the current model to have the given parameters.

        Parameters:
        ===========
         - T:         float
                      The temperature of the model to use. This can be updated with the 'update_model' method.

         - logg:      float
                      The surface gravity of the model to use.
                      This can be updated with the 'update_model' method.

         - feh:       float
                      The metallicity of the model to use. This can be updated with the 'update_model' method.

         - norm:      boolean
                      Should we fit a continuum to the model spectrum, or is it already normalized?

        Returns:
        ========
        None
        """
        # make sure this is not the model we already have
        if Teff == self._T and logg == self._logg and feh == self._feh:
            return

        # Interpolate the model
        model_flux = self.interpolator(dict(temp=Teff, logg=logg, Z=feh))
        model = DataStructures.xypoint(x=self.interpolator.wl / 10., y=model_flux)

        # Only keep the parts of the model we need
        idx = (model.x > self.x['wave'].min() - 10) & (model.x < self.x['wave'].max() + 10)
        self.model_spec = model[idx].copy()
        if norm:
            self.model_spec.cont = RobustFit(self.model_spec.x, self.model_spec.y, fitorder=3)
        else:
            self.model_spec.cont = np.ones(self.model_spec.size())

        # Update instance variables
        self._T = Teff
        self._logg = logg
        self._feh = feh
        return


    def _model(self, params, x):
        wave = -1 + 2 * (x['wave'] - self.low_wave) / (self.high_wave - self.low_wave)
        ap = -1 + 2 * (x['order'] - self.low_ap) / (self.high_ap - self.low_ap)
        chebvander = np.polynomial.chebyshev.chebvander2d(wave, ap, [self.x_degree, self.y_degree])
        cont = np.dot(chebvander, params)
        return (cont + 1) / 2.0 * (self.high_flux - self.low_flux) + self.low_flux


    def _get_continuum_pars(self, x, y, x_degree, y_degree, norm=norms.HuberT()):
        wave = -1 + 2 * (x['wave'] - self.low_wave) / (self.high_wave - self.low_wave)
        ap = -1 + 2 * (x['order'] - self.low_ap) / (self.high_ap - self.low_ap)
        flux = -1 + 2 * (y - self.low_flux) / (self.high_flux - self.low_flux)
        chebvander = np.polynomial.chebyshev.chebvander2d(wave, ap, [x_degree, y_degree])
        rlm = sm.RLM(flux, chebvander, M=norm).fit()
        return rlm.params


    def flatten_orders(self, pars, plot=False, return_lnlike=False, *args, **kwargs):
        """
        Flatten the input orders using the current model spectrum. This function performs a robust
        linear fit to the continuum as a function of wavelength and aperture number

        Parameters:
        ===========
         - pars:            An iterable of size 2
                            Should contain the radial velocity and vsini (both in km/s) to apply to the model spectrum

         - plot:            boolean
                            Whether or not to plot the flattened orders after calculating them.
                            Only useful for debugging, really.

         - return_lnlike:   boolean
                            If true, return the likelihood function of this model spectrum and set of parameters.

        Returns:
        =========
        A list of flattened orders corresponding to the input orders given in the __init__ method. If return_lnlike
        is true, this also returns a float with the likelihood of this set of parameters and model spectrum.
        """

        # Get the model spectrum
        rv, vsini = pars
        model = Broaden.RotBroad(self.model_spec.copy(), vsini * 1e5, linear=True)
        modelfcn = spline(model.x, model.y / model.cont)

        # Get the robust fit parameters
        norm = kwargs['norm'] if 'norm' in kwargs else norms.HuberT()
        rlm_params = self._get_continuum_pars(x=dict(wave=self.x['wave'], order=self.x['order']),
                                              y=self.y / modelfcn(self.x['wave'] * (1 + rv / clight)),
                                              x_degree=self.x_degree, y_degree=self.y_degree,
                                              norm=norm)

        orders = [o.copy() for o in self.echelle_orders]
        ll = 0.0
        for i, order in enumerate(orders):
            cont = self._model(rlm_params, x=dict(wave=order.x, order=np.ones_like(order.x) * self.order_numbers[i]))
            if plot:
                plt.plot(order.x, order.y / cont, 'k-', alpha=0.4)
            orders[i].y /= cont
            orders[i].err /= cont
            orders[i].cont = np.ones_like(cont)
            if return_lnlike:
                model_order = modelfcn(order.x * (1 + rv / clight))
                ll += -0.5 * np.sum((orders[i].y - model_order) ** 2 / orders[i].err ** 2)
        return (orders, ll) if return_lnlike else orders


    def shift_orders(self, orders):
        """ Shift each order so that the median flux in the overlap is the same.
        """
        shifted_orders = [orders[0].copy()]
        for left_order, right_order in zip(orders[:-1], orders[1:]):
            if left_order.x[-1] > right_order.x[0]:
                left_idx = left_order.x > right_order.x[0]
                right_idx = right_order.x < left_order.x[-1]
                left_med = np.median(left_order.y[left_idx])
                right_med = np.median(right_order.y[right_idx])
                right_order.y += left_med - right_med
            shifted_orders.append(right_order.copy())
        return shifted_orders


    def initialize_rv(self, M=norms.HuberT(), vsini_trials=10):
        """
        Uses a cross-correlation function to estimate the radial velocity and vsini of the primary star.
        It calculates several CCFS with various vsini trials. The correct vsini will have the highest peak, and
        the location of the peak gives the radial velocity.

        Parameters:
        ===========
         - M:             statsmodels.robust.norms class
                          Which M-estimator do you want to use? See statsmodels documentation for details.

         - vsini_trials:  integer
                          How many vsini trials should we take. More will take longer, but give a more accurate result.
                          The trials are calculated from 10 to 400 km/s in equal steps.

        Returns:
        ========
        rv, vsini (both floats)
        """

        # First, get an initial continuum guess using robust fitting.
        logging.info('Initializing continuum for RV guess.')
        idx = (self.x['wave'] < 480) | (self.x['wave'] > 490)
        rlm_params = self._get_continuum_pars(x=dict(wave=self.x['wave'][idx], order=self.x['order'][idx]),
                                              y=self.y[idx], x_degree=self.x_degree, y_degree=self.y_degree,
                                              norm=M)

        # Normalize the orders
        orders = [o.copy() for o in self.echelle_orders]
        for i, order in enumerate(orders):
            orders[i].cont = self._model(rlm_params, x=dict(wave=order.x,
                                                            order=np.ones_like(order.x) * self.order_numbers[i]))

        # Now, use cross-correlation to guess the RV and vsini of the star.
        logging.info('Estimating the RV and vsini by cross-correlation')
        vsini_vals = np.linspace(10, 400, vsini_trials)
        max_ccf = np.empty(vsini_vals.size)
        max_vel = np.empty(vsini_vals.size)
        for i, vsini in enumerate(vsini_vals):
            logging.debug('Trying vsini = {} km/s'.format(vsini))

            retdict = Correlate.GetCCF(orders, self.model_spec.copy(), resolution=None,
                                       process_model=True, rebin_data=True,
                                       vsini=vsini, addmode='simple')
            ccf = retdict['CCF']
            idx = np.argmax(ccf.y)
            max_ccf[i] = ccf.y[idx]
            max_vel[i] = ccf.x[idx]

        try:
            coeffs = np.polyfit(vsini_vals, max_ccf, 2)
            vsini_guess = -coeffs[1] / (2 * coeffs[0])
            idx = np.argmin(np.abs(vsini_vals - vsini_guess))
            rv_guess = max_vel[idx]
        except:
            rv_guess = -max_vel[np.argmax(max_ccf)]
            vsini_guess = vsini_vals[np.argmax(max_ccf)]

        return rv_guess, vsini_guess


    def _fit_logg_teff(self, logg=4.0, teff=10000, **kwargs):
        p_init = [teff, logg, self.rv_guess]
        normalize = kwargs['normalize'] if 'normalize' in kwargs else True
        out = minimize(self._teff_logg_rv_lnlike, p_init, args=(self.vsini_guess, normalize),
                       bounds=((7000, 30000), (3.0, 4.5), (-200, 200)),
                       method='L-BFGS-B', options=dict(ftol=1e-5, maxfun=200, eps=0.1))
        return out.x


    def _teff_logg_rv_lnlike(self, pars, vsini=100., normalize=True, **kwargs):
        logging.info('T = {}\nlogg = {}\nRV = {}'.format(pars[0], pars[1], pars[2]))
        self.update_model(Teff=pars[0], logg=pars[1], feh=self._feh, norm=normalize)
        p = (pars[2], vsini)
        _, ll = self.flatten_orders(plot=False, pars=p, return_lnlike=True, norm=norms.TukeyBiweight())
        logging.info('LogL = {}\n'.format(ll))
        return -ll


    def fit(self, x_degree=None, y_degree=None, **kwargs):
        """
        Find the best set of model parameters T, logg, and rv

        Parameters:
        ============
         - x_degree, y_degree: floats -- defaults = (4,6)
                               The degree of the chebyshev polynomial to use when fitting the continuum.
                               x is the wavelength, and y is the aperture number.
         - logg, teff:         floats -- defaults = (4.0, 10000)
                               Initial guesses for the parameters.

        Returns:
        ========
        best-fit values for teff, logg, and rv (in that order).
        """
        if x_degree is not None:
            self.x_degree = x_degree
        if y_degree is not None:
            self.y_degree = y_degree
        return self._fit_logg_teff(**kwargs)


def flatten_spec(filename, hdf5_lib, teff=9000, logg=4.0, feh=0.0, first_order=0, last_order=19, ordernums=None,
                 x_degree=4, y_degree=9, normalize_model=True, summary_file='Flatten.log'):
    """
    Flatten a spectrum and save a new file

    Parameters:
    ===========
    - filename:        string
                       The path to the file you want to flatten

    - hdf5_lib:        string
                       The path to an HDF5 file with model spectra

    - teff:            float, default = 9000
                       The effective temperature of the star (initial guess)

    - logg:            float, default = 4.0
                       The surface gravity of the star (initial guess)

    - feh:             float, default = 0.0
                       The metallicity of the star (this will NOT be fit)

    - first_order:     int, default = 0
                       The first order in the file to include in the flattening. Usually just include the blue orders.

    - last_order:      int, default = 19
                       The last order in the file to include in the flattening. Usually just include the blue orders.

    - ordernums:       iterable
                       If given, this over-rides first_order and last_order. It should be a list of order numbers
                       to use.

    - x_degree:        int, default = 4
                       The polynomial degree to fit in the dispersion direction for the continuum

    - y_degree:        int, default = 9
                       The polynomial degree to fit in the aperture number direction for the continuum
    
    - normalize_model: boolean
                       Should the model be normalized or is it already?

    - summary_file:    string:
                       A file to save the best-fit temperature, logg, and rv


    Returns:
    ===========
    A list of flattened spectra of size last_order - first_order + 1
    """

    # Read in the file and some header information
    all_orders = HelperFunctions.ReadExtensionFits(filename)
    header = fits.getheader(filename)
    starname = header['OBJECT']
    date = header['DATE-OBS']

    # Figure out which orders to use
    if ordernums is None:
        ordernums = range(first_order, last_order + 1)
    orders = [o.copy() for i, o in enumerate(all_orders) if i in ordernums]

    # Check if this file is already done
    fit = True
    df = pd.read_csv(summary_file, header=None, names=['fname', 'star', 'date_obs', 'teff', 'logg', 'rv'])
    subset = df.loc[(df.fname == filename) & (df.star == starname) & (df.date_obs == date)]
    if len(subset) == 1:
        # Found!
        inp = raw_input('This file has previously been fit. Do you want to re-fit? y/[N]: ')
        if 'n' in inp.lower() or inp.strip() == '':
            teff = subset.teff.values[0]
            logg = subset.logg.values[0]
            rv = subset.rv.values[0]
            fit = False
        else:
            good = df.loc[df.index != subset.index]
            good.to_csv(summary_file, header=None, index=False)
    else:
        logging.info('Did not find file ({}) in log ({})'.format(filename, summary_file))

    mcf = ModelContinuumFitter(orders, hdf5_lib, x_degree=x_degree, y_degree=y_degree,
                               T=teff, logg=logg, feh=feh, initialize=True, order_numbers=ordernums)
    logging.info('RV guess = {}\n\tvsini guess = {}'.format(mcf.rv_guess, mcf.vsini_guess))

    # Fit the model and rv
    if fit:
        logging.debug('Fitting the teff, logg, and rv. This will take a very long time...')
        teff, logg, rv = mcf.fit(teff=teff, logg=logg, normalize=normalize_model)

        # Save the best values to a file
        with open(summary_file, 'a') as outfile:
            outfile.write('{},{},{},{},{},{}\n'.format(filename, starname, date, teff, logg, rv))

    # Flatten the spectrum
    logging.info('Flattening the spectrum using the best-fit values')
    mcf.update_model(Teff=teff, logg=logg, feh=feh)
    p = (rv, mcf.vsini_guess)
    flattened = mcf.flatten_orders(pars=p, plot=False, norm=norms.HuberT())
    shifted_orders = mcf.shift_orders([o.copy() for o in flattened])

    # Fit a continuum to the whole thing now that the orders (mostly) overlap
    def continuum(x, y, lowreject=3, highreject=5, fitorder=3):
        done = False
        idx = (x < 480) | (x > 490)
        x2 = np.copy(x[idx])
        y2 = np.copy(y[idx])
        while not done:
            done = True
            pars = np.polyfit(x2, y2, fitorder)
            fit = np.poly1d(pars)

            residuals = y2 - fit(x2)
            mean = np.mean(residuals)
            std = np.std(residuals)
            badpoints = \
            np.where(np.logical_or((residuals - mean) < -lowreject * std, residuals - mean > highreject * std))[0]
            if badpoints.size > 0 and x2.size - badpoints.size > 5 * fitorder:
                done = False
                x2 = np.delete(x2, badpoints)
                y2 = np.delete(y2, badpoints)
        return np.poly1d(pars)

    # Re-normalize to remove the large scale stuff
    x = np.hstack([o.x for o in shifted_orders])
    y = np.hstack([o.y for o in shifted_orders])
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    contfcn = continuum(x, y, lowreject=1.5, highreject=20, fitorder=6)
    renormalized = []
    for order in shifted_orders:
        cont = contfcn(order.x)
        o = order.copy()
        o.y /= cont
        o.err /= cont
        o.cont = np.ones_like(o.x)
        renormalized.append(o.copy())

    # Output
    to_fits(renormalized, filename, '{}_renormalized.fits'.format(filename.split('.fits')[0]))
    to_fits(flattened, filename, '{}_flattened.fits'.format(filename.split('.fits')[0]))
    to_fits(shifted_orders, filename, '{}_shifted.fits'.format(filename.split('.fits')[0]))

    return renormalized, flattened, shifted_orders, mcf


def to_fits(orders, template_fname, outfilename):
    column_dicts = []
    for order in orders:
        column_dicts.append(dict(wavelength=order.x, error=order.err, continuum=np.ones(order.size()), flux=order.y))

    logging.info('Outputting spectrum to file {}'.format(outfilename))
    HelperFunctions.OutputFitsFileExtensions(column_dicts, template_fname, outfilename, mode='new')
