__author__ = 'Kevin Gullikson'
import os
import sys
import re
from collections import defaultdict
import warnings
from scipy.interpolate import InterpolatedUnivariateSpline as spline, LinearNDInterpolator, NearestNDInterpolator
import pandas

import numpy as np
from astropy import units
import DataStructures
import FittingUtilities
import h5py

import HelperFunctions
import Broaden


"""
This code provides the GetModelList function.
It is used in GenericSearch.py and SensitivityAnalysis.py
"""

if "darwin" in sys.platform:
    modeldir = "/Volumes/DATADRIVE/Stellar_Models/Sorted/Stellar/Vband/"
    HDF5_FILE = '/Volumes/DATADRIVE/Stellar_Models/Search_Grid.hdf5'
elif "linux" in sys.platform:
    modeldir = "/media/FreeAgent_Drive/SyntheticSpectra/Sorted/Stellar/Vband/"
    HDF5_FILE = '/media/ExtraSpace/PhoenixGrid/Search_Grid.hdf5'
else:
    modeldir = raw_input("sys.platform not recognized. Please enter model directory below: ")
    if not modeldir.endswith("/"):
        modeldir = modeldir + "/"


def GetModelList(type='phoenix',
                 metal=[-0.5, 0, 0.5],
                 logg=[4.5, ],
                 temperature=range(3000, 6900, 100),
                 alpha=[0, 0.2],
                 model_directory=modeldir,
                 hdf5_file=HDF5_FILE):
    """This function searches the model directory (hard coded in StellarModels.py) for stellar
       models with the appropriate parameters

    :param type: the type of models to get. Right now, only 'phoenix', 'kurucz', and 'hdf5' are implemented
    :param metal: a list of the metallicities to include
    :param logg: a list of the surface gravity values to include
    :param temperature: a list of the temperatures to include
    :param model_directory: The absolute path to the model directory (only used for type=phoenix or kurucz)
    :param hdf5_file: The absolute path to the HDF5 file with the models (only used for type=hdf5)
    :return: a list of filenames for the requested models
    """

    # Error checking
    if ( not ( HelperFunctions.IsListlike(metal) and
                   HelperFunctions.IsListlike(logg) and
                   HelperFunctions.IsListlike(temperature))):
        raise ValueError("The metal, log, and temperature arguments must ALL be list-like!")

    if type.lower() == 'phoenix':
        all_models = sorted([f for f in os.listdir(model_directory) if 'phoenix' in f.lower()])
        chosen_models = []
        for model in all_models:
            Teff, gravity, metallicity = ClassifyModel(model)
            if Teff in temperature and gravity in logg and metallicity in metal:
                chosen_models.append("{:s}{:s}".format(model_directory, model))


    elif type.lower() == "kurucz":
        all_models = [f for f in os.listdir(modeldir) if f.startswith("t") and f.endswith(".dat.bin.asc")]
        chosen_models = []
        for model in all_models:
            Teff, gravity, metallicity, a = ClassifyModel(model, type='kurucz')
            if Teff in temperature and gravity in logg and metallicity in metal and a in alpha:
                chosen_models.append("{:s}{:s}".format(model_directory, model))


    elif type.lower() == 'hdf5':
        hdf5_int = HDF5Interface(hdf5_file)
        chosen_models = []
        for par in hdf5_int.list_grid_points:
            if par['temp'] in temperature and par['logg'] in logg and par['Z'] in metal and par['alpha'] in alpha:
                chosen_models.append(par)

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    return chosen_models


def ClassifyModel(filename, type='phoenix'):
    """Get the effective temperature, log(g), and [Fe/H] of a stellar model from the filename

    :param filename:
    :param type: Currently, only phoenix type files are supported
    :return:
    """
    if not isinstance(filename, basestring):
        raise TypeError("Filename must be a string!")

    if type.lower() == 'phoenix':
        segments = re.split("-|\+", filename.split("/")[-1])
        temp = float(segments[0].split("lte")[-1]) * 100
        gravity = float(segments[1])
        metallicity = float(segments[2][:3])
        if not "+" in filename and metallicity > 0:
            metallicity *= -1
        return temp, gravity, metallicity

    elif type.lower() == 'kurucz':
        fname = filename.split('/')[-1]
        temp = float(fname[1:6])
        gravity = float(fname[8:12])
        metallicity = float(fname[14:16]) / 10.0
        alpha = float(fname[18:20]) / 10.0
        if fname[13] == "m":
            metallicity *= -1
        if fname[17] == "m":
            alpha *= -1
        return temp, gravity, metallicity, alpha

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    return temp, gravity, metallicity


def MakeModelDicts(model_list, vsini_values=[10, 20, 30, 40], type='phoenix',
                   vac2air=True, logspace=False, hdf5_file=HDF5_FILE, get_T_sens=False):
    """This will take a list of models, and output two dictionaries that are
    used by GenericSearch.py and Sensitivity.py

    :param model_list: A list of model filenames
    :param vsini_values: a list of vsini values to broaden the spectrum by (we do that later!)
    :param type: the type of models. Currently, phoenix, kurucz, and hdf5 are implemented
    :param vac2air: If true, assumes the model is in vacuum wavelengths and converts to air
    :param logspace: If true, it will rebin the data to a constant log-spacing
    :param hdf5_file: The absolute path to the HDF5 file with the models. Only used if type=hdf5
    :param get_T_sens: Boolean flag for getting the temperature sensitivity.
                       If true, it finds the derivative of each pixel dF/dT
    :return: A dictionary containing the model with keys of temperature, gravity, metallicity, and vsini,
             and another one with a processed flag with the same keys
    """

    if type.lower() == 'phoenix':
        modeldict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint))))
        processed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool))))
        for fname in model_list:
            temp, gravity, metallicity = ClassifyModel(fname)
            print "Reading in file %s" % fname
            data = pandas.read_csv(fname,
                                   header=None,
                                   names=["wave", "flux"],
                                   usecols=(0, 1),
                                   sep=' ',
                                   skipinitialspace=True)
            x, y = data['wave'].values, data['flux'].values
            # x, y = np.loadtxt(fname, usecols=(0, 1), unpack=True)
            if vac2air:
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
            model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm), y=10 ** y)
            if logspace:
                xgrid = np.logspace(np.log(model.x[0]), np.log(model.x[-1]), model.size(), base=np.e)
                model = FittingUtilities.RebinData(model, xgrid)
            for vsini in vsini_values:
                modeldict[temp][gravity][metallicity][vsini] = model
                processed[temp][gravity][metallicity][vsini] = False

    elif type.lower() == 'kurucz':
        modeldict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint)))))
        processed = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))))
        for fname in model_list:
            temp, gravity, metallicity, a = ClassifyModel(fname)
            print "Reading in file %s" % fname
            data = pandas.read_csv(fname,
                                   header=None,
                                   names=["wave", "flux"],
                                   usecols=(0, 1),
                                   sep=' ',
                                   skipinitialspace=True)
            x, y = data['wave'].values, data['flux'].values
            # x, y = np.loadtxt(fname, usecols=(0, 1), unpack=True)
            if vac2air:
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
            model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm), y=10 ** y)
            if logspace:
                xgrid = np.logspace(np.log(model.x[0]), np.log(model.x[-1]), model.size(), base=np.e)
                model = FittingUtilities.RebinData(model, xgrid)
            for vsini in vsini_values:
                modeldict[temp][gravity][metallicity][a][vsini] = model
                processed[temp][gravity][metallicity][a][vsini] = False

    elif type.lower() == 'hdf5':
        hdf5_int = HDF5Interface(hdf5_file)
        x = hdf5_int.wl
        wave_hdr = hdf5_int.wl_header
        if vac2air:
            if not wave_hdr['air']:
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
        elif wave_hdr['air']:
            raise GridError(
                'HDF5 grid is in air wavelengths, but you requested vacuum wavelengths. You need a new grid!')
        modeldict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint)))))
        processed = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))))
        for pars in model_list:
            temp, gravity, metallicity, a = pars['temp'], pars['logg'], pars['Z'], pars['alpha']
            y = hdf5_int.load_flux(pars)
            model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm), y=y)
            for vsini in vsini_values:
                modeldict[temp][gravity][metallicity][a][vsini] = model
                processed[temp][gravity][metallicity][a][vsini] = False

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    if get_T_sens:
        # Get the temperature sensitivity. Warning! This assumes the wavelength grid is the same in all models.
        sensitivity = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint)))))
        Tvals = sorted(modeldict.keys())
        for i, T in enumerate(Tvals):
            gvals = sorted(modeldict[T].keys())
            for gravity in gvals:
                metal_vals = sorted(modeldict[T][gravity].keys())
                for metal in metal_vals:
                    alpha_vals = sorted(modeldict[T][gravity][metal].keys())
                    for alpha in alpha_vals:
                        # get the temperature just under this one
                        lower, l_idx = get_model(modeldict, Tvals, i, gravity, metal, vsini_values[0], alpha, mode='lower')
                        upper, u_idx = get_model(modeldict, Tvals, i, gravity, metal, vsini_values[0], alpha, mode='upper')
                        T_low = Tvals[l_idx]
                        T_high = Tvals[u_idx]
                        slope = (upper.y - lower.y) / (T_high - T_low)
                        for vsini in vsini_values:
                            sensitivity[T][gravity][metal][alpha][vsini] = slope
        return modeldict, process, sensitivity

    return modeldict, processed


def get_model(mdict, Tvals, i, logg, metal, vsini, alpha=None, mode='same'):
    """
    Get the model with the requested parameters
    :param mode: How to get the model. valid options:
        - 'same': Get the model with the exact requested parameters.
        - 'lower': Get model with the exact values of everything except temperature (find the next lowest temperature)
        - 'upper': Get model with the exact values of everything except temperature (find the next highest temperature)
    """
    if mode == 'same':
        if alpha is None:
            mdict[Tvals[i]][logg][metal][vsini]
        else:
            return mdict[Tvals[i]][logg][metal][alpha][vsini]
    elif mode == 'lower':
        done = False
        idx = i - 1
        while not done:
            if idx == 0 or idx == len(Tvals) - 1:
                return get_model(mdict, Tvals, i, logg, metal, vsini, alpha, mode='same'), i
            try:
                return get_model(mdict, Tvals, idx, logg, metal, vsini, alpha, mode='same'), idx
            except KeyError:
                idx -= 1
    elif mode == 'upper':
        done = False
        idx = i +1
        while not done:
            if idx == 0 or idx == len(Tvals) - 1:
                return get_model(mdict, Tvals, i, logg, metal, vsini, alpha, mode='same'), i
            try:
                return get_model(mdict, Tvals, idx, logg, metal, vsini, alpha, mode='same'), idx
            except KeyError:
                idx += 1



class HDF5Interface:
    '''
    Connect to an HDF5 file that stores spectra. Stolen shamelessly from Ian Czekala's Starfish code
    '''

    def __init__(self, filename, ranges={"temp": (0, np.inf),
                                         "logg": (-np.inf, np.inf),
                                         "Z": (-np.inf, np.inf),
                                         "alpha": (-np.inf, np.inf)}):
        '''
            :param filename: the name of the HDF5 file
            :type param: string
            :param ranges: optionally select a smaller part of the grid to use.
            :type ranges: dict
        '''
        self.filename = filename
        self.flux_name = "t{temp:.0f}g{logg:.1f}z{Z:.1f}a{alpha:.1f}"
        grid_parameters = ("temp", "logg", "Z", "alpha")  # Allowed grid parameters
        grid_set = frozenset(grid_parameters)

        with h5py.File(self.filename, "r") as hdf5:
            self.wl = hdf5["wl"][:]
            self.wl_header = dict(hdf5["wl"].attrs.items())

            grid_points = []

            for key in hdf5["flux"].keys():
                # assemble all temp, logg, Z, alpha keywords into a giant list
                hdr = hdf5['flux'][key].attrs

                params = {k: hdr[k] for k in grid_set}

                #Check whether the parameters are within the range
                for kk, vv in params.items():
                    low, high = ranges[kk]
                    if (vv < low) or (vv > high):
                        break
                else:
                    #If all parameters have passed successfully through the ranges, allow.
                    grid_points.append(params)

            self.list_grid_points = grid_points

        # determine the bounding regions of the grid by sorting the grid_points
        temp, logg, Z, alpha = [], [], [], []
        for param in self.list_grid_points:
            temp.append(param['temp'])
            logg.append(param['logg'])
            Z.append(param['Z'])
            alpha.append(param['alpha'])

        self.bounds = {"temp": (min(temp), max(temp)),
                       "logg": (min(logg), max(logg)),
                       "Z": (min(Z), max(Z)),
                       "alpha": (min(alpha), max(alpha))}

        self.points = {"temp": np.unique(temp),
                       "logg": np.unique(logg),
                       "Z": np.unique(Z),
                       "alpha": np.unique(alpha)}

        self.ind = None  #Overwritten by other methods using this as part of a ModelInterpolator

    def load_flux(self, parameters):
        '''
        Load just the flux from the grid, with possibly an index truncation.

        :param parameters: the stellar parameters
        :type parameters: dict

        :raises KeyError: if spectrum is not found in the HDF5 file.

        :returns: flux array
        '''

        key = self.flux_name.format(**parameters)
        with h5py.File(self.filename, "r") as hdf5:
            try:
                if self.ind is not None:
                    fl = hdf5['flux'][key][self.ind[0]:self.ind[1]]
                else:
                    fl = hdf5['flux'][key][:]
            except KeyError as e:
                raise GridError(e)

        # Note: will raise a KeyError if the file is not found.

        return fl


class GridError(Exception):
    '''
    Raised when a spectrum cannot be found in the grid.
    '''

    def __init__(self, msg):
        self.msg = msg


class KuruczGetter():
    def __init__(self, modeldir, rebin=True, T_min=7000, T_max=9000, logg_min=3.5, logg_max=4.5, metal_min=-0.5,
                 metal_max=0.5, alpha_min=0.0, alpha_max=0.4, wavemin=0, wavemax=np.inf, debug=False):
        """
        This class will read in a directory with Kurucz models

        The associated methods can be used to interpolate a model at any
        temperature, gravity, metallicity, and [alpha/Fe] value that
        falls within the grid

        modeldir: The directory where the models are stored. Can be a list of model directories too!
        rebin: If True, it will rebin the models to a constant x-spacing
        other args: The minimum and maximum values for the parameters to search.
                    You need to keep this as small as possible to avoid memory issues!
                    The whole grid would take about 36 GB of RAM!
        """
        self.rebin = rebin
        self.debug = debug

        # First, read in the grid
        if HelperFunctions.IsListlike(modeldir):
            # There are several directories to combine
            Tvals = []
            loggvals = []
            metalvals = []
            alphavals = []
            for i, md in enumerate(modeldir):
                if i == 0:
                    T, G, Z, A, S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max, logg_min=logg_min,
                                                   logg_max=logg_max, metal_min=metal_min, metal_max=metal_max,
                                                   alpha_min=alpha_min, alpha_max=alpha_max, wavemin=wavemin,
                                                   wavemax=wavemax,
                                                   xaxis=None)
                    spectra = np.array(S)
                else:
                    T, G, Z, A, S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max, logg_min=logg_min,
                                                   logg_max=logg_max, metal_min=metal_min, metal_max=metal_max,
                                                   alpha_min=alpha_min, alpha_max=alpha_max, wavemin=wavemin,
                                                   wavemax=wavemax,
                                                   xaxis=self.xaxis)
                    S = np.array(S)
                    spectra = np.vstack((spectra, S))

                Tvals = np.hstack((Tvals, T))
                loggvals = np.hstack((loggvals, G))
                metalvals = np.hstack((metalvals, Z))
                alphavals = np.hstack((alphavals, A))
        else:
            Tvals, loggvals, metalvals, alphavals, spectra = self.read_grid(modeldir,
                                                                            rebin=rebin,
                                                                            T_min=T_min,
                                                                            T_max=T_max,
                                                                            logg_min=logg_min,
                                                                            logg_max=logg_max,
                                                                            metal_min=metal_min,
                                                                            metal_max=metal_max,
                                                                            alpha_min=alpha_min,
                                                                            alpha_max=alpha_max,
                                                                            wavemin=wavemin,
                                                                            wavemax=wavemax,
                                                                            xaxis=None)

        # Check if there are actually two different values of alpha/Fe
        alpha_varies = True if max(alphavals) - min(alphavals) > 0.1 else False

        # Scale the variables so they all have about the same range
        self.T_scale = ((max(Tvals) + min(Tvals)) / 2.0, max(Tvals) - min(Tvals))
        self.metal_scale = ((max(metalvals) + min(metalvals)) / 2.0, max(metalvals) - min(metalvals))
        self.logg_scale = ((max(loggvals) + min(loggvals)) / 2.0, max(loggvals) - min(loggvals))
        if alpha_varies:
            self.alpha_scale = ((max(alphavals) + min(alphavals)) / 2.0, max(alphavals) - min(alphavals))
        Tvals = (np.array(Tvals) - self.T_scale[0]) / self.T_scale[1]
        loggvals = (np.array(loggvals) - self.logg_scale[0]) / self.logg_scale[1]
        metalvals = (np.array(metalvals) - self.metal_scale[0]) / self.metal_scale[1]
        if alpha_varies:
            alphavals = (np.array(alphavals) - self.alpha_scale[0]) / self.alpha_scale[1]
        print self.T_scale
        print self.metal_scale
        print self.logg_scale
        if alpha_varies:
            print self.alpha_scale

        # Make the grid and interpolator instances
        if alpha_varies:
            self.grid = np.array((Tvals, loggvals, metalvals, alphavals)).T
        else:
            self.grid = np.array((Tvals, loggvals, metalvals)).T
        self.spectra = np.array(spectra)
        self.interpolator = LinearNDInterpolator(self.grid, self.spectra)  # , rescale=True)
        self.NN_interpolator = NearestNDInterpolator(self.grid, self.spectra)  # , rescale=True)
        self.alpha_varies = alpha_varies


    def read_grid(self, modeldir, rebin=True, T_min=7000, T_max=9000, logg_min=3.5, logg_max=4.5, metal_min=-0.5,
                  metal_max=0.5, alpha_min=0.0, alpha_max=0.4, wavemin=0, wavemax=np.inf, xaxis=None):
        Tvals = []
        loggvals = []
        metalvals = []
        alphavals = []
        spectra = []
        firstkeeper = True
        modelfiles = [f for f in os.listdir(modeldir) if f.startswith("t") and f.endswith(".dat.bin.asc")]
        for i, fname in enumerate(modelfiles):
            T = float(fname[1:6])
            logg = float(fname[8:12])
            metal = float(fname[14:16]) / 10.0
            alpha = float(fname[18:20]) / 10.0
            if fname[13] == "m":
                metal *= -1
            if fname[17] == "m":
                alpha *= -1

            # Read in and save file if it falls in the correct parameter range
            if (T_min <= T <= T_max and
                            logg_min <= logg <= logg_max and
                            metal_min <= metal <= metal_max and
                            alpha_min <= alpha <= alpha_max):

                if self.debug:
                    print "Reading in file {:s}".format(fname)
                data = pandas.read_csv("{:s}/{:s}".format(modeldir, fname),
                                       header=None,
                                       names=["wave", "norm"],
                                       usecols=(0, 3),
                                       sep=' ',
                                       skipinitialspace=True)
                x, y = data['wave'].values, data['norm'].values
                # x, y = np.loadtxt("{:s}/{:s}".format(modeldir, fname), usecols=(0, 3), unpack=True)
                x *= units.angstrom.to(units.nm)
                y[np.isnan(y)] = 0.0

                left = np.searchsorted(x, wavemin)
                right = np.searchsorted(x, wavemax)
                x = x[left:right]
                y = y[left:right]

                if rebin:
                    if firstkeeper:
                        xgrid = np.logspace(np.log10(x[0]), np.log10(x[-1]), x.size)
                    else:
                        xgrid = self.xaxis
                    fcn = spline(x, y)
                    x = xgrid
                    y = fcn(xgrid)

                if firstkeeper:
                    self.xaxis = x if xaxis is None else xaxis
                    firstkeeper = False
                elif np.max(np.abs(self.xaxis - x) > 1e-4):
                    warnings.warn("x-axis for file {:s} is different from the master one! Not saving!".format(fname))
                    continue

                Tvals.append(T)
                loggvals.append(logg)
                metalvals.append(metal)
                alphavals.append(alpha)
                spectra.append(y)

        return Tvals, loggvals, metalvals, alphavals, spectra


    def __call__(self, T, logg, metal, alpha, vsini=0.0, return_xypoint=True, **kwargs):
        """
        Given parameters, return an interpolated spectrum

        If return_xypoint is False, then it will only return
          a numpy.ndarray with the spectrum

        Before interpolating, we will do some error checking to make
        sure the requested values fall within the grid
        """

        # Scale the requested values
        if self.debug:
            print T, logg, metal, alpha, vsini
        T = (T - self.T_scale[0]) / self.T_scale[1]
        logg = (logg - self.logg_scale[0]) / self.logg_scale[1]
        metal = (metal - self.metal_scale[0]) / self.metal_scale[1]
        if self.alpha_varies:
            alpha = (alpha - self.alpha_scale[0]) / self.alpha_scale[1]


        # Get the minimum and maximum values in the grid
        T_min = min(self.grid[:, 0])
        T_max = max(self.grid[:, 0])
        logg_min = min(self.grid[:, 1])
        logg_max = max(self.grid[:, 1])
        metal_min = min(self.grid[:, 2])
        metal_max = max(self.grid[:, 2])
        alpha_min = min(self.grid[:, 3]) if self.alpha_varies else 0.0
        alpha_max = max(self.grid[:, 3]) if self.alpha_varies else 0.0
        if self.alpha_varies:
            input_list = (T, logg, metal, alpha)
        else:
            input_list = (T, logg, metal)

        # Check to make sure the requested values fall within the grid
        if (T_min <= T <= T_max and
                        logg_min <= logg <= logg_max and
                        metal_min <= metal <= metal_max and
                (not self.alpha_varies or alpha_min <= alpha <= alpha_max)):

            y = self.interpolator(input_list)
        else:
            if self.debug:
                warnings.warn("The requested parameters fall outside the model grid. Results may be unreliable!")
            # print T, T_min, T_max
            # print logg, logg_min, logg_max
            #print metal, metal_min, metal_max
            #print alpha, alpha_min, alpha_max
            y = self.NN_interpolator(input_list)

        # Test to make sure the result is valid. If the requested point is
        # outside the Delaunay triangulation, it will return NaN's
        if np.any(np.isnan(y)):
            if self.debug:
                warnings.warn("Found NaNs in the interpolated spectrum! Falling back to Nearest Neighbor")
            y = self.NN_interpolator(input_list)

        model = DataStructures.xypoint(x=self.xaxis, y=y)
        vsini *= units.km.to(units.cm)
        model = Broaden.RotBroad(model, vsini, linear=self.rebin)


        # Return the appropriate object
        if return_xypoint:
            return model
        else:
            return model.y


"""
=======================================================================
=======================================================================
=======================================================================
"""


class PhoenixGetter():
    def __init__(self, modeldir, rebin=True, T_min=3000, T_max=6800, metal_min=-0.5,
                 metal_max=0.5, wavemin=0, wavemax=np.inf, debug=False):
        """
        This class will read in a directory with Phoenix models

        The associated methods can be used to interpolate a model at any
        temperature, and metallicity value that
        falls within the grid

        modeldir: The directory where the models are stored. Can be a list of model directories too!
        rebin: If True, it will rebin the models to a constant x-spacing
        other args: The minimum and maximum values for the parameters to search.
                    You need to keep this as small as possible to avoid memory issues!
        """
        self.rebin = rebin
        self.debug = debug

        # First, read in the grid
        if HelperFunctions.IsListlike(modeldir):
            # There are several directories to combine
            Tvals = []
            metalvals = []
            for i, md in enumerate(modeldir):
                if i == 0:
                    T, Z, S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max,
                                             metal_min=metal_min, metal_max=metal_max,
                                             wavemin=wavemin, wavemax=wavemax, xaxis=None)
                    spectra = np.array(S)
                else:
                    T, Z, S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max,
                                             metal_min=metal_min, metal_max=metal_max,
                                             wavemin=wavemin, wavemax=wavemax, xaxis=self.xaxis)
                    S = np.array(S)
                    spectra = np.vstack((spectra, S))

                Tvals = np.hstack((Tvals, T))
                metalvals = np.hstack((metalvals, Z))
        else:
            Tvals, metalvals, spectra = self.read_grid(modeldir, rebin=rebin,
                                                       T_min=T_min, T_max=T_max,
                                                       metal_min=metal_min, metal_max=metal_max,
                                                       wavemin=wavemin, wavemax=wavemax, xaxis=None)

        # Scale the variables so they all have about the same range
        self.T_scale = ((max(Tvals) + min(Tvals)) / 2.0, max(Tvals) - min(Tvals))
        self.metal_scale = ((max(metalvals) + min(metalvals)) / 2.0, max(metalvals) - min(metalvals))
        Tvals = (np.array(Tvals) - self.T_scale[0]) / self.T_scale[1]
        metalvals = (np.array(metalvals) - self.metal_scale[0]) / self.metal_scale[1]

        # Make the grid and interpolator instances
        self.grid = np.array((Tvals, metalvals)).T
        self.spectra = np.array(spectra)
        self.interpolator = LinearNDInterpolator(self.grid, self.spectra)  # , rescale=True)
        self.NN_interpolator = NearestNDInterpolator(self.grid, self.spectra)  # , rescale=True)


    def read_grid(self, modeldir, rebin=True, T_min=3000, T_max=6800, metal_min=-0.5,
                  metal_max=0.5, wavemin=0, wavemax=np.inf, xaxis=None, debug=False):
        Tvals = []
        metalvals = []
        spectra = []
        firstkeeper = True
        modelfiles = [f for f in os.listdir(modeldir) if
                      f.startswith("lte") and "PHOENIX" in f and f.endswith(".sorted")]
        for i, fname in enumerate(modelfiles):
            T, logg, metal = ClassifyModel(fname)

            # Read in and save file if it falls in the correct parameter range
            if (T_min <= T <= T_max and
                            metal_min <= metal <= metal_max and
                        logg == 4.5):

                if self.debug:
                    print "Reading in file {:s}".format(fname)
                data = pandas.read_csv("{:s}{:s}".format(modeldir, fname),
                                       header=None,
                                       names=["wave", "flux", "continuum"],
                                       usecols=(0, 1, 2),
                                       sep=' ',
                                       skipinitialspace=True)
                x, y, c = data['wave'].values, data['flux'].values, data['continuum'].values
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
                x *= units.angstrom.to(units.nm)
                y = 10 ** y / 10 ** c

                left = np.searchsorted(x, wavemin)
                right = np.searchsorted(x, wavemax)
                x = x[left:right]
                y = y[left:right]

                if rebin:
                    if firstkeeper:
                        xgrid = np.logspace(np.log10(x[0]), np.log10(x[-1]), x.size)
                    else:
                        xgrid = self.xaxis
                    fcn = spline(x, y)
                    x = xgrid
                    y = fcn(xgrid)

                if firstkeeper:
                    self.xaxis = x if xaxis is None else xaxis
                    firstkeeper = False
                elif np.max(np.abs(self.xaxis - x) > 1e-4):
                    warnings.warn("x-axis for file {:s} is different from the master one! Not saving!".format(fname))
                    continue

                Tvals.append(T)
                metalvals.append(metal)
                spectra.append(y)

        return Tvals, metalvals, spectra


    def __call__(self, T, metal, vsini=0.0, return_xypoint=True, **kwargs):
        """
        Given parameters, return an interpolated spectrum

        If return_xypoint is False, then it will only return
          a numpy.ndarray with the spectrum

        Before interpolating, we will do some error checking to make
        sure the requested values fall within the grid
        """

        # Scale the requested values
        T = (T - self.T_scale[0]) / self.T_scale[1]
        metal = (metal - self.metal_scale[0]) / self.metal_scale[1]

        # Get the minimum and maximum values in the grid
        T_min = min(self.grid[:, 0])
        T_max = max(self.grid[:, 0])
        metal_min = min(self.grid[:, 1])
        metal_max = max(self.grid[:, 1])
        input_list = (T, metal)

        # Check to make sure the requested values fall within the grid
        if (T_min <= T <= T_max and
                        metal_min <= metal <= metal_max):

            y = self.interpolator(input_list)
        else:
            if self.debug:
                warnings.warn("The requested parameters fall outside the model grid. Results may be unreliable!")
            print T, T_min, T_max
            print metal, metal_min, metal_max
            y = self.NN_interpolator(input_list)

        # Test to make sure the result is valid. If the requested point is
        # outside the Delaunay triangulation, it will return NaN's
        if np.any(np.isnan(y)):
            if self.debug:
                warnings.warn("Found NaNs in the interpolated spectrum! Falling back to Nearest Neighbor")
            y = self.NN_interpolator(input_list)

        model = DataStructures.xypoint(x=self.xaxis, y=y)
        vsini *= units.km.to(units.cm)
        model = Broaden.RotBroad(model, vsini, linear=self.rebin)


        # Return the appropriate object
        if return_xypoint:
            return model
        else:
            return model.y