__author__ = 'Kevin Gullikson'
import os
import sys
import re
from collections import defaultdict

import numpy as np
from astropy import units

import HelperFunctions
import DataStructures
from scipy.interpolate import InterpolatedUnivariateSpline as spline, LinearNDInterpolator, NearestNDInterpolator
import warnings
import pandas
import Broaden


"""
This code provides the GetModelList function.
It is used in GenericSearch.py and SensitivityAnalysis.py
"""

if "darwin" in sys.platform:
    modeldir = "/Volumes/DATADRIVE/Stellar_Models/PHOENIX/Stellar/Vband/"
elif "linux" in sys.platform:
    modeldir = "/media/FreeAgent_Drive/SyntheticSpectra/Sorted/Stellar/Vband/"
else:
    modeldir = raw_input("sys.platform not recognized. Please enter model directory below: ")
    if not modeldir.endswith("/"):
        modeldir = modeldir + "/"


def GetModelList(type='phoenix',
                 metal=[-0.5, 0, 0.5],
                 logg=[4.5, ],
                 temperature=range(3000, 6900, 100),
                 alpha=[0, 0.2],
                 model_directory=modeldir):
    """This function searches the model directory (hard coded in StellarModels.py) for stellar
       models with the appropriate parameters

    :param type: the type of models to get. Right now, only 'phoenix' is implemented
    :param metal: a list of the metallicities to include
    :param logg: a list of the surface gravity values to include
    :param temperature: a list of the temperatures to include
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
        temp = int(segments[0].split("lte")[-1]) * 100
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


def MakeModelDicts(model_list, vsini_values=[10, 20, 30, 40], type='phoenix', vac2air=True):
    """This will take a list of models, and output two dictionaries that are
    used by GenericSearch.py and Sensitivity.py

    :param model_list: A list of model filenames
    :param vsini_values: a list of vsini values to broaden the spectrum by (we do that later!)
    :param type: the type of models. Currently, only phoenix is implemented
    :param vac2air: If true, assumes the model is in vacuum wavelengths and converts to air
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
            #x, y = np.loadtxt(fname, usecols=(0, 1), unpack=True)
            if vac2air:
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
            model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm), y=10 ** y)
            for vsini in vsini_values:
                modeldict[temp][gravity][metallicity][a][vsini] = model
                processed[temp][gravity][metallicity][a][vsini] = False

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    return modeldict, processed





class KuruczGetter():
    def __init__(self, modeldir, rebin=True, T_min=7000, T_max=9000, logg_min=3.5, logg_max=4.5, metal_min=-0.5,
                 metal_max=0.5, alpha_min=0.0, alpha_max=0.4, wavemin=0, wavemax=np.inf):
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

        # First, read in the grid
        if HelperFunctions.IsListlike(modeldir):
            #There are several directories to combine
            Tvals = []
            loggvals = []
            metalvals = []
            alphavals = []
            for i, md in enumerate(modeldir):
                if i == 0:
                    T,G,Z,A,S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max, logg_min=logg_min,
                                          logg_max=logg_max, metal_min=metal_min, metal_max=metal_max,
                                          alpha_min=alpha_min, alpha_max=alpha_max, wavemin=wavemin, wavemax=wavemax,
                                          xaxis=None)
                    spectra = np.array(S)
                else:
                    T,G,Z,A,S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max, logg_min=logg_min,
                                          logg_max=logg_max, metal_min=metal_min, metal_max=metal_max,
                                          alpha_min=alpha_min, alpha_max=alpha_max, wavemin=wavemin, wavemax=wavemax,
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


        # Scale the variables so they all have about the same range
        self.T_scale = ((max(Tvals) + min(Tvals)) / 2.0, max(Tvals) - min(Tvals))
        self.metal_scale = ((max(metalvals) + min(metalvals)) / 2.0, max(metalvals) - min(metalvals))
        self.logg_scale = ((max(loggvals) + min(loggvals)) / 2.0, max(loggvals) - min(loggvals))
        self.alpha_scale = ((max(alphavals) + min(alphavals)) / 2.0, max(alphavals) - min(alphavals))
        self.vsini_scale = (150.0, 300.0)
        Tvals = (np.array(Tvals) - self.T_scale[0]) / self.T_scale[1]
        loggvals = (np.array(loggvals) - self.logg_scale[0]) / self.logg_scale[1]
        metalvals = (np.array(metalvals) - self.metal_scale[0]) / self.metal_scale[1]
        alphavals = (np.array(alphavals) - self.alpha_scale[0]) / self.alpha_scale[1]
        print self.T_scale
        print self.metal_scale
        print self.logg_scale
        print self.alpha_scale

        # Make the grid and interpolator instances
        self.grid = np.array((Tvals, loggvals, metalvals, alphavals)).T
        self.spectra = np.array(spectra)
        self.interpolator = LinearNDInterpolator(self.grid, self.spectra)  # , rescale=True)
        self.NN_interpolator = NearestNDInterpolator(self.grid, self.spectra)  # , rescale=True)

        # Set up arrays for things already interpolated
        self.Tvals = list(Tvals)
        self.loggvals = list(loggvals)
        self.metalvals = list(metalvals)
        self.alphavals = list(alphavals)
        self.vsinivals = [-0.5] * len(Tvals)
        self.spec = list(self.spectra)


    def read_grid(self,modeldir, rebin=True, T_min=7000, T_max=9000, logg_min=3.5, logg_max=4.5, metal_min=-0.5,
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
                    xgrid = np.linspace(x[0], x[-1], x.size) if firstkeeper else self.xaxis
                    fcn = spline(x, y)
                    x = xgrid
                    y = fcn(xgrid)

                if firstkeeper:
                    if xaxis is None:
                        self.xaxis = x
                    else:
                        self.xaxis = xaxis
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


    def make_vsini_interpolator(self):
        """
        Makes an interpolator that uses the values already interpolated between
        :return: The interpolator
        """
        grid = np.array((self.Tvals, self.loggvals, self.metalvals, self.alphavals, self.vsinivals)).T
        spectra = np.array(self.spec)
        interpolator = LinearNDInterpolator(grid, spectra)
        return interpolator


    def __call__(self, T, logg, metal, alpha, vsini=0.0, return_xypoint=True, first_interpolator=None):
        """
        Given parameters, return an interpolated spectrum

        If return_xypoint is False, then it will only return
          a numpy.ndarray with the spectrum

        if first_interpolator is given, it will try to interpolate with that first and check for NaNs

        Before interpolating, we will do some error checking to make
        sure the requested values fall within the grid
        """

        # Scale the requested values
        print T, logg, metal, alpha, vsini
        T = (T - self.T_scale[0]) / self.T_scale[1]
        logg = (logg - self.logg_scale[0]) / self.logg_scale[1]
        metal = (metal - self.metal_scale[0]) / self.metal_scale[1]
        alpha = (alpha - self.alpha_scale[0]) / self.alpha_scale[1]
        vsini = (vsini - self.vsini_scale[0]) / self.vsini_scale[1]

        # Try first_interpolator
        fallback = True
        if first_interpolator is not None:
            y = first_interpolator((T, logg, metal, alpha, vsini))
            if np.all(-np.isnan(y)):
                fallback = False

        if fallback:
            #savespectrum = False
            #if (T not in self.Tvals and logg not in self.loggvals and metal not in self.metalvals and
            #            alpha not in self.metalvals and vsini not in self.vsinivals):
            #    self.Tvals.append(T)
            #    self.loggvals.append(logg)
            #    self.metalvals.append(metal)
            #    self.alphavals.append(alpha)
            #    self.vsinivals.append(vsini)
            #    savespectrum = True

            # Get the minimum and maximum values in the grid
            T_min = min(self.grid[:, 0])
            T_max = max(self.grid[:, 0])
            logg_min = min(self.grid[:, 1])
            logg_max = max(self.grid[:, 1])
            metal_min = min(self.grid[:, 2])
            metal_max = max(self.grid[:, 2])
            alpha_min = min(self.grid[:, 3])
            alpha_max = max(self.grid[:, 3])

            # Check to make sure the requested values fall within the grid
            if (T_min <= T <= T_max and
                            logg_min <= logg <= logg_max and
                            metal_min <= metal <= metal_max and
                            alpha_min <= alpha <= alpha_max):

                y = self.interpolator((T, logg, metal, alpha))
            else:
                warnings.warn("The requested parameters fall outside the model grid. Results may be unreliable!")
                print T, T_min, T_max
                print logg, logg_min, logg_max
                print metal, metal_min, metal_max
                print alpha, alpha_min, alpha_max
                y = self.NN_interpolator((T, logg, metal, alpha))

            # Test to make sure the result is valid. If the requested point is
            # outside the Delaunay triangulation, it will return NaN's
            if np.any(np.isnan(y)):
                warnings.warn("Found NaNs in the interpolated spectrum! Falling back to Nearest Neighbor")
                y = self.NN_interpolator((T, logg, metal, alpha))

        model = DataStructures.xypoint(x=self.xaxis, y=y)
        vsini = (vsini * self.vsini_scale[1] + self.vsini_scale[0]) * units.km.to(units.cm)
        model = Broaden.RotBroad(model, vsini, linear=self.rebin)

        #if fallback and savespectrum:
        #    self.spec.append(model.y)

        #Return the appropriate object
        if return_xypoint:
            return model
        else:
            return model.y
