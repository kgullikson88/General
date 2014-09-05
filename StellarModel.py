__author__ = 'Kevin Gullikson'
import os
import sys
import re
from collections import defaultdict

import numpy as np
from astropy import units

import HelperFunctions
import DataStructures


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


def GetModelList(type='phoenix', metal=[-0.5, 0, 0.5], logg=[4.5, ], temperature=range(3000, 6900, 100), model_directory=modeldir):
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

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    return temp, gravity, metallicity


def MakeModelDicts(model_list, vsini_values=[10, 20, 30, 40], type='phoenix'):
    """This will take a list of models, and output two dictionaries that are
    used by GenericSearch.py and Sensitivity.py

    :param model_list: A list of model filenames
    :param vsini_values: a list of vsini values to broaden the spectrum by (we do that later!)
    :param type: the type of models. Currently, only phoenix is implemented
    :return: A dictionary containing the model with keys of temperature, gravity, metallicity, and vsini,
             and another one with a processed flag with the same keys
    """
    modeldict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint))))
    processed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool))))
    if type.lower() == 'phoenix':
        for fname in model_list:
            temp, gravity, metallicity = ClassifyModel(fname)
            print "Reading in file %s" % fname
            x, y = np.loadtxt(fname, usecols=(0, 1), unpack=True)
            model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm), y=10 ** y)
            for vsini in vsini_values:
                modeldict[temp][gravity][metallicity][vsini] = model
                processed[temp][gravity][metallicity][vsini] = False

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    return modeldict, processed