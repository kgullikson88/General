import os
import sys
import FittingUtilities
from re import search

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from astropy.io import fits
from astropy.io import ascii
from astropy import units, constants

import GenericSearch

import StellarModel
import DataStructures
import SpectralTypeRelations
from PlotBlackbodies import Planck
import GenericSmooth
import HelperFunctions
import Broaden
import Correlate


MS = SpectralTypeRelations.MainSequence()


def GetFluxRatio(sptlist, Tsec, xgrid):
    """
      Returns the flux ratio between the secondary star of temperature Tsec
      and the (possibly multiple) primary star(s) given in the
      'sptlist' list (given as spectral types)
      xgrid is a np.ndarray containing the x-coordinates to find the
        flux ratio at (in nm)

    """
    prim_flux = np.zeros(xgrid.size)

    # Determine the flux from the primary star(s)
    for spt in sptlist:
        end = search("[0-9]", spt).end()
        T = MS.Interpolate(MS.Temperature, spt[:end])
        R = MS.Interpolate(MS.Radius, spt[:end])
        prim_flux += Planck(xgrid * units.nm.to(units.cm), T) * R ** 2

    # Determine the secondary star flux
    s_spt = MS.GetSpectralType(MS.Temperature, Tsec)
    R = MS.Interpolate(MS.Radius, s_spt)
    sec_flux = Planck(xgrid * units.nm.to(units.cm), Tsec) * R ** 2

    return sec_flux / prim_flux


def GetMass(spt):
    """
    Returns the mass of the system in solar masses
    spt: Spectral type of the star
    """

    # Get temperature
    end = search("[0-9]", spt).end()
    return MS.Interpolate(MS.Mass, spt[:end])




def Analyze(fileList,
            vsini_secondary=20 * units.km.to(units.cm),
            resolution=60000,
            smooth_factor=0.8,
            vel_list=range(-400, 450, 50),
            companion_file="%s/Dropbox/School/Research/AstarStuff/TargetLists/Multiplicity.csv" % (os.environ["HOME"]),
            vsini_file="%s/School/Research/Useful_Datafiles/Vsini.csv" % (os.environ["HOME"]),
            tolerance=5.0,
            vsini_skip=10,
            vsini_idx=1,
            badregions=[],
            trimsize=1,
            object_keyword="object",
	        modeldir="./",
            debug=False):

    """
      Perform the sensitivity analysis.
    :param fileList:  The files to analyze
    :param vsini_secondary:   The vsini of the secondary star
    :param resolution: The detector resolution
    :param smooth_factor: The factor to multiply the vsini of the primary by when smoothing
    :param vel_list: The list of velocities to try
    :param companion_file: A csv file giving the known companions to each primary star
    :param vsini_file: A csv file giving the vsini of each primary star
    :param vsini_skip: How many lines to skip in vsini_file
    :param vsini_idx: What index the vsini is in in the vsini_file
    :param tolerance: How far off the peak can be and still be counted as a detection (default: 5 km/s)
    :param badregions: Regions to ignore. These should be the same as in Search!
    :param trimsize: The number of pixels to cut out of each echelle order (from both sides)
    :param object_keyword: The keyword of the object name in the fits header
    :param modeldir: The directory storing the models
    :param debug:
    """

    # Define some constants to use
    lightspeed = constants.c.cgs.value * units.cm.to(units.km)

    # Make sure each output file exists:
    logfilenames = {}
    output_directories = {}
    for fname in fileList:
        output_dir = "Sensitivity/"
        outfilebase = fname.split(".fits")[0]
        if "/" in fname:
            dirs = fname.split("/")
            output_dir = ""
            outfilebase = dirs[-1].split(".fits")[0]
            for directory in dirs[:-1]:
                output_dir = output_dir + directory + "/"
            output_dir = output_dir + "Sensitivity/"
        HelperFunctions.ensure_dir(output_dir)
        output_directories[fname] = output_dir

        # Make the summary file
        logfile = open(output_dir + "logfile.dat", "w")
        logfile.write("Sensitivity Analysis:\n*****************************\n\n")
        logfile.write(
            "Filename\t\t\tPrimary Temperature\tSecondary Temperature\tMass (Msun)\tMass Ratio\tVelocity\tPeak Correct?\tSignificance\n")
        logfile.close()
        logfilenames[fname] = output_dir + "logfile.dat"


    # Read in the companion file
    companions = ascii.read(companion_file)[20:]

    # Read in the vsini file
    vsini_data = ascii.read(vsini_file)[vsini_skip:]

    # Now, start loop over the models:
    model_list = StellarModel.GetModelList(metal=[0, ], temperature=range(3000, 6100, 100), model_directory=modeldir)
    for modelnum, modelfile in enumerate(model_list):
        temp, gravity, metallicity = StellarModel.ClassifyModel(modelfile)
        print "Reading in file %s" % modelfile
        x, y, c = np.loadtxt(modelfile, usecols=(0, 1, 2), unpack=True)
        print "Processing file..."
        # c = FittingUtilities.Continuum(x, y, fitorder=2, lowreject=1.5, highreject=5)
        n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4  #Index of refraction of air
        model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm) / n, y=10 ** y, cont=10 ** c)
        model = FittingUtilities.RebinData(model, np.linspace(model.x[0], model.x[-1], model.size()))
        model = Broaden.RotBroad(model, vsini_secondary)
        model = Broaden.ReduceResolution2(model, resolution)
        modelfcn = interp(model.x, model.y / model.cont)


        # Now that we have a spline function for the broadened data,
        # begin looping over the files
        for fname in fileList:
            print fname
            output_dir = output_directories[fname]
            outfile = open(logfilenames[fname], "a")

            # Read in and process the data like I am about to look for a companion
            orders_original = HelperFunctions.ReadExtensionFits(fname)
            # orders_original = GenericSearch.Process_Data(fname, badregions=badregions, trimsize=trimsize)

            #Find the vsini of the primary star with my spreadsheet
            starname = fits.getheader(fname)[object_keyword]
            found = False
            for data in vsini_data:
                if data[0] == starname:
                    vsini = abs(float(data[vsini_idx]))
                    found = True
            if not found:
                sys.exit("Cannot find %s in the vsini data: %s" % (starname, vsini_file))

            if debug:
                print starname, vsini

            # Check for companions in my master spreadsheet
            known_stars = []
            if starname in companions.field(0):
                row = companions[companions.field(0) == starname]
                known_stars.append(row['col1'].item())
                ncompanions = int(row['col4'].item())
                for comp in range(ncompanions):
                    spt = row["col%i" % (6 + 4 * comp)].item()
                    if not "?" in spt and (spt[0] == "O" or spt[0] == "B" or spt[0] == "A" or spt[0] == "F"):
                        sep = row["col%i" % (7 + 4 * comp)].item()
                        if (not "?" in sep) and float(sep) < 4.0:
                            known_stars.append(spt)
            else:
                sys.exit("Star ({:s}) not found in multiplicity library ({:s}!".format(starname, companion_file))

            # Determine the age of the system and properties of the primary and secondary star
            primary_spt = known_stars[0]
            end = search("[0-9]", primary_spt).end()
            primary_temp = MS.Interpolate(MS.Temperature, primary_spt[:end])
            primary_mass = GetMass(primary_spt)
            secondary_spt = MS.GetSpectralType(MS.Temperature, temp)
            secondary_mass = GetMass(secondary_spt)
            massratio = secondary_mass / primary_mass

            for rv in vel_list:
                print "Testing model with rv = ", rv
                orders = [order.copy() for order in orders_original]  # Make a copy of orders
                model_orders = []
                for ordernum, order in enumerate(orders):
                    # Get the flux ratio
                    scale = GetFluxRatio(known_stars, temp, order.x)
                    if debug:
                        print "Scale factor for order %i is %.3g" % (ordernum, scale.mean())

                    # Add the model to the data
                    model = (modelfcn(order.x * (1.0 + rv / lightspeed)) - 1.0) * scale
                    order.y += model * order.cont


                    # Smooth data using the vsini of the primary star
                    dx = order.x[1] - order.x[0]
                    npixels = max(21, GenericSmooth.roundodd(vsini / lightspeed * order.x.mean() / dx * smooth_factor))
                    smoothed = GenericSmooth.SmoothData(order,
                                                        windowsize=npixels,
                                                        smoothorder=3,
                                                        lowreject=3,
                                                        highreject=3,
                                                        expand=10,
                                                        numiters=10,
                                                        normalize=False)
                    order.y /= smoothed.y
                    orders[ordernum] = order


                # Trim the data now
                orders = GenericSearch.Process_Data(orders, badregions=badregions, trimsize=trimsize)
                orderweights = []
                for ordernum, order in enumerate(orders):
                    # log-space the data
                    start = np.log(order.x[0])
                    end = np.log(order.x[-1])
                    xgrid = np.logspace(start, end, order.size(), base=np.e)
                    logspacing = np.log(xgrid[1] / xgrid[0])
                    order = FittingUtilities.RebinData(order, xgrid)

                    # Generate a model with the same log-spacing (and no rv shift)
                    dlambda = order.x[order.size() / 2] * 1000 * 1.5 / lightspeed
                    start = np.log(order.x[0] - dlambda)
                    end = np.log(order.x[-1] + dlambda)
                    xgrid = np.exp(np.arange(start, end + logspacing, logspacing))
                    model = DataStructures.xypoint(x=xgrid, cont=np.ones(xgrid.size))
                    model.y = modelfcn(xgrid)

                    # Save model order
                    model_orders.append(model)
                    orders[ordernum] = order
                    scale = GetFluxRatio(known_stars, temp, order.x)
                    orderweights.append(scale.mean())

                # Do the actual cross-correlation
                print "Cross-correlating..."
                corr = Correlate.Correlate(orders,
                                           model_orders,
                                           debug=debug,
                                           outputdir="Sensitivity_Testing/",
                                           addmode="ML",
                                           orderweights=orderweights)

                # Check if we found the companion
                idx = np.argmax(corr.y)
                vmax = corr.x[idx]
                fit = FittingUtilities.Continuum(corr.x, corr.y, fitorder=2, lowreject=3, highreject=2.5)
                corr.y -= fit
                goodindices = np.where(np.abs(corr.x - rv) > 100)[0]
                mean = corr.y[goodindices].mean()
                std = corr.y[goodindices].std()
                significance = (corr.y[idx] - mean) / std
                if debug:
                    corrfile = "%s%s_t%i_v%i" % (output_dir, fname.split("/")[-1].split(".fits")[0], temp, rv)
                    print "Outputting CCF to %s" % corrfile
                    np.savetxt(corrfile, np.transpose((corr.x, corr.y - mean, np.ones(corr.size()) * std)),
                               fmt="%.10g")
                if abs(vmax - rv) <= tolerance:
                    #Found
                    outfile.write("%s\t%i\t\t\t%i\t\t\t\t%.2f\t\t%.4f\t\t%i\t\tyes\t\t%.2f\n" % (
                        fname, primary_temp, temp, secondary_mass, massratio, rv, significance))
                else:
                    #Not found
                    outfile.write("%s\t%i\t\t\t%i\t\t\t\t%.2f\t\t%.4f\t\t%i\t\tno\t\tN/A\n" % (
                        fname, primary_temp, temp, secondary_mass, massratio, rv))
                print "Done with rv ", rv
            outfile.close()
