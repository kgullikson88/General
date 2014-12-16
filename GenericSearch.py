__author__ = 'Kevin Gullikson'

"""
This is a general script for doing the cross-correlations in my companion search.
It is called by several smaller scripts in each of the instrument-specific repositories
"""

import FittingUtilities

import numpy as np
import DataStructures

import Correlate
import HelperFunctions
import StellarModel

try:
    import pyraf

    pyraf_import = True
except ImportError:
    pyraf_import = False
from astropy.io import fits
from astropy.time import Time
import subprocess
from collections import defaultdict
import StarData
import SpectralTypeRelations
import GenericSmooth
import Broaden
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import re
import sys

if pyraf_import:
    pyraf.iraf.noao()
    pyraf.iraf.noao.rv()


def convert(coord, delim=":"):
    segments = coord.split(delim)
    s = -1.0 if "-" in segments[0] else 1.0
    return s * (abs(float(segments[0])) + float(segments[1]) / 60.0 + float(segments[2]) / 3600.0)


if pyraf_import:
    def HelCorr_IRAF(header, observatory="CTIO", debug=False):
        """
        Get the heliocentric correction for an observation
        """
        jd = header['jd']
        t = Time(jd, format='jd', scale='utc')
        dt = t.datetime
        output = pyraf.iraf.noao.rv.rvcorrect(epoch='INDEF',
                                              epoch_vsun='INDEF',
                                              observatory=observatory,
                                              year=dt.year,
                                              month=dt.month,
                                              day=dt.day,
                                              ut=header['ut'],
                                              ra=header['ra'],
                                              dec=header['dec'],
                                              files="",
                                              images="",
                                              input='no',
                                              Stdout=1)
        vbary = float(output[-1].split()[2])
        if debug:
            for line in output:
                print line
        return vbary
else:
    def HelCorr_IRAF(header, observatory="CTIO", debug=False):
        print "pyraf is not installed! Trying to use the idl version!"
        return 1e-3 * HelCorr(header, observatory=observatory, debug=debug)


def HelCorr(header, observatory="CTIO", idlpath="/Applications/itt/idl/bin/idl", debug=False):
    ra = 15.0 * convert(header['RA'])
    dec = convert(header['DEC'])
    jd = float(header['jd'])

    cmd_list = [idlpath,
                '-e',
                ("print, barycorr({:.8f}, {:.8f}, {:.8f}, 0,"
                 " obsname='CTIO')".format(jd, ra, dec)),
    ]
    if debug:
        print "RA: ", ra
        print "DEC: ", dec
        print "JD: ", jd
    output = subprocess.check_output(cmd_list).split("\n")
    if debug:
        for line in output:
            print line
    return float(output[-2])


def Process_Data(fname, badregions=[], interp_regions=[], extensions=True,
                 trimsize=1, vsini=None, logspacing=False, oversample=1.0):
    """

    :param fname: The filename to read in (should be a fits file)
    :param badregions: a list of regions to exclude (contains strong telluric or stellar line residuals)
    :param interp_regions: a list of regions to interpolate over
    :param extensions: A boolean flag for whether the fits file is separated into extensions
    :param trimsize: The amount to exclude from both ends of every order (where it is very noisy)
    :param vsini: the primary star vsini. If given subtract an estimate of the primary star model obtained by
                  denoising and smoothing with a kernel size set by the vsini.
    :logspacing: If true, interpolate each order into a constant log-spacing.
    :return:
    """
    if isinstance(fname, list) and isinstance(fname[0], DataStructures.xypoint):
        orders = fname
    else:
        if extensions:
            orders = HelperFunctions.ReadExtensionFits(fname)

        else:
            orders = HelperFunctions.ReadFits(fname, errors=2)

    numorders = len(orders)
    for i, order in enumerate(orders[::-1]):
        # Trim data, and make sure the wavelength spacing is constant
        xgrid = np.linspace(order.x[trimsize], order.x[-trimsize], order.size() - 2 * trimsize)
        order = FittingUtilities.RebinData(order, xgrid)

        # Smooth the data
        if vsini is not None:
            dx = order.x[1] - order.x[0]
            smooth_factor = 0.8
            theta = GenericSmooth.roundodd(vsini / 3e5 * order.x.mean() / dx * smooth_factor)
            denoised = HelperFunctions.Denoise(order.copy())
            smooth = FittingUtilities.savitzky_golay(denoised.y, theta, 5)
            order.y = order.y - smooth + order.cont.mean()

        # Remove bad regions from the data
        for region in badregions:
            left = np.searchsorted(order.x, region[0])
            right = np.searchsorted(order.x, region[1])
            if left > 0 and right < order.size():
                print "Warning! Bad region covers the middle of order %i" % i
                print "Removing full order!"
                left = 0
                right = order.size()
            order.x = np.delete(order.x, np.arange(left, right))
            order.y = np.delete(order.y, np.arange(left, right))
            order.cont = np.delete(order.cont, np.arange(left, right))
            order.err = np.delete(order.err, np.arange(left, right))

        # Interpolate over interp_regions:
        for region in interp_regions:
            left = np.searchsorted(order.x, region[0])
            right = np.searchsorted(order.x, region[1])
            order.y[left:right] = order.cont[left:right]


        # Remove whole order if it is too small
        remove = False
        if order.x.size <= 1:
            remove = True
        else:
            velrange = 3e5 * (np.median(order.x) - order.x[0]) / np.median(order.x)
            if velrange <= 1050.0:
                remove = True
        if remove:
            print "Removing order %i" % (numorders - 1 - i)
            orders.pop(numorders - 1 - i)
        else:
            # Find outliers from e.g. bad telluric line or stellar spectrum removal.
            order.cont = FittingUtilities.Continuum(order.x, order.y, lowreject=3, highreject=3)
            outliers = HelperFunctions.FindOutliers(order, expand=10, numsiglow=5, numsighigh=5)
            # plt.plot(order.x, order.y / order.cont, 'k-')
            if len(outliers) > 0:
                # plt.plot(order.x[outliers], (order.y / order.cont)[outliers], 'r-')
                order.y[outliers] = order.cont[outliers]
                order.cont = FittingUtilities.Continuum(order.x, order.y, lowreject=3, highreject=3)
                order.y[outliers] = order.cont[outliers]

            # Save this order
            orders[numorders - 1 - i] = order.copy()

    # plt.show()

    # Rebin the data to a constant log-spacing (if requested)
    if logspacing:
        for i, order in enumerate(orders):
            start = np.log(order.x[0])
            end = np.log(order.x[-1])
            neworder = order.copy()
            neworder.x = np.logspace(start, end, order.size() * oversample, base=np.e)
            neworder = FittingUtilities.RebinData(order, neworder.x)
            orders[i] = neworder

    return orders



def process_model(model, data, vsini_model=None, resolution=None, vsini_primary=None,
                  maxvel=1000.0, debug=False, oversample=1, logspacing=True):
    # Read in the model if necessary
    if isinstance(model, str):
        if debug:
            print "Reading in the input model from %s" % model
        x, y = np.loadtxt(model, usecols=(0, 1), unpack=True)
        x = x * units.angstrom.to(units.nm)
        y = 10 ** y
        left = np.searchsorted(x, data[0].x[0] - 10)
        right = np.searchsorted(x, data[-1].x[-1] + 10)
        model = DataStructures.xypoint(x=x[left:right], y=y[left:right])
    elif not isinstance(model, DataStructures.xypoint):
        raise TypeError(
            "Input model is of an unknown type! Must be a DataStructures.xypoint or a string with the filename.")


    # Linearize the x-axis of the model (in log-spacing)
    if logspacing:
        if debug:
            print "Linearizing model"
        xgrid = np.logspace(np.log10(model.x[0]), np.log10(model.x[-1]), model.size())
        model = FittingUtilities.RebinData(model, xgrid)


    # Broaden
    if vsini_model is not None and vsini_model > 1.0 * units.km.to(units.cm):
        if debug:
            print "Rotationally broadening model to vsini = %g km/s" % (vsini_model * units.cm.to(units.km))
        model = Broaden.RotBroad(model, vsini_model, linear=True)


    # Reduce resolution
    if resolution is not None and 5000 < resolution < 500000:
        if debug:
            print "Convolving to the detector resolution of %g" % resolution
        model = FittingUtilities.ReduceResolutionFFT(model, resolution)

    # Divide by the same smoothing kernel as we used for the data
    if vsini_primary is not None:
        smooth_factor = 0.8
        d_logx = np.log(xgrid[1]/xgrid[0])
        theta = GenericSmooth.roundodd(vsini_primary / 3e5 * smooth_factor / d_logx)
        print "Window size = {}\ndlogx = {}\nvsini = {}".format(theta, d_logx, vsini_primary)
        smooth = FittingUtilities.savitzky_golay(model.y, theta, 5)
        model.y = model.y - smooth
        minval = min(model.y)
        model.y += abs(minval)
        model.cont = abs(minval) * np.ones(model.size())

    # Rebin subsets of the model to the same spacing as the data
    model_orders = []
    model_fcn = spline(model.x, model.y)
    if debug:
        model.output("Test_model.dat")
    for i, order in enumerate(data):
        if debug:
            sys.stdout.write("\rGenerating model subset for order %i in the input data" % (i + 1))
            sys.stdout.flush()
        # Find how much to extend the model so that we can get maxvel range.
        dlambda = order.x[order.size() / 2] * maxvel * 1.5 / 3e5
        left = np.searchsorted(model.x, order.x[0] - dlambda)
        right = np.searchsorted(model.x, order.x[-1] + dlambda)
        right = min(right, model.size() - 2)

        # Figure out the log-spacing of the data
        logspacing = np.log(order.x[1] / order.x[0])

        # Finally, space the model segment with the same log-spacing
        start = np.log(model.x[left])
        end = np.log(model.x[right])
        xgrid = np.exp(np.arange(start, end + logspacing, logspacing))

        segment = DataStructures.xypoint(x=xgrid, y=model_fcn(xgrid))
        #segment = FittingUtilities.RebinData(model[left:right + 1].copy(), xgrid)
        segment.cont = FittingUtilities.Continuum(segment.x, segment.y, lowreject=1.5, highreject=5, fitorder=2)
        model_orders.append(segment)



    print "\n"
    return model_orders


def CompanionSearch(fileList,
                    badregions=[],
                    interp_regions=[],
                    extensions=True,
                    resolution=60000,
                    trimsize=1,
                    vsini_values=(10, 20, 30, 40),
                    Tvalues=range(3000, 6900, 100),
                    metal_values=(-0.5, 0.0, +0.5),
                    logg_values=(4.5,),
                    modeldir="models/",
                    vbary_correct=True,
                    observatory="CTIO",
                    addmode="ML",
                    debug=False):
    model_list = StellarModel.GetModelList(model_directory=modeldir,
                                           temperature=Tvalues,
                                           metal=metal_values,
                                           logg=logg_values)
    modeldict, processed = StellarModel.MakeModelDicts(model_list, vsini_values=vsini_values, vac2air=True)

    get_weights = True if addmode.lower() == "weighted" else False
    orderweights = None

    MS = SpectralTypeRelations.MainSequence()

    # Do the cross-correlation
    datadict = defaultdict(list)
    temperature_dict = defaultdict(float)
    vbary_dict = defaultdict(float)
    for temp in sorted(modeldict.keys()):
        for gravity in sorted(modeldict[temp].keys()):
            for metallicity in sorted(modeldict[temp][gravity].keys()):
                for vsini in vsini_values:
                    for fname in fileList:
                        if vbary_correct:
                            if fname in vbary_dict:
                                vbary = vbary_dict[fname]
                            else:
                                vbary = HelCorr_IRAF(fits.getheader(fname), observatory=observatory)
                                vbary_dict[fname] = vbary
                        process_data = False if fname in datadict else True
                        if process_data:
                            orders = Process_Data(fname, badregions, interp_regions=interp_regions,
                                                  extensions=extensions, trimsize=trimsize)
                            header = fits.getheader(fname)
                            spt = StarData.GetData(header['object']).spectype
                            match = re.search('[0-9]', spt)
                            if match is None:
                                spt = spt[0] + "5"
                            else:
                                spt = spt[:match.start() + 1]
                            temperature_dict[fname] = MS.Interpolate(MS.Temperature, spt)
                        else:
                            orders = datadict[fname]

                        output_dir = "Cross_correlations/"
                        outfilebase = fname.split(".fits")[0]
                        if "/" in fname:
                            dirs = fname.split("/")
                            output_dir = ""
                            outfilebase = dirs[-1].split(".fits")[0]
                            for directory in dirs[:-1]:
                                output_dir = output_dir + directory + "/"
                            output_dir = output_dir + "Cross_correlations/"
                        HelperFunctions.ensure_dir(output_dir)

                        model = modeldict[temp][gravity][metallicity][vsini]
                        pflag = not processed[temp][gravity][metallicity][vsini]
                        # if pflag:
                        # orderweights = None
                        retdict = Correlate.GetCCF(orders,
                                                   model,
                                                   resolution=resolution,
                                                   vsini=vsini,
                                                   rebin_data=process_data,
                                                   process_model=pflag,
                                                   debug=debug,
                                                   outputdir=output_dir.split("Cross_corr")[0],
                                                   addmode=addmode,
                                                   orderweights=orderweights,
                                                   get_weights=get_weights,
                                                   prim_teff=temperature_dict[fname])
                        corr = retdict["CCF"]
                        if pflag:
                            processed[temp][gravity][metallicity][vsini] = True
                            modeldict[temp][gravity][metallicity][vsini] = retdict["model"]
                            #orderweights = retdict['weights']
                        if process_data:
                            datadict[fname] = retdict['data']

                        outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}".format(output_dir, outfilebase,
                                                                                              vsini, temp, gravity,
                                                                                              metallicity)
                        print "Outputting to ", outfilename, "\n"
                        if vbary_correct:
                            corr.x += vbary
                        np.savetxt(outfilename, np.transpose((corr.x, corr.y)), fmt="%.10g")

                        if debug:
                            # Save the individual spectral inputs and CCF orders (unweighted)
                            output_dir2 = output_dir.replace("Cross_correlations", "CCF_inputs")
                            HelperFunctions.ensure_dir(output_dir2)
                            HelperFunctions.ensure_dir("%sCross_correlations/" % (output_dir2))

                            for i, (o, m, c) in enumerate(
                                    zip(retdict['data'], retdict['model'], retdict['CCF_orders'])):
                                print "Saving CCF inputs for order {}".format(i + 1)
                                outfilename = "{0:s}Cross_correlations/{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini,
                                    temp, gravity,
                                    metallicity, i + 1)
                                c.output(outfilename)
                                outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.data.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini,
                                    temp, gravity,
                                    metallicity, i + 1)
                                o.output(outfilename)
                                outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.model.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini,
                                    temp, gravity,
                                    metallicity, i + 1)
                                m.output(outfilename)



                    # Delete the model. We don't need it anymore and it just takes up ram.
                    modeldict[temp][gravity][metallicity][vsini] = []

    return


"""
===============================================================
        SLOW COMPANION SEARCH (BUT MORE ACCURATE)
===============================================================
"""

def slow_companion_search(fileList,
                          primary_vsini,
                          badregions=[],
                          interp_regions=[],
                          extensions=True,
                          resolution=60000,
                          trimsize=1,
                          vsini_values=(10, 20, 30, 40),
                          Tvalues=range(3000, 6900, 100),
                          metal_values=(-0.5, 0.0, +0.5),
                          logg_values=(4.5,),
                          modeldir="models/",
                          vbary_correct=True,
                          observatory="CTIO",
                          addmode="ML",
                          debug=False):
    """
    This function runs a companion search, but makes a new model for each file. That is necessary because it
    subtracts the 'smoothed' model from the model spectrum before correlating
    """

    model_list = StellarModel.GetModelList(model_directory=modeldir,
                                           temperature=Tvalues,
                                           metal=metal_values,
                                           logg=logg_values)
    modeldict, processed = StellarModel.MakeModelDicts(model_list, vsini_values=vsini_values,
                                                       vac2air=True, logspace=True)


    get_weights = True if addmode.lower() == "weighted" else False
    orderweights = None

    MS = SpectralTypeRelations.MainSequence()

    # Do the cross-correlation
    datadict = defaultdict(list)
    temperature_dict = defaultdict(float)
    vbary_dict = defaultdict(float)
    for temp in sorted(modeldict.keys()):
        for gravity in sorted(modeldict[temp].keys()):
            for metallicity in sorted(modeldict[temp][gravity].keys()):
                for vsini_sec in vsini_values:
                    # broaden the model
                    model = modeldict[temp][gravity][metallicity][vsini_sec]
                    model = Broaden.RotBroad(model, vsini_sec*u.km.to(u.cm), linear=True)
                    model = FittingUtilities.ReduceResolutionFFT(model, resolution)

                    for fname, vsini_prim in zip(fileList, primary_vsini):
                        if vbary_correct:
                            if fname in vbary_dict:
                                vbary = vbary_dict[fname]
                            else:
                                vbary = HelCorr_IRAF(fits.getheader(fname), observatory=observatory)
                                vbary_dict[fname] = vbary
                        process_data = False if fname in datadict else True
                        if process_data:
                            orders = Process_Data(fname, badregions, interp_regions=interp_regions, logspacing=True,
                                                  extensions=extensions, trimsize=trimsize, vsini=vsini_prim)
                            header = fits.getheader(fname)
                            spt = StarData.GetData(header['object']).spectype
                            match = re.search('[0-9]', spt)
                            if match is None:
                                spt = spt[0] + "5"
                            else:
                                spt = spt[:match.start() + 1]
                            temperature_dict[fname] = MS.Interpolate(MS.Temperature, spt)
                        else:
                            orders = datadict[fname]

                        # Now, process the model
                        model_orders = process_model(model, orders, vsini_primary=vsini_prim, maxvel=1000.0,
                                                     debug=debug, oversample=1, logspacing=True)

                        # Make sure the output directory exists
                        output_dir = "Cross_correlations/"
                        outfilebase = fname.split(".fits")[0]
                        if "/" in fname:
                            dirs = fname.split("/")
                            output_dir = ""
                            outfilebase = dirs[-1].split(".fits")[0]
                            for directory in dirs[:-1]:
                                output_dir = output_dir + directory + "/"
                            output_dir = output_dir + "Cross_correlations/"
                        HelperFunctions.ensure_dir(output_dir)

                        corr = Correlate.Correlate(orders, model_orders, addmode=addmode, outputdir=output_dir,
                                                   get_weights=get_weights, prim_teff=temperature_dict[fname],
                                                   debug=debug)
                        if debug:
                            corr, ccf_orders = corr

                        # Output the ccf
                        outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}".format(output_dir, outfilebase,
                                                                                              vsini_sec, temp, gravity,
                                                                                              metallicity)
                        print "Outputting to ", outfilename, "\n"
                        if vbary_correct:
                            corr.x += vbary
                        np.savetxt(outfilename, np.transpose((corr.x, corr.y)), fmt="%.10g")

                        if debug:
                            # Save the individual spectral inputs and CCF orders (unweighted)
                            output_dir2 = output_dir.replace("Cross_correlations", "CCF_inputs")
                            HelperFunctions.ensure_dir(output_dir2)
                            HelperFunctions.ensure_dir("%sCross_correlations/" % (output_dir2))

                            for i, (o, m, c) in enumerate(zip(orders, model_orders, ccf_orders)):
                                print "Saving CCF inputs for order {}".format(i + 1)
                                outfilename = "{0:s}Cross_correlations/{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini_sec,
                                    temp, gravity,
                                    metallicity, i + 1)
                                c.output(outfilename)
                                outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.data.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini_sec,
                                    temp, gravity,
                                    metallicity, i + 1)
                                o.output(outfilename)
                                outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.model.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini_sec,
                                    temp, gravity,
                                    metallicity, i + 1)
                                m.output(outfilename)



                    # Delete the model. We don't need it anymore and it just takes up ram.
                    modeldict[temp][gravity][metallicity][vsini_sec] = []

    return