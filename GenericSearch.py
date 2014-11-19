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
import re

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


def Process_Data(fname, badregions=[], interp_regions=[], extensions=True, trimsize=1):
    """

    :param fname: The filename to read in (should be a fits file)
    :param badregions: a list of regions to exclude (contains strong telluric or stellar line residuals)
    :param interp_regions: a list of regions to interpolate over
    :param extensions: A boolean flag for whether the fits file is separated into extensions
    :param trimsize: The amount to exclude from both ends of every order (where it is very noisy)
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
                #plt.plot(order.x[outliers], (order.y / order.cont)[outliers], 'r-')
                order.y[outliers] = order.cont[outliers]
                order.cont = FittingUtilities.Continuum(order.x, order.y, lowreject=3, highreject=3)
                order.y[outliers] = order.cont[outliers]

            # Save this order
            orders[numorders - 1 - i] = order.copy()

    # plt.show()
    return orders


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
                        #    orderweights = None
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

                        outfilename = "%s%s.%.0fkps_%sK%+.1f%+.1f" % (
                            output_dir, outfilebase, vsini, temp, gravity, metallicity)
                        print "Outputting to ", outfilename, "\n"
                        if vbary_correct:
                            corr.x += vbary
                        np.savetxt(outfilename, np.transpose((corr.x, corr.y)), fmt="%.10g")


                    # Delete the model. We don't need it anymore and it just takes up ram.
                    modeldict[temp][gravity][metallicity][vsini] = []

    return


