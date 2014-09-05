__author__ = 'Kevin Gullikson'

"""
This is a general script for doing the cross-correlations in my companion search.
It is called by several smaller scripts in each of the instrument-specific repositories
"""

import FittingUtilities

import numpy as np
import matplotlib.pyplot as plt

import Correlate
import HelperFunctions
import StellarModel


def Process_Data(fname, badregions=[], extensions=True, trimsize=1):
    """

    :param fname: The filename to read in (should be a fits file)
    :param badregions: a list of regions to exclude (contains strong telluric or stellar line residuals)
    :param extensions: A boolean flag for whether the fits file is separated into extensions
    :param trimsize: The amount to exclude from both ends of every order (where it is very noisy)
    :return:
    """
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


        #Remove whole order if it is too small
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
            plt.plot(order.x, order.y / order.cont, 'k-')
            if len(outliers) > 0:
                plt.plot(order.x[outliers], (order.y / order.cont)[outliers], 'r-')
                order.y[outliers] = order.cont[outliers]
                order.cont = FittingUtilities.Continuum(order.x, order.y, lowreject=3, highreject=3)
                order.y[outliers] = order.cont[outliers]

            # Save this order
            orders[numorders - 1 - i] = order.copy()

    plt.show()
    return orders


def CompanionSearch(fileList,
                    badregions=[],
                    extensions=True,
                    resolution=60000,
                    trimsize=1,
                    vsini_values=[10, 20, 30, 40],
                    debug=False):
    model_list = StellarModel.GetModelList()
    modeldict, processed = StellarModel.MakeModelDicts(model_list, vsini_values=vsini_values)


    # Do the cross-correlation
    for temp in sorted(modeldict.keys()):
        for gravity in sorted(modeldict[temp].keys()):
            for metallicity in sorted(modeldict[temp][gravity].keys()):
                for vsini in vsini_values:
                    for fname in fileList:
                        orders = Process_Data(fname, badregions, extensions=extensions, trimsize=trimsize)

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
                        retdict = Correlate.GetCCF(orders,
                                                   model,
                                                   resolution=resolution,
                                                   vsini=vsini,
                                                   rebin_data=True,
                                                   process_model=pflag,
                                                   debug=False,
                                                   outputdir=output_dir.split("Cross_corr")[0])
                        corr = retdict["CCF"]
                        if pflag:
                            processed[temp][gravity][metallicity][vsini] = True
                            modeldict[temp][gravity][metallicity][vsini] = retdict["model"]

                        outfilename = "%s%s.%.0fkps_%sK%+.1f%+.1f" % (
                            output_dir, outfilebase, vsini, temp, gravity, metallicity)
                        print "Outputting to ", outfilename, "\n"
                        np.savetxt(outfilename, np.transpose((corr.x, corr.y)), fmt="%.10g")


                    #Delete the model. We don't need it anymore and it just takes up ram.
                    modeldict[temp][gravity][metallicity][vsini] = []

    return


