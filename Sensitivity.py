import DataStructures
import FittingUtilities
import numpy as np

import Correlate
import SpectralTypeRelations
from PlotBlackbodies import Planck
import Smooth
import RotBroad_Fast as RotBroad
as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from astropy import units, constants
import warnings
import os


home = os.environ['HOME']


def Analyze(data,  # A list of xypoint instances
            model,  #A model spectrum as an xypoint instance
            resolution=None,  #The detector resolution to smooth the model to
            vsini=None,  #The rotational velocity to smooth the model by (in km/s)
            vels=range(-400, 450, 50),  #Velocities to add the model at
            prim_temp=10000.0,  #The temperature of the primary star (can be a list for close binary)
            sec_temp=5000.0,  #The temperature of the secondary star
            age='MS',  #The age of the system (either 'MS' or a number in years)
            smoothing_windowsize=101,  #The window size for a savitzky-golay smoothing
            smoothing_order=5,  #The order of the SV smoothing function
            tolerance=10,  #How far away the highest peak can be from correct to still count as being found (in km/s)
            outdir="Sensitivity/",  #Output directory. Only used if debug == True
            outfilebase="Output",  #Beginning of output filename. Only used for debug=True
            process_model=True,
            model_orders=None,
            debug=False):  # Debugging flag

    #Convert prim_temp to a list if it is not already
    if isinstance(prim_temp, float) or isinstance(prim_temp, int):
        prim_temp = [prim_temp, ]
    elif not isinstance(prim_temp, list):
        raise ValueError("Unrecognized variable type given for prim_temp!")
    prim_radius = []


    #First, we want to get the radius of the primary and secondary stars,
    #  for use in determining the flux ratio
    MS = SpectralTypeRelations.MainSequence()
    for T in prim_temp:
        prim_spt = MS.GetSpectralType(MS.Temperature, T, interpolate=True)
        prim_radius.append(MS.Interpolate(MS.Radius, prim_spt))
    if age == 'MS':
        sec_spt = MS.GetSpectralType(MS.Temperature, sec_temp, interpolate=True)
        sec_radius = MS.Interpolate(MS.Radius, sec_spt)

    elif isinstance(age, float) or isinstance(age, int):
        PMS = SpectralTypeRelations.PreMainSequence(
            pms_tracks_file="%s/Dropbox/School/Research/Stellar_Evolution/Baraffe_Tracks.dat" % home,
            track_source="Baraffe")
        sec_radius = PMS.GetFromTemperature(age, sec_temp, key='Radius')
        f = PMS.GetFactor(sec_temp, key='Radius')
        if f > 1.5:
            warnings.warn("Radius scaling factor to force agreement with MS relations is large (%f)" % f)
        sec_radius *= f


    else:
        raise ValueError("Unrecognized variable type given for age!")


    #Do some initial processing on the model
    if vsini != None:
        model = RotBroad.Broaden(model, vsini * units.km.to(units.cm))
    if resolution != None:
        model = FittingUtilites.ReduceResolution2(model, resolution)
    model.cont = FittingUtilities.Continuum(model.x, model.y, fitorder=9, lowreject=1, highreject=10)
    model_fcn = spline(model.x, model.y)


    #Now, start the loop over velocities
    found = []
    sig = []
    for velocity in vels:
        orders = []
        if debug:
            print "Adding model to data with an RV shift of %g km/s" % velocity
        for i, order in enumerate(data):
            order = order.copy()
            #Re-fit the continuum in the data
            order.cont = FittingUtilities.Continuum(order.x, order.y, fitorder=3, lowreject=1.5, highreject=7)

            # rebin to match the data
            left = np.searchsorted(model.x, order.x[0] - 1)
            right = np.searchsorted(model.x, order.x[-1] + 1)
            model2 = DataStructures.xypoint(x=model.x[left:right])
            model2.y = model_fcn(model2.x * (1.0 + velocity / (constants.c.cgs.value * units.cm.to(units.km))))
            model2 = FittingUtilities.RebinData(model2, order.x)
            model2.cont = FittingUtilities.Continuum(model2.x, model2.y, fitorder=3, lowreject=1.5, highreject=10)

            # scale to be at the appropriate flux ratio
            primary_flux = np.zeros(order.size())
            for T, R in zip(prim_temp, prim_radius):
                primary_flux += Planck(order.x * units.nm.to(units.cm), T) * R ** 2
            secondary_flux = Planck(order.x * units.nm.to(units.cm), sec_temp) * sec_radius ** 2
            scale = secondary_flux / primary_flux
            if debug:
                print "Scale for order %i is %.4g" % (i, np.mean(scale))
            model2.y = (model2.y / model2.cont - 1.0) * scale
            order.y += model2.y * order.cont

            # Smooth data in the same way I would normally
            smoothed = Smooth.SmoothData(order, smoothing_windowsize, smoothing_order)
            order.y /= smoothed.y
            order.cont = FittingUtilities.Continuum(order.x, order.y, fitorder=2)
            orders.append(order.copy())

        #Do the cross-correlation
        #corr = Correlate.PyCorr(orders, resolution=None, models=[model,], vsini=None, debug=debug, save_output=False, outdir=outdir, outfilebase=outfilebase)[0]
        if process_model:
            result = Correlate.GetCCF(orders, model, vsini=0.0, resolution=0.0, process_model=True, debug=debug)
        elif model_orders != None:
            result = Correlate.GetCCF(orders, model_orders, vsini=0.0, resolution=0.0, process_model=False, debug=debug)
        else:
            raise ValueError("Must give model_orders if process_model is False!")
        corr = result["CCF"]

        #output
        if debug:
            outfilename = "%s%s_t%i_v%i" % (outdir, outfilebase, sec_temp, velocity)
            print "Outputting CCF to %s" % outfilename
            np.savetxt(outfilename, np.transpose((corr.x, corr.y)), fmt="%.10g")

        #Check if we found the companion
        idx = np.argmax(corr.y)
        vmax = corr.x[idx]
        fit = FittingUtilities.Continuum(corr.x, corr.y, fitorder=2, lowreject=3, highreject=3)
        corr.y -= fit
        mean = corr.y.mean()
        std = corr.y.std()
        significance = (corr.y[idx] - mean) / std
        if np.abs(vmax - velocity) <= tolerance:
            #Signal found!
            found.append(True)
            sig.append(significance)
        else:
            found.append(False)
            sig.append(None)

    return found, sig

      
