"""
    Just a set of helper functions that I use often
    VERY miscellaneous!
"""
import os
import csv
from collections import defaultdict
from scipy.misc import factorial
from scipy.optimize import fminbound, fmin, brent, golden, minimize_scalar, bisect
from scipy.linalg import solve_banded
from scipy.stats import scoreatpercentile
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.signal import kaiserord, firwin, lfilter
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from astropy.io import fits as pyfits
import DataStructures
import numpy as np
from astropy import units, constants

import pySIMBAD as sim
import SpectralTypeRelations
import readmultispec as multispec


try:
    import emcee
except ImportError:
    print "Warning! emcee module not loaded! BayesFit Module will not be available!"
from pysynphot.observation import Observation
from pysynphot.spectrum import ArraySourceSpectrum, ArraySpectralElement
import FittingUtilities
import mlpy
import warnings


def ensure_dir(f):
    """
      Ensure that a directory exists. Create if it doesn't
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def GetStarData(starname):
    """
      Return a dictionary with the SimBad data for a given star.
    """
    link = sim.buildLink(starname)
    star = sim.simbad(link)
    return star


WDS_location = "%s/Dropbox/School/Research/AstarStuff/TargetLists/WDS_MagLimited.csv" % (os.environ["HOME"])


def CheckMultiplicityWDS(starname):
    """
      Check to see if the given star is a binary in the WDS catalog
      If so, return the most recent separation and magnitude of all
      components.
    """
    if type(starname) == str:
        star = GetStarData(starname)
    elif isinstance(starname, sim.simbad):
        star = starname
    else:
        print "Error! Unrecognized variable type in HelperFunctions.CheckMultiplicity!"
        return False

    all_names = star.names()

    # Check if one of them is a WDS name
    WDSname = ""
    for name in all_names:
        if "WDS" in name[:4]:
            WDSname = name
    if WDSname == "":
        return False

    #Get absolute magnitude of the primary star, so that we can determine
    #   the temperature of the secondary star from the magnitude difference
    MS = SpectralTypeRelations.MainSequence()
    print star.SpectralType()[:2]
    p_Mag = MS.GetAbsoluteMagnitude(star.SpectralType()[:2], 'V')


    #Now, check the WDS catalog for this star
    searchpart = WDSname.split("J")[-1].split("A")[0]
    infile = open(WDS_location, 'rb')
    lines = csv.reader(infile, delimiter=";")
    components = defaultdict(lambda: defaultdict())
    for line in lines:
        if searchpart in line[0]:
            sep = float(line[9])
            mag_prim = float(line[10])
            component = line[2]
            try:
                mag_sec = float(line[11])
                s_Mag = p_Mag + (mag_sec - mag_prim)  #Absolute magnitude of the secondary
                s_spt = MS.GetSpectralType(MS.AbsMag, s_Mag)  #Spectral type of the secondary
            except ValueError:
                mag_sec = "Unknown"
                s_spt = "Unknown"
            components[component]["Separation"] = sep
            components[component]["Primary Magnitude"] = mag_prim
            components[component]["Secondary Magnitude"] = mag_sec
            components[component]["Secondary SpT"] = s_spt
    return components


SB9_location = "%s/Dropbox/School/Research/AstarStuff/TargetLists/SB9public" % (os.environ["HOME"])


def CheckMultiplicitySB9(starname):
    """
      Check to see if the given star is a binary in the SB9 catalog
      Ef so, return some orbital information about all the components
    """
    # First, find the record number in SB9
    infile = open("%s/Alias.dta" % SB9_location)
    lines = infile.readlines()
    infile.close()

    index = -1
    for line in lines:
        segments = line.split("|")
        name = segments[1] + " " + segments[2].strip()
        if starname == name:
            index = int(segments[0])
    if index < 0:
        #Star not in SB9
        return False

    #Now, get summary information for our star
    infile = open("%s/Main.dta" % SB9_location)
    lines = infile.readlines()
    infile.close()

    companion = {}

    num_matches = 0
    for line in lines:
        segments = line.split("|")
        if int(segments[0]) == index:
            num_matches += 1
            #information found
            component = segments[3]
            mag1 = float(segments[4]) if len(segments[4]) > 0 else "Unknown"
            filt1 = segments[5]
            mag2 = float(segments[6]) if len(segments[6]) > 0 else "Unknown"
            filt2 = segments[7]
            spt1 = segments[8]
            spt2 = segments[9]
            if filt1 == "V":
                companion["Magnitude"] = mag2
            else:
                companion["Magnitude"] = Unknown  #TODO: work out from blackbody
            companion["SpT"] = spt2

    #Finally, get orbit information for our star (Use the most recent publication)
    infile = open("%s/Orbits.dta" % SB9_location)
    lines = infile.readlines()
    infile.close()

    matches = []
    for line in lines:
        segments = line.split("|")
        if int(segments[0]) == index:
            matches.append(line)
    if len(matches) == 1:
        line = matches[0]
    else:
        date = 0
        line = matches[0]
        for match in matches:
            try:
                year = int(match.split("|")[22][:4])
                if year > date:
                    date = year
                    line = match
            except ValueError:
                continue

    #information found
    period = float(segments[2]) if len(segments[2]) > 0 else "Unknown"
    T0 = float(segments[4]) if len(segments[4]) > 0 else "Unknown"
    e = float(segments[7]) if len(segments[7]) > 0 else "Unknown"
    omega = float(segments[9]) if len(segments[9]) > 0 else "Unknown"
    K1 = float(segments[11]) if len(segments[11]) > 0 else "Unknown"
    K2 = float(segments[13]) if len(segments[13]) > 0 else "Unknown"

    companion["Period"] = period
    companion["Periastron Time"] = T0
    companion["Eccentricity"] = e
    companion["Argument of Periastron"] = omega
    companion["K1"] = K1
    companion["K2"] = K2

    return companion


def BinomialErrors(nobs, Nsamp, alpha=0.16):
    """
    One sided confidence interval for a binomial test.

    If after Nsamp trials we obtain nobs
    trials that resulted in success, find c such that

    P(nobs/Nsamp < mle; theta = c) = alpha

    where theta is the success probability for each trial. 

    Code stolen shamelessly from stackoverflow: 
    http://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
    """
    from scipy.stats import binom

    p0 = float(nobs) / float(Nsamp)
    to_minimise = lambda c: binom.cdf(nobs, Nsamp, c) - alpha
    upper_errfcn = lambda c: binom.cdf(nobs, Nsamp, c) - alpha
    lower_errfcn = lambda c: binom.cdf(nobs, Nsamp, c) - (1.0 - alpha)
    return p0, bisect(lower_errfcn, 0, 1), bisect(upper_errfcn, 0, 1)


def GetSurrounding(full_list, value, return_index=False):
    """
    Takes a list and a value, and returns the two list elements
      closest to the value
    If return_index is True, it will return the index of the surrounding
      elements rather than the elements themselves
    """
    sorter = np.argsort(full_list)
    full_list = sorted(full_list)
    closest = np.argmin([abs(v - value) for v in full_list])
    next_best = closest - 1 if full_list[closest] > value or closest == len(full_list) - 1 else closest + 1
    if return_index:
        return sorter[closest], sorter[next_best]
    else:
        return full_list[closest], full_list[next_best]


def ReadExtensionFits(datafile):
    """
      A convenience function for reading in fits extensions without needing to
      give the name of the standard field names that I use.
    """
    return ReadFits(datafile,
                    extensions=True,
                    x="wavelength",
                    y="flux",
                    cont="continuum",
                    errors="error")


def ReadFits(datafile, errors=False, extensions=False, x=None, y=None, cont=None, debug=False):
    """
    Read a fits file. If extensions=False, it assumes IRAF's multispec format.
    Otherwise, it assumes the file consists of several fits extensions with
    binary tables, with the table names given by the x,y,cont, and errors keywords.

    See ReadExtensionFits for a convenience function that assumes my standard names
    """
    if debug:
        print "Reading in file %s: " % datafile

    if extensions:
        # This means the data is in fits extensions, with one order per extension
        #At least x and y should be given (and should be strings to identify the field in the table record array)
        if type(x) != str:
            x = raw_input("Give name of the field which contains the x array: ")
        if type(y) != str:
            y = raw_input("Give name of the field which contains the y array: ")
        orders = []
        hdulist = pyfits.open(datafile)
        if cont == None:
            if not errors:
                for i in range(1, len(hdulist)):
                    data = hdulist[i].data
                    xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y))
                    orders.append(xypt)
            else:
                if type(errors) != str:
                    errors = raw_input("Give name of the field which contains the errors/sigma array: ")
                for i in range(1, len(hdulist)):
                    data = hdulist[i].data
                    xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), err=data.field(errors))
                    orders.append(xypt)
        elif type(cont) == str:
            if not errors:
                for i in range(1, len(hdulist)):
                    data = hdulist[i].data
                    xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), cont=data.field(cont))
                    orders.append(xypt)
            else:
                if type(errors) != str:
                    errors = raw_input("Give name of the field which contains the errors/sigma array: ")
                for i in range(1, len(hdulist)):
                    data = hdulist[i].data
                    xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), cont=data.field(cont),
                                                  err=data.field(errors))
                    orders.append(xypt)

    else:
        # Data is in multispec format rather than in fits extensions
        #Call Rick White's script
        try:
            retdict = multispec.readmultispec(datafile, quiet=not debug)
        except ValueError:
            warnings.warn("Wavelength not found in file %s. Using a pixel grid instead!" % datafile)
            hdulist = pyfits.open(datafile)
            data = hdulist[0].data
            hdulist.close()
            numpixels = data.shape[-1]
            numorders = data.shape[-2]
            wave = np.array([np.arange(numpixels) for i in range(numorders)])
            retdict = {'flux': data,
                       'wavelen': wave}

        #Check if wavelength units are in angstroms (common, but I like nm)
        hdulist = pyfits.open(datafile)
        header = hdulist[0].header
        hdulist.close()
        wave_factor = 1.0  #factor to multiply wavelengths by to get them in nanometers
        for key in sorted(header.keys()):
            if "WAT1" in key:
                if "label=Wavelength" in header[key] and "units" in header[key]:
                    waveunits = header[key].split("units=")[-1]
                    if waveunits == "angstroms" or waveunits == "Angstroms":
                        #wave_factor = Units.nm/Units.angstrom
                        wave_factor = units.angstrom.to(units.nm)
                        if debug:
                            print "Wavelength units are Angstroms. Scaling wavelength by ", wave_factor

        if errors == False:
            numorders = retdict['flux'].shape[0]
        else:
            numorders = retdict['flux'].shape[1]
        orders = []
        for i in range(numorders):
            wave = retdict['wavelen'][i] * wave_factor
            if errors == False:
                flux = retdict['flux'][i]
                err = np.ones(flux.size) * 1e9
                err[flux > 0] = np.sqrt(flux[flux > 0])
            else:
                if type(errors) != int:
                    errors = int(raw_input("Enter the band number (in C-numbering) of the error/sigma band: "))
                flux = retdict['flux'][0][i]
                err = retdict['flux'][errors][i]
            cont = FittingUtilities.Continuum(wave, flux, lowreject=2, highreject=4)
            orders.append(DataStructures.xypoint(x=wave, y=flux, err=err, cont=cont))
    return orders


def OutputFitsFileExtensions(column_dicts, template, outfilename, mode="append", headers_info=[]):
    """
    Function to output a fits file
    column_dict is a dictionary where the key is the name of the column
       and the value is a np array with the data. Example of a column
       would be the wavelength or flux at each pixel
    template is the filename of the template fits file. The header will
       be taken from this file and used as the main header
    mode determines how the outputted file is made. Append will just add
       a fits extension to the existing file (and then save it as outfilename)
       "new" mode will create a new fits file.
       header_info takes a list of lists. Each sub-list should have size 2 where the first element is the name of the new keyword, and the second element is the corresponding value. A 3rd element may be added as a comment
    """

    # Get header from template. Use this in the new file
    if mode == "new":
        header = pyfits.getheader(template)

    if not isinstance(column_dicts, list):
        column_dicts = [column_dicts, ]
    if len(headers_info) < len(column_dicts):
        for i in range(len(column_dicts) - len(headers_info)):
            headers_info.append([])

    if mode == "append":
        hdulist = pyfits.open(template)
    elif mode == "new":
        header = pyfits.getheader(template)
        pri_hdu = pyfits.PrimaryHDU(header=header)
        hdulist = pyfits.HDUList([pri_hdu, ])

    for i in range(len(column_dicts)):
        column_dict = column_dicts[i]
        header_info = headers_info[i]
        columns = []
        for key in column_dict.keys():
            columns.append(pyfits.Column(name=key, format="D", array=column_dict[key]))
        cols = pyfits.ColDefs(columns)
        tablehdu = pyfits.BinTableHDU.from_columns(cols)

        #Add keywords to extension header
        num_keywords = len(header_info)
        header = tablehdu.header
        for i in range(num_keywords):
            info = header_info[i]
            if len(info) > 2:
                header.set(info[0], info[1], info[2])
            elif len(info) == 2:
                header.set(info[0], info[1])

        hdulist.append(tablehdu)

    hdulist.writeto(outfilename, clobber=True, output_verify='ignore')
    hdulist.close()


def LowPassFilter(data, vel, width=5, linearize=False):
    """
    Function to apply a low-pass filter to data.
      Data must be in an xypoint container, and have linear wavelength spacing
      vel is the width of the features you want to remove, in velocity space (in cm/s)
      width is how long it takes the filter to cut off, in units of wavenumber
    """

    if linearize:
        data = data.copy()
        datafcn = spline(data.x, data.y, k=1)
        errorfcn = spline(data.x, data.err, k=1)
        contfcn = spline(data.x, data.cont, k=1)
        linear = DataStructures.xypoint(data.x.size)
        linear.x = np.linspace(data.x[0], data.x[-1], linear.size())
        linear.y = datafcn(linear.x)
        linear.err = errorfcn(linear.x)
        linear.cont = contfcn(linear.x)
        data = linear

    # Figure out cutoff frequency from the velocity.
    featuresize = data.x.mean() * vel / constants.c.cgs.value  #vel MUST be given in units of cm
    dlam = data.x[1] - data.x[0]  #data.x MUST have constant x-spacing
    Npix = featuresize / dlam
    cutoff_hz = 1.0 / Npix  #Cutoff frequency of the filter
    cutoff_hz = 1.0 / featuresize

    nsamples = data.size()
    sample_rate = 1.0 / dlam
    nyq_rate = sample_rate / 2.0  # The Nyquist rate of the signal.
    width /= nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

    #Extend data to prevent edge effects
    y = np.r_[data.y[::-1], data.y, data.y[::-1]]

    # Use lfilter to filter data with the FIR filter.
    smoothed_y = lfilter(taps, 1.0, y)

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate
    delay_idx = np.searchsorted(data.x, data.x[0] + delay) - 1
    smoothed_y = smoothed_y[data.size() + delay_idx:-data.size() + delay_idx]
    if linearize:
        return linear.x, smoothed_y
    else:
        return smoothed_y


def IterativeLowPass(data, vel, numiter=100, lowreject=3, highreject=3, width=5, linearize=False):
    """
    An iterative version of LowPassFilter.
    It will ignore outliers in the low pass filter
    """

    datacopy = data.copy()
    if linearize:
        datafcn = spline(datacopy.x, datacopy.y, k=3)
        errorfcn = spline(datacopy.x, datacopy.err, k=1)
        contfcn = spline(datacopy.x, datacopy.cont, k=1)
        linear = DataStructures.xypoint(datacopy.x.size)
        linear.x = np.linspace(datacopy.x[0], datacopy.x[-1], linear.size())
        linear.y = datafcn(linear.x)
        linear.err = errorfcn(linear.x)
        linear.cont = contfcn(linear.x)
        datacopy = linear.copy()

    done = False
    iter = 0
    datacopy.cont = FittingUtilities.Continuum(datacopy.x, datacopy.y, fitorder=9, lowreject=2.5, highreject=5)
    while not done and iter < numiter:
        done = True
        iter += 1
        smoothed = LowPassFilter(datacopy, vel, width=width)
        residuals = datacopy.y / smoothed
        mean = np.mean(residuals)
        std = np.std(residuals)
        badpoints = np.where(np.logical_or((residuals - mean) < -lowreject * std, residuals - mean > highreject * std))[
            0]
        if badpoints.size > 0:
            done = False
            datacopy.y[badpoints] = smoothed[badpoints]
    if linearize:
        return linear.x, smoothed
    else:
        return smoothed


def HighPassFilter(data, vel, width=5, linearize=False):
    """
    Function to apply a high-pass filter to data.
      Data must be in an xypoint container, and have linear wavelength spacing
      vel is the width of the features you want to remove, in velocity space (in cm/s)
      width is how long it takes the filter to cut off, in units of wavenumber
    """

    if linearize:
        data = data.copy()
        datafcn = spline(data.x, data.y, k=3)
        errorfcn = spline(data.x, data.err, k=3)
        contfcn = spline(data.x, data.cont, k=3)
        linear = DataStructures.xypoint(data.x.size)
        linear.x = np.linspace(data.x[0], data.x[-1], linear.size())
        linear.y = datafcn(linear.x)
        linear.err = errorfcn(linear.x)
        linear.cont = contfcn(linear.x)
        data = linear

    # Figure out cutoff frequency from the velocity.
    featuresize = 2 * data.x.mean() * vel / constants.c.cgs.value  #vel MUST be given in units of cm
    dlam = data.x[1] - data.x[0]  #data.x MUST have constant x-spacing
    Npix = featuresize / dlam

    nsamples = data.size()
    sample_rate = 1.0 / dlam
    nyq_rate = sample_rate / 2.0  # The Nyquist rate of the signal.
    width /= nyq_rate
    cutoff_hz = min(1.0 / featuresize, nyq_rate - width * nyq_rate / 2.0)  #Cutoff frequency of the filter

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    if N % 2 == 0:
        N += 1

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero=False)

    #Extend data to prevent edge effects
    y = np.r_[data.y[::-1], data.y, data.y[::-1]]

    # Use lfilter to filter data with the FIR filter.
    smoothed_y = lfilter(taps, 1.0, y)

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate
    delay_idx = np.searchsorted(data.x, data.x[0] + delay) - 1
    smoothed_y = smoothed_y[data.size() + delay_idx:-data.size() + delay_idx]
    if linearize:
        return linear.x, smoothed_y
    else:
        return smoothed_y


def Denoise(data):
    """
    This function implements the denoising given in the url below:
    http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4607982&tag=1

    with title "Astronomical Spectra Denoising Based on Simplifed SURE-LET Wavelet Thresholding"
    """
    y, boolarr = mlpy.wavelet.pad(data.y)
    WC = mlpy.wavelet.dwt(y, 'd', 10, 0)
    # Figure out the unknown parameter 'a'
    sum1 = 0.0
    sum2 = 0.0
    numlevels = int(np.log2(WC.size))
    start = 2 ** (numlevels - 1)
    median = np.median(WC[start:])
    sigma = np.median(np.abs(WC[start:] - median)) / 0.6745
    for w in WC:
        phi = w * np.exp(-w ** 2 / (12.0 * sigma ** 2))
        dphi = np.exp(-w ** 2 / (12.0 * sigma ** 2)) * (1 - 2 * w ** 2 / (12 * sigma ** 2) )
        sum1 += sigma ** 2 * dphi
        sum2 += phi ** 2
    a = -sum1 / sum2

    #Adjust all wavelet coefficients
    WC = WC + a * WC * np.exp(-WC ** 2 / (12 * sigma ** 2))

    #Now, do a soft threshold
    threshold = scoreatpercentile(WC, 80.0)
    WC[np.abs(WC) <= threshold] = 0.0
    WC[np.abs(WC) > threshold] -= threshold * np.sign(WC[np.abs(WC) > threshold])

    #Transform back
    y2 = mlpy.wavelet.idwt(WC, 'd', 10)
    data.y = y2[boolarr]
    return data


# Kept for legacy support, since I was using Denoise3 in several codes in the past.
def Denoise3(data):
    return Denoise(data)


def BayesFit(data, model_fcn, priors, limits=None, burn_in=100, nwalkers=100, nsamples=100, nthreads=1,
             full_output=False, a=2):
    """
    This function will do a Bayesian fit to the model.

    Parameter description:
      data:         A DataStructures.xypoint instance containing the data
      model_fcn:    A function that takes an x-array and parameters,
                       and returns a y-array. The number of parameters
                       should be the same as the length of the 'priors'
                       parameter
      priors:       Either a 2d np array or a list of lists. Each index
                       should contain the expected value and the uncertainty
                       in that value (assumes all Gaussian priors!).
      limits:       If given, it should be a list of the same shape as
                       'priors', giving the limits of each parameter
      burn_in:      The burn-in period for the MCMC before you start counting
      nwalkers:     The number of emcee 'walkers' to use.
      nsamples:     The number of samples to use in the MCMC sampling. Note that
                        the actual number of samples is nsamples * nwalkers
      nthreads:     The number of processing threads to use (parallelization)
                        This probably needs MPI installed to work. Not sure though...
      full_ouput:   Return the full sample chain instead of just the mean and
                        standard deviation of each parameter.
      a:            See emcee.EnsembleSampler. Basically, it controls the step size
    """

    # Priors needs to be a np array later, so convert to that first
    priors = np.array(priors)

    # Define the likelihood, prior, and posterior probability functions
    likelihood = lambda pars, data, model_fcn: np.sum(-(data.y - model_fcn(data.x, pars)) ** 2 / (2.0 * data.err ** 2))
    if limits == None:
        prior = lambda pars, priors: np.sum(-(pars - priors[:, 0]) ** 2 / (2.0 * priors[:, 1] ** 2))
        posterior = lambda pars, data, model_fcn, priors: likelihood(pars, data, model_fcn) + prior(pars, priors)
    else:
        limits = np.array(limits)
        prior = lambda pars, priors, limits: -9e19 if any(
            np.logical_or(pars < limits[:, 0], pars > limits[:, 1])) else np.sum(
            -(pars - priors[:, 0]) ** 2 / (2.0 * priors[:, 1] ** 2))
        posterior = lambda pars, data, model_fcn, priors, limits: likelihood(pars, data, model_fcn) + prior(pars,
                                                                                                            priors,
                                                                                                            limits)


    # Set up the MCMC sampler
    ndim = priors.shape[0]
    if limits == None:
        p0 = [np.random.normal(loc=priors[:, 0], scale=priors[:, 1]) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, threads=nthreads, args=(data, model_fcn, priors),
                                        a=4)
    else:
        ranges = np.array([l[1] - l[0] for l in limits])
        p0 = [np.random.rand(ndim) * ranges + limits[:, 0] for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, threads=nthreads,
                                        args=(data, model_fcn, priors, limits), a=a)

    # Burn-in the sampler
    pos, prob, state = sampler.run_mcmc(p0, burn_in)

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Run the sampler
    pos, prob, state = sampler.run_mcmc(pos, nsamples, rstate0=state)

    print "Acceptance fraction = %f" % np.mean(sampler.acceptance_fraction)
    maxprob_indice = np.argmax(prob)
    priors[:, 0] = pos[maxprob_indice]
    #Get the parameter estimates
    chain = sampler.flatchain
    for i in range(ndim):
        priors[i][1] = np.std(chain[:, i])

    if full_output:
        return priors, chain
    return priors


def Gauss(x, mu, sigma, amp=1):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def FindOutliers(data, numsiglow=6, numsighigh=3, numiters=10, expand=0):
    """
    Find outliers in the data. Outliers are defined as
    points that are more than numsiglow standard deviations
    below the mean, or numsighigh standard deviations above
    the mean. Returns the index of the outliers in the data.

    Data should be an xypoint instance
    The expand keyword will expand the rejected points some number
      from every rejected point.
    """

    done = False
    i = 0
    good = np.arange(data.size()).astype(int)

    while not done and i < numiters:
        sig = np.std(data.y[good] / data.cont[good])
        outliers = np.where(np.logical_or(data.y / data.cont - 1.0 > numsighigh * sig,
                                          data.y / data.cont - 1.0 < -numsiglow * sig))[0]
        good = np.where(np.logical_and(data.y / data.cont - 1.0 <= numsighigh * sig,
                                       data.y / data.cont - 1.0 >= -numsiglow * sig))[0]
        i += 1
        if outliers.size < 1:
            break

    # Now, expand the outliers by 'expand' pixels on either
    exclude = []
    for outlier in outliers:
        for i in range(max(0, outlier - expand), min(outlier + expand + 1, data.size())):
            exclude.append(i)

    #Remove duplicates from 'exclude'
    temp = []
    [temp.append(i) for i in exclude if not i in temp]
    return np.array(temp)
  
