"""
    Just a set of helper functions that I use often
    VERY miscellaneous!
"""
import os
import csv
from collections import defaultdict

from scipy.optimize import bisect
from scipy.stats import scoreatpercentile
from scipy.signal import kaiserord, firwin, lfilter
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from astropy.io import fits as pyfits
import numpy as np
from astropy import units, constants
import DataStructures
from lmfit import Model, Parameters
from astropy.time import Time

import pySIMBAD as sim
import SpectralTypeRelations
import readmultispec as multispec


try:
    import emcee
except ImportError:
    print "Warning! emcee module not loaded! BayesFit Module will not be available!"
import FittingUtilities
import mlpy
import warnings


def ensure_dir(f):
    """
      Ensure that a directory exists. Create if it doesn't
    """
    d = os.path.dirname(f)
    if d == "":
        d = f
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
            companion["Magnitude"] = mag2 if filt1 == "V" else "Unknown"
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


def ReadFits(datafile, errors=False, extensions=False, x=None, y=None, cont=None, return_aps=False, debug=False):
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
                       'wavelen': wave,
                       'wavefields': np.zeros(data.shape)}

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
        if return_aps:
            # Return the aperture wavefields too
            orders = [orders, retdict['wavefields']]
    return orders


def OutputFitsFileExtensions(column_dicts, template, outfilename, mode="append", headers_info=[], primary_header={}):
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
    primary_header takes a dictionary with keywords to insert into the primary fits header (and not each extension)
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

    if len(primary_header.keys()) > 0:
        for key in primary_header:
            hdulist[0].header[key] = primary_header[key]


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
    likelihood = lambda pars, data, model_fcn: np.sum(-(data.y - model_fcn(data.x, *pars)) ** 2 / (2.0 * data.err ** 2))
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
        return priors, sampler
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


def IsListlike(arg):
    """This function just test to check if the object acts like a list

    :param arg:
    :return:
    """
    if isinstance(arg, basestring):  # Python 3: isinstance(arg, str)
        return False
    try:
        tmp = [x for x in arg]
        return True
        # return '<' + ", ".join(srepr(x) for x in arg) + '>'
    except TypeError:  # catch when for loop fails
        return False


class ListModel(Model):
    """
    Subclass of lmfit's Model, which can take a list of xypoints.
    The fit method reforms the list into a single array, and then
    passes off to the lmfit method.

    This is very bare bones now (Sep 25, 2014). Will probably need to add more later.
    """

    def __init__(self, fcn, **kws):
        Model.__init__(self, fcn, **kws)


    def fit(self, data, fitcont=True, fit_kws=None, **kws):
        x = np.hstack([d.x for d in data])
        y = np.hstack([d.y for d in data])
        w = np.hstack([1.0 / d.err for d in data])
        self.order_lengths = [d.size() for d in data]
        kws['x'] = x
        self.fitcont = fitcont
        output = Model.fit(self, y, weights=w, fit_kws=fit_kws, **kws)

        # Need to re-shape the best-fit
        best_fit = []
        length = 0
        for i in range(len(data)):
            best_fit.append(output.best_fit[length:length + data[i].size()])
            length += data[i].size()
        output.best_fit = best_fit
        return output


    def _residual(self, params, data, weights=None, **kwargs):
        "default residual:  (data-model)*weights"
        #Make sure the parameters are in the right format
        if not isinstance(params, Parameters):
            if 'names' in kwargs:
                parnames = kwargs['names']
            else:
                raise KeyError ("Must give the parameter names if the params are just list instances!")
            parnames = list(self.param_names)
            d = {name: value for name, value in zip(parnames, params)}
            print d
            params = self.make_params(**d)
        #print params

        model = Model.eval(self, params, **kwargs)
        length = 0
        loglikelihood = []
        for i, l in enumerate(self.order_lengths):
            x = kwargs['x'][length:length + l]
            y = data[length:length + l]
            m = model[length:length + l]
            ratio = y / m
            if self.fitcont:
                cont = FittingUtilities.Continuum(x, ratio, fitorder=5, lowreject=2, highreject=2)
            else:
                cont = np.ones(x.size)
            loglikelihood.append((y - cont * m) ** 2)

            length += l

        loglikelihood = np.hstack(loglikelihood)
        if weights is not None:
            loglikelihood *= weights
        return loglikelihood

    def MCMC_fit(self, data, priors, names, prior_type='flat', fitcont=True, model_getter=None):
        """
        Do a fit using emcee

        :param data: list of xypoints
        :param priors: list of priors (each value must be a 2-D list)
        :param names: The names of the variables, in the same order as the priors list
        :keyword prior_type: The type of prior. Choices are 'flat' or 'gaussian'
        :keyword fitcont: Should we fit the continuum in each step?
        :param fit_kws:
        :param kws:
        :return:
        """
        x = np.hstack([d.x for d in data])
        y = np.hstack([d.y for d in data])
        c = np.hstack([d.cont for d in data])
        e = np.hstack([d.err for d in data])
        fulldata = DataStructures.xypoint(x=x, y=y, err=e, cont=c)
        weights = 1.0 / e ** 2
        self.order_lengths = [d.size() for d in data]
        self.fitcont = fitcont

        # Define the prior functions
        priors = np.array(priors)
        if prior_type.lower() == 'gauss':
            lnprior = lambda pars, prior_vals: np.sum(-(pars - prior_vals[:, 0]) ** 2 / (2.0 * prior_vals[:, 1] ** 2))
            guess = [p[0] for p in priors]
            scale = [p[1] / 10.0 for p in priors]
        elif prior_type.lower() == 'flat':
            def lnprior(pars, prior_vals):
                tmp = [prior_vals[i][0] < pars[i] < prior_vals[i][1] for i in range(len(pars))]
                return 0.0 if all(tmp) else -np.inf

            guess = [(p[0] + p[1]) / 2.0 for p in priors]
            scale = [(p[1] - p[0]) / 10.0 for p in priors]
        else:
            raise ValueError("prior_type must be one of 'gauss' or 'flat'")

        # Define the full probability functions
        def lnprob(pars, priors, data, weights, **kwargs):
            lp = lnprior(pars, priors)
            if not np.isfinite(lp):
                return -np.inf
            return lp + self._residual(pars, data, weights, **kwargs)


        # Set up the emcee sampler
        ndim = len(priors)
        nwalkers = 100
        pars = np.array(guess)
        pos = [pars + scale * np.random.randn(ndim) for i in range(nwalkers)]
        if model_getter is None:
            model_getter = self.opts['model_getter']
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(priors, fulldata, fulldata.err),
                                        kwargs={'model_getter': model_getter, 'names': names, 'x': x})

        return sampler, pos
        """

        # Run the sampler
        mcmc_output = sampler.run_mcmc(pos, 300)


        #output = BayesFit(fulldata, self.func, priors, burn_in=200, nsamples=500, nthreads=1, full_output=True)
        return output
        """


def mad(arr):
    """
    Median average deviation
    :param arr: A list-like object
    :return:
    """
    if not IsListlike(arr):
        raise ValueError("The input to mad must be a list-like object!")

    median = np.median(arr)
    arr = np.array(arr)
    return np.median(np.abs(arr - median))


def radec2altaz(ra, dec, obstime, lat=None, long=None, debug=False):
    """
    calculates the altitude and azimuth, given an ra, dec, time, and observatory location
    :param ra: right ascension of the target (in degrees)
    :param dec: declination of the target (in degrees)
    :param obstime: an astropy.time.Time object containing the time of the observation.
                    Can also contain the observatory location
    :param lat: The latitude of the observatory. Not needed if given in the obstime object
    :param long: The longitude of the observatory. Not needed if given in the obstime object
    :return:
    """

    if lat is None:
        lat = obstime.lat.degree
    if long is None:
        long = obstime.lon.degree
    obstime = Time(obstime.isot, format='isot', scale='utc', location=(long, lat))

    # Find the number of days since J2000
    j2000 = Time("2000-01-01T12:00:00.0", format='isot', scale='utc')
    dt = (obstime - j2000).value  # number of days since J2000 epoch

    # get the UT time
    tstring = obstime.isot.split("T")[-1]
    segments = tstring.split(":")
    ut = float(segments[0]) + float(segments[1]) / 60.0 + float(segments[2]) / 3600

    # Calculate Local Sidereal Time
    lst = obstime.sidereal_time('mean').deg

    # Calculate the hour angle
    HA = lst - ra
    while HA < 0.0 or HA > 360.0:
        s = -np.sign(HA)
        HA += s * 360.0

    # convert everything to radians
    dec *= np.pi / 180.0
    ra *= np.pi / 180.0
    lat *= np.pi / 180.0
    long *= np.pi / 180.0
    HA *= np.pi / 180.0

    # Calculate the altitude
    alt = np.arcsin(np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(HA))

    # calculate the azimuth
    az = np.arccos((np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat)))
    if np.sin(HA) > 0:
        az = 2.0 * np.pi - az

    if debug:
        print "UT: ", ut
        print "LST: ", lst
        print "HA: ", HA * 180.0 / np.pi

    return alt * 180.0 / np.pi, az * 180.0 / np.pi


def convert_hex_string(string, delimiter=":"):
    """
    Converts a hex coordinate string to a decimal
    :param string: The string to convert
    :param delimiter: The delimiter
    :return: the decimal number
    """
    segments = string.split(delimiter)
    s = np.sign(float(segments[0]))
    return s * (abs(float(segments[0])) + float(segments[1]) / 60.0 + float(segments[2]) / 3600.0)


def GetZenithDistance(header=None, date=None, ut=None, ra=None, dec=None, lat=None, long=None, debug=False):
    """
    Function to get the zenith distance to an object
    :param header: the fits header (or a dictionary with the keys 'date-obs', 'ra', and 'dec')
    :param date: The UT date of the observation (only used if header not given)
    :param ut: The UTC time of the observation (only used if header not given)
    :param ra: The right ascension of the observation, in degrees (only used if header not given)
    :param dec: The declination of the observation, in degrees (only used if header not given)
    :param lat: The latitude of the observatory, in degrees
    :param long: The longitude of the observatory, in degrees
    :return: The zenith distance of the object, in degrees
    """

    if header is None:
        obstime = Time("{}T{}".format(date, ut), format='isot', scale='utc', location=(long, lat))
    else:
        obstime = Time(header['date-obs'], format='isot', scale='utc', location=(long, lat))
        delimiter = ":" if ":" in header['ra'] else " "
        ra = 15.0 * convert_hex_string(header['ra'], delimiter=delimiter)
        dec = convert_hex_string(header['dec'], delimiter=delimiter)

    if debug:
        print ra, dec
    alt, az = radec2altaz(ra, dec, obstime, debug=debug)
    return 90.0 - alt


def get_max_velocity(p_spt, s_temp):
    MS = SpectralTypeRelations.MainSequence()
    s_spt = MS.GetSpectralType(MS.Temperature, s_temp, interpolate=True)
    R1 = MS.Interpolate(MS.Radius, p_spt)
    T1 = MS.Interpolate(MS.Temperature, p_spt)
    M1 = MS.Interpolate(MS.Mass, p_spt)
    M2 = MS.Interpolate(MS.Mass, s_spt)
    G = constants.G.cgs.value
    Msun = constants.M_sun.cgs.value
    Rsun = constants.R_sun.cgs.value
    v2 = 2.0 * G * Msun * (M1 + M2) / (Rsun * R1 * (T1 / s_temp) ** 2)
    return np.sqrt(v2) * units.cm.to(units.km)



