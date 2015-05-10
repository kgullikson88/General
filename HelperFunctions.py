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
from scipy.interpolate import InterpolatedUnivariateSpline as spline, UnivariateSpline

from astropy.io import fits as pyfits
import numpy as np
from astropy import units, constants
from astropy.time import Time
import DataStructures

import pySIMBAD as sim
import SpectralTypeRelations
import readmultispec as multispec
import Fitters


try:
    import emcee
    emcee_import = True
except ImportError:
    print "Warning! emcee module not loaded! BayesFit Module will not be available!"
    emcee_import = False
import FittingUtilities
try:
    import mlpy
    mlpy_import = True
except ImportError:
    print "Warning! mlpy not loaded! Denoise will not be available"
    mlpy_import = False
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

    # Get absolute magnitude of the primary star, so that we can determine
    # the temperature of the secondary star from the magnitude difference
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
        # Star not in SB9
        return False

    # Now, get summary information for our star
    infile = open("%s/Main.dta" % SB9_location)
    lines = infile.readlines()
    infile.close()

    companion = {}

    num_matches = 0
    for line in lines:
        segments = line.split("|")
        if int(segments[0]) == index:
            num_matches += 1
            # information found
            component = segments[3]
            mag1 = float(segments[4]) if len(segments[4]) > 0 else "Unknown"
            filt1 = segments[5]
            mag2 = float(segments[6]) if len(segments[6]) > 0 else "Unknown"
            filt2 = segments[7]
            spt1 = segments[8]
            spt2 = segments[9]
            companion["Magnitude"] = mag2 if filt1 == "V" else "Unknown"
            companion["SpT"] = spt2

    # Finally, get orbit information for our star (Use the most recent publication)
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
        # At least x and y should be given (and should be strings to identify the field in the table record array)
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
        # Call Rick White's script
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

        # Check if wavelength units are in angstroms (common, but I like nm)
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

        # Add keywords to extension header
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
    featuresize = data.x.mean() * vel / constants.c.cgs.value  # vel MUST be given in units of cm
    dlam = data.x[1] - data.x[0]  # data.x MUST have constant x-spacing
    Npix = featuresize / dlam
    cutoff_hz = 1.0 / Npix  # Cutoff frequency of the filter
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

    # Extend data to prevent edge effects
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
    featuresize = 2 * data.x.mean() * vel / constants.c.cgs.value  # vel MUST be given in units of cm
    dlam = data.x[1] - data.x[0]  # data.x MUST have constant x-spacing
    Npix = featuresize / dlam

    nsamples = data.size()
    sample_rate = 1.0 / dlam
    nyq_rate = sample_rate / 2.0  # The Nyquist rate of the signal.
    width /= nyq_rate
    cutoff_hz = min(1.0 / featuresize, nyq_rate - width * nyq_rate / 2.0)  # Cutoff frequency of the filter

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    if N % 2 == 0:
        N += 1

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero=False)

    # Extend data to prevent edge effects
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


if mlpy_import:
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

        # Adjust all wavelet coefficients
        WC = WC + a * WC * np.exp(-WC ** 2 / (12 * sigma ** 2))

        # Now, do a soft threshold
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

if emcee_import:
    BayesFit = Fitters.BayesFit


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

    # Remove duplicates from 'exclude'
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


ListModel = Fitters.ListModel


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


def split_radec(radec, to_float=False):
    """
    Splits an RA/DEC string into separate RA and DEC strings
    :param radec: The string of the form "00 10 02.20293 +11 08 44.9280"
    :keyword to_float: If true, it will convert the RA and DEC values to floats
    """
    delim = '+' if '+' in radec else '-'
    segments = radec.split(delim)
    ra = segments[0].strip()
    dec = delim + segments[1].strip()

    if to_float:
        ra = 15 * convert_hex_string(ra, delimiter=' ')
        dec = convert_hex_string(dec, delimiter=' ')

    return ra, dec


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


def safe_convert(s, default=0):
    try:
        v = float(s)
    except ValueError:
        v = default
    return v


def convert_hex_string(string, delimiter=":", debug=False):
    """
    Converts a hex coordinate string to a decimal
    :param string: The string to convert
    :param delimiter: The delimiter
    :return: the decimal number
    """
    if debug:
        print('Parsing hex string {}'.format(string))
    segments = string.split(delimiter)
    s = -1.0 if '-' in string else 1.0
    return s * (abs(safe_convert(segments[0])) + safe_convert(segments[1]) / 60.0 + safe_convert(segments[2]) / 3600.0)


def convert_to_hex(val, delimiter=':', force_sign=False, debug=False):
    """
    Converts a numerical value into a hexidecimal string
    """
    s = np.sign(val)
    s_factor = 1 if s > 0 else -1
    val = np.abs(val)
    degree = int(val)
    minute = int((val  - degree)*60)
    second = (val - degree - minute/60.0)*3600.
    if degree == 0 and s_factor < 0:
        deg_str = '-00'
        return '-00{2:s}{0:02d}{2:s}{1:.2f}'.format(minute, second, delimiter)
    elif force_sign or s_factor < 0:
        deg_str = '{:+03d}'.format(degree * s_factor)    
    else:
        deg_str = '{:02d}'.format(degree * s_factor)
    return '{0:s}{3:s}{1:02d}{3:s}{2:.2f}'.format(deg_str, minute, second, delimiter)


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
    s_spt = MS.GetSpectralType('temperature', s_temp, prec=1e-3)
    R1 = MS.Interpolate('radius', p_spt)
    T1 = MS.Interpolate('temperature', p_spt)
    M1 = MS.Interpolate('mass', p_spt)
    M2 = MS.Interpolate('mass', s_spt)
    G = constants.G.cgs.value
    Msun = constants.M_sun.cgs.value
    Rsun = constants.R_sun.cgs.value
    v2 = 2.0 * G * Msun * (M1 + M2) / (Rsun * R1 * (T1 / s_temp) ** 2)
    return np.sqrt(v2) * units.cm.to(units.km)


def FindOrderNums(orders, wavelengths):
    """
      Given a list of xypoint orders and
      another list of wavelengths, this
      finds the order numbers with the
      requested wavelengths
    """
    nums = []
    for wave in wavelengths:
        for i, order in enumerate(orders):
            if order.x[0] < wave and order.x[-1] > wave:
                nums.append(i)
                break
    return nums


RobustFit = Fitters.RobustFit


def add_magnitudes(mag_list):
    """
    Combine magnitudes in the right way
    :param mag_list: a list-like object of magnitudes
    :return: the total magnitude
    """
    flux_list = [10 ** (-m / 2.5) for m in mag_list]
    total_flux = np.sum(flux_list)
    total_mag = -2.5 * np.log10(total_flux)
    return total_mag


def fwhm(x, y, k=10, ret_roots=False):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the dataset.  The function
    uses a spline interpolation with smoothing parameter k ('s' in scipy.interpolate.UnivariateSpline).
    """

    class MultiplePeaks(Exception):
        pass

    class NoPeaksFound(Exception):
        pass

    half_max = np.max(y) / 2.0
    s = UnivariateSpline(x, y - half_max, s=k)
    roots = s.roots()

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                            "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                           "the dataset is flat (e.g. all zeros).")
    else:
        if ret_roots:
            return roots[0], roots[1]

        return abs(roots[1] - roots[0])