"""
  This class performs the analysis of B stars to determine the following:
  - vsini
  -rv
  -Teff
  -logg
  -vmacro
  -vmicro
  -[Fe/H]


  Example usage:
  --------------
  Put usage example here
"""

# import matplotlib
#matplotlib.use("GTKAgg")
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import interp1d
from collections import defaultdict, namedtuple, deque
import os
from os.path import isfile
import warnings
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import DataStructures
import FittingUtilities
from astropy import units, constants
from astropy.io import fits

import HelperFunctions
import Correlate
import SpectralTypeRelations
import Broaden
import astrolib  #Ian Crossfield's script for rv correction



# Define a structure to store my parameter information and associate chi-squared values
ParameterValues = namedtuple("ParameterValues", "Teff, logg, Q, beta, He, Si, vmacro, chisq")


class Analyse():
    def __init__(self, gridlocation="/Volumes/DATADRIVE/Stellar_Models/BSTAR06", SpT=None, debug=False, fname=None,
                 Teff=None, logg=None, resolution=60000.0):
        # Just define some class variables
        self.gridlocation = gridlocation
        self.debug = debug
        self.windval2str = {-13.8: "A",
                            -14.0: "a",
                            -13.4: "B",
                            -13.6: "b",
                            -13.15: "C",
                            -12.7: "D",
                            -14.30: "O"}
        self.grid = defaultdict(lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: defaultdict(
                                    lambda: defaultdict(DataStructures.xypoint)))))))))

        self.Teffs = range(10000, 20000, 500) + range(20000, 31000, 1000)
        self.Heliums = (0.1, 0.15, 0.2)
        self.Silicons = (-4.19, -4.49, -4.79)
        self.logQ = (-14.3, -14.0, -13.8, -13.6, -13.4, -13.15, -12.7)
        self.betas = (0.9, 1.2, 1.5, 2.0, 3.0)
        self.species = ['BR10', 'BR11', 'BRALPHA', 'BRBETA', 'BRGAMMA',
                        'HALPHA', 'HBETA', 'HDELTA', 'HEI170', 'HEI205',
                        'HEI211', 'HEI4010', 'HEI4026', 'HEI4120', 'HEI4140',
                        'HEI4387', 'HEI4471', 'HEI4713', 'HEI4922', 'HEI6678',
                        'HEII218', 'HEII4200', 'HEII4541', 'HEII4686', 'HEII57',
                        'HEII611', 'HEII6406', 'HEII6527', 'HEII6683', 'HEII67',
                        'HEII712', 'HEII713', 'HEPS', 'HGAMMA', 'PALPHA',
                        'PBETA', 'PF10', 'PF9', 'PFGAMMA', 'PGAMMA',
                        'SiII4128', 'SiII4130', 'SiII5041', 'SiII5056', 'SiIII4552',
                        'SiIII4567', 'SiIII4574', 'SiIII4716', 'SiIII4813', 'SiIII4819',
                        'SiIII4829', 'SiIII5739', 'SiIV4089', 'SiIV4116', 'SiIV4212',
                        'SiIV4950', 'SiIV6667', 'SiIV6701']
        self.visible_species = {}

        # Get spectral type if the user didn't enter it
        if SpT is None:
            SpT = raw_input("Enter Spectral Type: ")
        self.SpT = SpT

        #Use the spectral type to get effective temperature and log(g)
        #   (if those keywords were not given when calling this)
        MS = SpectralTypeRelations.MainSequence()
        if Teff is None:
            Teff = MS.Interpolate(MS.Temperature, SpT)
        if logg is None:
            M = MS.Interpolate(MS.Mass, SpT)
            R = MS.Interpolate(MS.Radius, SpT)
            G = constants.G.cgs.value
            Msun = constants.M_sun.cgs.value
            Rsun = constants.R_sun.cgs.value
            logg = np.log10(G * M * Msun / (R * Rsun ** 2))
        self.Teff_guess = Teff
        self.logg_guess = logg
        self.vsini = 300
        self.resolution = resolution


        # Read the filename if it is given
        self.data = None
        if fname is not None:
            self.InputData(fname)

            # Initialize a figure for drawing
            #plt.figure(1)
            #plt.plot([1], [1], 'ro')
            #plt.show(block=False)
            #plt.cla()


    def GetModel(self, Teff, logg, Q, beta, helium, silicon, species, vmacro, xspacing=None):
        """
          This method takes the following values, and finds the closest match
            in the grid. It will warn the user if the values are not actual
            grid points, which they should be!

          Parameters:
          -----------
          Teff:           Effective temperature of the star (K)
                          Options: 10000-20000K in 500K steps, 20000-30000 in 1000K steps

          logg:           log of the surface gravity of the star (cgs)
                          Options: 4.5 to 4log(Teff) - 15.02 in 0.1 dex steps

          Q:              log of the wind strength logQ = log(Mdot (R * v_inf)^-1.5
                          Options: -14.3, -14.0, -13.8, -13.6, -13.4, -13.15, -12.7

          beta:           Wind velocity law for the outer expanding atmosphere
                          Options: 0.9, 1.2, 1.5, 2.0, 3.0

          helium:         Helium fraction of the star's atmsophere
                          Options: 0.10, 0.15, 0.20

          silicon:        Relative silicon abundance as log(nSi/nH)
                          Options: -4.19, -4.49, -4.79

          vmacro:         Macroturbulent velocity (km/s)
                          Options: 3,6,10,12,15 for Teff<20000
                                   6,10,12,15,20 for Teff>20000

          species:        The name of the spectral line you want
                          Options: Many. Just check the model grid.

          xspacing:       An optional argument. If provided, we will resample the line
                          to have the given x-axis spacing.
        """

        # Check to see if the input is in the grid
        if Teff not in self.Teffs:
            warnings.warn("Teff value (%g) not in model grid!" % Teff)
            Teff = HelperFunctions.GetSurrounding(self.Teffs, Teff)[0]
            print "\tClosest value is %g\n\n" % Teff
        # logg and vmacro depend on Teff, so make those lists now
        loggs = [round(g, 2) for g in np.arange(4.5, 4 * np.log10(Teff) - 15.02, -0.1)]
        if Teff < 20000:
            self.vmacros = (3, 6, 10, 12, 15)
        else:
            self.vmacros = (6, 10, 12, 15, 20)

        # Continue checking if the inputs are on a grid point
        if logg not in loggs:
            warnings.warn("log(g) value (%g) not in model grid!" % logg)
            logg = HelperFunctions.GetSurrounding(loggs, logg)[0]
            print "\tClosest value is %g\n\n" % logg
        if Q not in self.logQ:
            warnings.warn("log(Q) wind value (%g) not in model grid!" % Q)
            Q = HelperFunctions.GetSurrounding(self.logQ, Q)[0]
            print "\tClosest value is %g\n\n" % Q
        if beta not in self.betas:
            warnings.warn("Beta value (%g) not in model grid!" % beta)
            beta = HelperFunctions.GetSurrounding(self.betas, beta)[0]
            print "\tClosest value is %g\n\n" % beta
        if helium not in self.Heliums:
            warnings.warn("Helium fraction (%g) not in model grid!" % helium)
            helium = HelperFunctions.GetSurrounding(self.Heliums, helium)[0]
            print "\tClosest value is %g\n\n" % helium
        if silicon not in self.Silicons:
            warnings.warn("Silicon relative abundance (%g) not in model grid!" % silicon)
            silicon = HelperFunctions.GetSurrounding(self.Silicons, silicon)[0]
            print "\tClosest value is %g\n\n" % silicon
        if species not in self.species:
            raise ValueError("Desired species ( %s ) not in grid!" % species)
        if vmacro not in self.vmacros:
            warnings.warn("Macroturbulent velocity (%g) not in model grid!" % vmacro)
            vmacro = HelperFunctions.GetSurrounding(self.vmacros, vmacro)[0]
            print "\tClosest value is %g\n\n" % vmacro


        # Now, make the filename that this model (or the closest one) is at
        windstr = self.windval2str[Q]
        abstr = "He%iSi%i" % (helium * 100, -silicon * 100)
        fname = "%s/T%i/g%i/%s%.2i/%s/OUT.%s_VT%.3i.gz" % (
        self.gridlocation, Teff, logg * 10, windstr, beta * 10, abstr, species, vmacro)
        if not isfile(fname):
            warnings.warn("File %s not found! Skipping. This could cause errors later!" % fname)
            return DataStructures.xypoint(x=np.arange(300, 1000, 10))

        # Gunzip that file to a temporary one.
        tmpfile = "tmp%f" % time.time()
        lines = subprocess.check_output(['gunzip', '-c', fname])
        output = open(tmpfile, "w")
        output.writelines(lines)
        output.close()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, y = np.genfromtxt(tmpfile, invalid_raise=False, usecols=(2, 4), unpack=True)

        #Removes NaNs from random extra lines in the files...
        while np.any(np.isnan(y)):
            x = x[:-1]
            y = y[:-1]

        # Check for duplicate x points (why does this grid suck so much?!)
        xdiff = np.array([x[i + 1] - x[i] for i in range(x.size - 1)])
        goodindices = np.where(xdiff > 1e-7)[0]
        x = x[goodindices]
        y = y[goodindices]

        #Convert from angstrom to nm, and switch to air wavelengths
        x = x * units.angstrom.to(units.nm) / 1.00026

        # delete the temporary file
        subprocess.check_call(['rm', tmpfile])

        if xspacing is not None:
            modelfcn = spline(x, y, k=1)
            x = np.arange(x[0], x[-1] + xspacing, xspacing)
            y = modelfcn(x)
        return DataStructures.xypoint(x=x, y=y)


    def InputData(self, fname, resample=True):
        """
          This takes a fits file, and reads it as a bunch of echelle orders.
          It also saves the header for later use.
          If resample==True, it will resample the x-axis to a constant spacing
        """
        orders = HelperFunctions.ReadFits(fname, extensions=True, x="wavelength", y="flux", cont="continuum",
                                          errors="error")
        for i, order in enumerate(orders):
            orders[i].err = np.sqrt(order.y)
        self.data = orders
        if resample:
            for i, order in enumerate(self.data):
                self.data[i] = self._resample(order)
        self.fname = fname.split("/")[-1]
        hdulist = fits.open(fname)
        self.headers = []
        for i in range(len(hdulist)):
            self.headers.append(hdulist[i].header)
        return


    def FindVsini(self, vsini_lines="%s/School/Research/Useful_Datafiles/vsini.list" % os.environ["HOME"]):
        """
          This function will read in the linelist useful for determining vsini.
        For each one, it will ask the user if the line is usable by bringing up
        a plot of the appropriate order in the data. If the user says it is, then
        the vsini is determined as in Simon-Diaz (2007).
        """

        # First, check to make sure the user entered a datafile
        if self.data is None:
            fname = raw_input("Enter filename for the data: ")
            self.InputData(fname)

        # Read in the vsini linelist file
        center, left, right = np.loadtxt(vsini_lines, usecols=(1, 2, 3), unpack=True)
        center /= 10.0
        left /= 10.0
        right /= 10.0

        # Find each line in the data
        plt.figure(1)
        vsini_values = []
        for c, l, r in zip(center, left, right):
            found = False
            for order in self.data:
                if order.x[0] < c < order.x[-1]:
                    found = True
                    break
            if not found:
                continue
            first = np.searchsorted(order.x, l)
            last = np.searchsorted(order.x, r)
            segment = order[first:last]
            segment.cont = FittingUtilities.Continuum(segment.x, segment.y, fitorder=1, lowreject=1, highreject=5)
            segment.y /= segment.cont

            plt.plot(segment.x, segment.y)
            yrange = plt.gca().get_ylim()
            plt.plot((c, c), yrange, 'r--', lw=2)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Relative flux")
            plt.draw()
            valid = raw_input("Is this line usable for vsini determination (y/n)? ")
            if "n" in valid.lower():
                plt.cla()
                continue

            # Ask if the continuum needs to be renormalized
            done = False
            while not done:
                renorm = raw_input("Renormalize continuum (y/n)? ")
                if "y" in renorm.lower():
                    plt.cla()
                    segment = self.UserContinuum(segment)
                    plt.plot(segment.x, segment.y)
                    plt.plot(segment.x, segment.cont)
                    plt.draw()
                else:
                    segment.y /= segment.cont
                    done = True

            # Fourier transform the line, and let the user click on the first minimum
            plt.cla()
            vsini = self.UserVsini(segment)
            vsini_values.append(vsini)

            plt.cla()

        # Save the mean and standard deviation in the file 'vsini.dat'
        outfile = open("vsini.dat", "a")
        outfile.write("%s%.2f\t%.3f\n" % (self.fname.ljust(20), np.mean(vsini_values), np.std(vsini_values)))
        self.vsini = np.mean(vsini_values)
        return


    def CorrectVelocity(self, rvstar=0.0, bary=True, resample=True):
        """
          This function corrects for the radial velocity of the star.
          - rvstar: the radial velocity of the star, in heliocentric velocity km/s
          - bary: a bool variable to decide whether the barycentric velocity
                  should be corrected. If true, it uses the header from the
                  data most recently read in.
          - resample: a bool variable to decide whether to resample
                      the data into a constant wavelength spacing
                      after doing the correction
        """
        # First, check to make sure the user entered a datafile
        if self.data is None:
            fname = raw_input("Enter filename for the data: ")
            self.InputData(fname)

        rv = rvstar
        if bary:
            header = self.headers[0]
            jd = header['jd']
            observatory = header['observat']
            if "MCDONALD" in observatory:
                latitude = 30.6714
                longitude = 104.0225
                altitude = 2070.0
            elif "CTIO" in observatory:
                latitude = -30.1697
                longitude = 70.8065
                altitude = 2200.0
            ra = header['ra']
            dec = header['dec']
            ra_seg = ra.split(":")
            dec_seg = dec.split(":")
            ra = float(ra_seg[0]) + float(ra_seg[1]) / 60.0 + float(ra_seg[2]) / 3600.0
            dec = float(dec_seg[0]) + float(dec_seg[1]) / 60.0 + float(dec_seg[2]) / 3600.0
            rv += astrolib.helcorr(longitude, latitude, altitude, ra, dec, jd, debug=self.debug)[0]
        c = constants.c.cgs.value * units.cm.to(units.km)
        for i, order in enumerate(self.data):
            order.x *= (1.0 + rv / c)
            if resample:
                self.data[i] = self._resample(order)
            else:
                self.data[i] = order
        return


    def GridSearch(self, windguess=-14.3, betaguess=0.9):
        """
          This method will do the actual search through the grid, tallying the chi-squared
        value for each set of parameters. The guess parameters are determined from the
        spectral type given in the __init__ call to this class.

          It does the grid search in a few steps. First, it determines the best Teff
        and logg for the given wind and metallicity guesses (which default to solar
        metallicity and no wind). Then, it searches the subgrid near the best Teff/logg
        to nail down the best metallicity, silicon value, and macroturbelent velocity

         - windguess is the guess value for the wind. If not given, it defaults to no wind
         - betaguess is the guess value for the wind velocity parameter 'beta'. Ignored
                     if windguess is None; otherwise it MUST be given!


          It will return the best-fit parameters, as well as the list of
        parameters tested and their associated chi-squared values
        """

        # First, check to make sure the user entered a datafile
        if self.data is None:
            fname = raw_input("Enter filename for the data: ")
            self.InputData(fname)

        #Now, find the spectral lines that are visible in this data.
        self._ConnectLineToOrder()

        # Find the best log(g) for the guess temperature
        bestlogg, dlogg, seplogg = self._FindBestLogg(self.Teff_guess, self.logg_guess, windguess, betaguess, 0.1,
                                                      -4.49, 10.0)

        """
          Start loop for determination of Teff, Si-abundance, and microturbulence
        """



        # Find the best Teff and log(g)
        if windguess is None:
            Teff, logg, parlist = self._FindBestTemperature(self.Teff_guess, self.logg_guess, -14.3, 0.9, 0.1, -4.49,
                                                            10.0)
        else:
            Teff, logg, parlist = self._FindBestTemperature(self.Teff_guess, self.logg_guess, windguess, betaguess, 0.1,
                                                            -4.49, 10.0)
        print Teff, logg
        print parlist
        self.parlist = parlist  #TEMPORARY! REMOVE WHEN I AM DONE WITH THIS FUNCTION!

        #For Teff and logg close to the best ones, find the best other parameters (search them all?)
        tidx = np.argmin(abs(np.array(self.Teffs) - Teff))
        for i in range(max(0, tidx - 1), min(len(self.Teffs), tidx + 2)):
            T = self.Teffs[i]
            loggs = np.array([round(g, 2) for g in np.arange(4.5, 4 * np.log10(T) - 15.02, -0.1)])
            gidx = np.argmin(abs(loggs - logg))
            for j in range(max(0, gidx - 1), min(len(loggs), gidx + 2)):
                pars = self._FitParameters(T, loggs[j], parlist)
        self.parlist = parlist  #TEMPORARY! REMOVE WHEN I AM DONE WITH THIS FUNCTION!


    def _ConnectLineToOrder(self, force=False):
        """
          This private method is to determine which lines exist in the data,
        and in what spectral order they are. It is called right before starting
        the parameter search, in order to minimize the number of models we need
        to read in.
          If force==True, then we will do this whether or not it was already done
        """
        # Don't need to do this if we already have
        if len(self.visible_species.keys()) > 0 and not force:
            print "Already connected lines to spectral orders. Not repeating..."
            return

        species = {}

        Teff = self.Teffs[4]
        logg = [round(g, 2) for g in np.arange(4.5, 4 * np.log10(Teff) - 15.02, -0.1)][0]
        for spec in self.species:
            print "\nGetting model for %s" % spec
            model = self.GetModel(Teff,
                                  logg,
                                  -14.3,
                                  0.9,
                                  0.1,
                                  -4.49,
                                  spec,
                                  10)
            # Find the appropriate order
            w0 = (model.x[0] + model.x[-1]) / 2.0
            idx = -1
            diff = 9e9
            for i, order in enumerate(self.data):
                x0 = (order.x[0] + order.x[-1]) / 2.0
                if abs(x0 - w0) < diff and order.x[0] < w0 < order.x[-1]:
                    diff = abs(x0 - w0)
                    idx = i
            if idx < 0 or (idx == i and diff > 10.0):
                continue
            species[spec] = idx

        self.visible_species = species
        return


    def _FindBestLogg(self, Teff, logg_values, wind, beta, He, Si, vmacro, vmicro):
        """
          This semi-private method finds the best log(g) value for specific values of
        the other parameters. It does so by fitting the Balmer line wings
        """

        xlims = {"HGAMMA": [430.0, 434.047, 438.0],
                 "HDELTA": [406.0, 410.174, 414.0],
                 "HBETA": [480.0, 486.133, 492.0]}

        species = ["HGAMMA", "HDELTA"]
        if wind < -13.8:
            species.append("HBETA")

        # Begin chi-squared loop
        chisquared = [[] for i in range(len(species))]
        loggs_tested = [[] for i in range(len(species))]
        for i, spec in enumerate(species):
            if spec not in self.visible_species.keys():
                continue
            order = self.data[self.visible_species[spec]]
            xlow, lambda0, xhigh = xlims[spec]

            #We don't want to include the inner region in the fit.
            delta = self._GetDelta(order, lambda0, Teff, self.vsini, vmacro, vmicro)
            goodindices = np.where(np.logical_or(order.x < lambda0 - delta,
                                                 order.x > lambda0 + delta))[0]
            waveobs = order.x[goodindices]
            fluxobs = (order.y / order.cont)[goodindices]
            errorobs = (order.err / order.cont)[goodindices]

            # Further reduce the region to search so that it is between xlow and xhigh
            goodindices = np.where(np.logical_and(waveobs > xlow,
                                                  waveobs < xhigh))[0]
            waveobs = waveobs[goodindices]
            fluxobs = fluxobs[goodindices]
            errorobs = errorbs[goodindices]


            # Loop over logg values
            lineprofiles = []
            for logg in logg_values:
                model = self.GetModel(Teff,
                                      logg,
                                      wind,
                                      beta,
                                      He,
                                      Si,
                                      spec,
                                      vmicro,
                                      xspacing=order.x[1] - order.x[0])
                model = Broaden.RotBroad(model, vsini * units.km.to(units.cm))
                model = Broaden.MacroBroad(model, vmacro)
                model = Broaden.ReduceResolution(model, self.resolution)
                model = FittingUtilities.RebinData(model, waveobs)
                lineprofiles.append(model)


                # Get the chi-squared for this data
                chi2 = self._GetChiSquared(waveobs, fluxobs, errorobs, model)
                chisquared[i].append(chi2)
                loggs_tested[i].append(logg)


            # Find the best chi-squared, summed over the lines considered, and the best individual one
            chisquared = np.array(chisquared)
            bestlogg = logg_values[np.argmin(chisquared[i])]
            separate_best = np.argmin(chisquared[i])
            separate_bestlogg[i] = logg_values[separate_best]

            # Find where there are large deviations (other lines)
            modelflux = lineprofiles[separate_best]
            sigma = np.std(modelflux - fluxobs)
            good = np.where(abs(modelflux - fluxobs) < 3.0 * sigma)
            waveobs = waveobs[good]
            fluxobs = fluxobs[good]
            errorobs = errorobs[good]

            j = 0
            for logg, profile in zip(logg_values, lineprofiles):
                chi2 = self._GetChiSquared(waveobs, fluxobs, errorobs, profile)
                chisquared[i][j] = chi2
                j += 1

            bestlogg = logg_values[np.argmin(chisquared[i])]
            separate_best = np.argmin(chisquared[i])
            separate_bestlogg[i] = logg_values[separate_best]

        total = np.sum(chisquared, index=0)
        separate_bestlogg = np.array(separate_bestlogg)

        # Find the best logg over all lines considered
        best = np.argmin(total)
        bestgrav = logg_values[best]
        loggmin = min(separate_bestlogg)
        loggmax = max(separate_bestlogg)

        # Determine the error the the logg-determination as the
        # maximal deviation between the separately determined
        # loggs and the general best matching one
        deltalogg_minus = np.sqrt((bestgrav - loggmin) ** 2 + sigma ** 2)
        deltalogg_plus = np.sqrt((bestgrav - loggmax) ** 2 + sigma ** 2)
        deltalogg = max(0.5, deltalogg_minus, deltalogg_plus)

        return [bestgrav, deltalogg, separate_bestlogg]


    def _GetChiSquared(self, waveobs, fluxobs, errorobs, model):
        """
          This private method determines the log-likelihood of the data
        given the model.
        """
        # Make sure the observation and model overlap
        goodindices = np.where(np.logical_and(waveobs > model.x[0], waveobs < model.x[-1]))[0]
        wavecompare = waveobs[goodindices]
        fluxcompare = fluxobs[goodindices]
        errorcompare = errorobs[goodindices]

        # Interpolate model onto the same wavelength grid as the data
        model = FittingUtilities.RebinData(model, wavecompare)

        # Let the x-axis shift by +/- 5 pixels
        chisq = []
        for shift in range(-5, 6):
            flux = self._shift(fluxcompare, shift)
            error = self._shift(errorcompare, shift)
            chisq.append(((flux - model.y) / error) ** 2)
        return min(chisq)


    def _shift(self, array, amount):
        """
          Shifts array by amount indices. Uses collections.deque objects to do so efficiently
        """
        array = deque(array)
        return list(array.rotate(amount))


    def _GetDelta(self, order, lambda0, Teff, vsini, vmacro, vmicro, minmax="min"):
        """
          This private method finds the inner region of a line to ignore
          in the logg fit to the line wings
        """
        # FHWM
        idx = np.argmin(abs(order.x - lambda0))
        flux0 = order.y[idx] / order.cont[idx]
        fluxhalf = 0.5(1.0 + flux0)

        idx = max(np.where(np.logical_and(order.x / order.y > fluxhalf, order.x < lambda0))[0])
        waveblue = order.x[idx]

        idx = min(np.where(np.logical_and(order.x / order.y > fluxhalf, order.x > lambda0))[0])
        wavered = order.x[idx]

        delta1 = min(lambda0 - waveblue, wavered - lambda0)

        # vsini and macroturbulent velocity
        c = constants.c.cgs.value * units.cm.to(units.km)
        delta2 = (vsini + vmacro) / (2.0 * c) * lambda0


        # thermal velocity
        mp = constants.m_p.cgs.value
        kB = constants.k_B.cgs.value
        vth_square = 2 * kB * Teff / mp
        vmic = vmicro * 10 ** 5
        vtherm = np.sqrt(vth_square + vmic ** 2)
        delta3 = 3 * vtherm * 10 ** -5 / c * lambda0

        if minmax.lower == "min":
            return min(delta1, delta2, delta3)
        elif minmax.lower == "max":
            return max(delta1, delta2, delta3)


    def _FindBestTemperature(self, Teff_guess, logg_guess, wind, beta, He, Si, vmicro, vmacro):
        """
          This semi-private method determines the best temperature and log(g) values,
        given specific values for the wind, metallicity, and macroturbulent velocity
        parameters.
        """

        # Keep a list of the parameters and associated chi-squared values
        pars = []

        # Set the range in temperature and log(g) to search
        dT = 2000
        dlogg = 1.5

        # Determine range of temperatures to search
        Teff_low = HelperFunctions.GetSurrounding(self.Teffs, Teff_guess - dT)[0]
        Teff_high = HelperFunctions.GetSurrounding(self.Teffs, Teff_guess + dT)[0]
        first = self.Teffs.index(Teff_low)
        last = self.Teffs.index(Teff_high)
        if last < len(self.Teffs) - 1:
            last += 1

        # Define the species to search
        xlims = {"HGAMMA": [430.0, 434.047, 438.0],
                 "HDELTA": [406.0, 410.174, 414.0],
                 "HBETA": [480.0, 486.133, 492.0]}

        species = ["HGAMMA", "HDELTA"]
        if wind < -13.8:
            species.append("HBETA")

        # Begin loop over temperatures
        chisquared = {}
        loggvals = []
        loggerrs = []
        for Teff in self.Teffs[first:last]:
            if self.debug:
                print "T = %g" % Teff
            loggs = [round(g, 2) for g in np.arange(4.5, 4 * np.log10(Teff) - 15.02, -0.1)][::-1]
            logg_low = HelperFunctions.GetSurrounding(loggs, self.logg_guess - dlogg)[0]
            logg_high = HelperFunctions.GetSurrounding(loggs, self.logg_guess + dlogg)[0]
            first2 = loggs.index(logg_low)
            last2 = loggs.index(logg_high)
            if last2 < len(loggs) - 1:
                last2 += 1

            # Do the search over log(g) for this temperature
            bestgrav, deltalogg, separate_bestlogg = self._FindBestLogg(Teff,
                                                                        logs[first2:last2],
                                                                        wind,
                                                                        beta,
                                                                        He,
                                                                        Si,
                                                                        vmacro,
                                                                        vmicro)
            #pars_temp = self._FindBestLogg(Teff, loggs[first2:last2], wind, beta, He, Si, vmacro)

            loggvals.append(bestgrav)
            loggerrs.append(deltalogg)

            for spec in species:
                if spec not in self.visible_species.keys():
                    continue
                order = self.data[self.visible_species[spec]]
                xlow, lambda0, xhigh = xlims[spec]

                #We want to include ONLY the inner region in the fit.
                delta = self._GetDelta(order, lambda0, Teff, self.vsini, vmacro, vmicro, minmax="max")
                goodindices = np.where(np.logical_or(order.x >= lambda0 - delta,
                                                     order.x <= lambda0 + delta))[0]
                waveobs = order.x[goodindices]
                fluxobs = (order.y / order.cont)[goodindices]
                errorobs = (order.err / order.cont)[goodindices]

                # Further reduce the region to search so that it is between xlow and xhigh
                goodindices = np.where(np.logical_and(waveobs > xlow,
                                                      waveobs < xhigh))[0]
                waveobs = waveobs[goodindices]
                fluxobs = fluxobs[goodindices]
                errorobs = errorbs[goodindices]

                # Generate the model
                model = self.GetModel(Teff,
                                      bestlogg,
                                      wind,
                                      beta,
                                      He,
                                      Si,
                                      spec,
                                      vmicro,
                                      xspacing=order.x[1] - order.x[0])
                model = Broaden.RotBroad(model, vsini * units.km.to(units.cm))
                model = Broaden.MacroBroad(model, vmacro)
                model = Broaden.ReduceResolution(model, self.resolution)
                model = FittingUtilities.RebinData(model, waveobs)


                # Get the chi-squared for this data
                chi2 = self._GetChiSquared(waveobs, fluxobs, errorobs, model)
                chisquared[spec].append(chi2)



        # Now, find the best chi-squared value
        # First, the single best one:
        chi2arr = []
        for spec in species:
            if spec not in self.visible_species.keys():
                continue
            chi2arr.append(chisquared[spec])
        idx = np.argmin(chi2arr) % int(last - first)
        bestindividual = self.Teffs[first + idx]

        # Now, the best one summed over the lines
        idx = np.argmin(np.sum(chi2arr, axis=0))
        bestT = self.Teffs[first + idx]

        return bestT, loggvals[idx], loggerrs[idx]


    def _FitParameters(self, Teff, logg, parlist):
        """
          This method takes a specific value of Teff and logg, and
        searches through the wind parameters, the metallicities, and
        the macroturbulent velocities.
          -Teff: the effective temperature to search within
          -logg: the log(g) to search within
          -parlist: the list of parameters already searched. It will not
                    duplicate already searched parameters
        """
        if Teff < 20000:
            vmacros = (3, 6, 10, 12, 15)
        else:
            vmacros = (6, 10, 12, 15, 20)
        for He in self.Heliums:
            if self.debug:
                print "Helium fraction = %g" % He
            for Si in self.Silicons:
                if self.debug:
                    print "Log(Silicon abundance) = %g" % Si
                for Q in self.logQ[:4]:
                    if self.debug:
                        print "Wind speed parameter = %g" % Q
                    print "test"
                    for beta in self.betas[:4]:
                        if self.debug:
                            print "Wind velocity scale parameter (beta) = %g" % beta
                        for vmacro in vmacros:
                            if self.debug:
                                print "Macroturbulent velocity = %g" % vmacro
                            # Check if this is already in the parameter list
                            done = False
                            for p in parlist:
                                if p.Teff == Teff and p.logg == logg and p.He == He and p.Si == Si and p.Q == Q and p.beta == beta and p.vmacro == vmacro:
                                    done = True
                            if done:
                                continue

                            chisq = 0.0
                            normalization = 0.0
                            for spec in self.visible_species.keys():
                                print "\t\t", spec
                                order = self.data[self.visible_species[spec]]
                                model = self.GetModel(Teff,
                                                      logg,
                                                      -14.3,
                                                      0.9,
                                                      0.1,
                                                      -4.49,
                                                      spec,
                                                      10,
                                                      xspacing=order.x[1] - order.x[0])
                                model = Broaden.RotBroad(model, self.vsini * units.km.to(units.cm))
                                model = Broaden.ReduceResolution(model, 60000.0)
                                model = FittingUtilities.RebinData(model, order.x)
                                chisq += np.sum((order.y - model.y * order.cont) ** 2 / order.err ** 2)
                                normalization += float(order.size())
                            p = ParameterValues(Teff, logg, Q, beta, He, Si, vmacro, chisq / (normalization - 7.0))
                            parlist.append(p)

        return parlist


    def GetRadialVelocity(self):
        """
          DO NOT USE THIS! IT DOESN'T WORK VERY WELL, AND THE 'CorrectVelocity'
          METHOD SHOULD WORK WELL ENOUGH FOR WHAT I NEED!

          This function will get the radial velocity by cross-correlating a model
        of the star against all orders of the data. The maximum of the CCF will
        likely be broad due to rotational broadening, but will still encode the
        rv of the star (plus Earth, if the observations are not barycentric-corrected)
        """
        # Get all of the models with the appropriate temperature and log(g)
        # We will assume solar abundances of everything, and no wind for this

        xgrid = np.arange(self.data[0].x[0] - 20, self.data[-1].x[-1] + 20, 0.01)
        full_model = DataStructures.xypoint(x=xgrid, y=np.ones(xgrid.size))
        Teff = HelperFunctions.GetSurrounding(self.Teffs, self.Teff_guess)[0]
        loggs = [round(g, 2) for g in np.arange(4.5, 4 * np.log10(Teff) - 15.02, -0.1)]
        logg = HelperFunctions.GetSurrounding(loggs, self.logg_guess)[0]
        corrlist = []
        normalization = 0.0
        species = ['BRALPHA', 'BRBETA', 'BRGAMMA',
                   'HALPHA', 'HBETA', 'HDELTA', 'HGAMMA']
        for spec in species:
            print "\nGetting model for %s" % spec
            model = self.GetModel(Teff,
                                  logg,
                                  -14.3,
                                  0.9,
                                  0.1,
                                  -4.49,
                                  spec,
                                  10)
            # Find the appropriate order
            w0 = (model.x[0] + model.x[-1]) / 2.0
            idx = -1
            diff = 9e9
            for i, order in enumerate(self.data):
                x0 = (order.x[0] + order.x[-1]) / 2.0
                if abs(x0 - w0) < diff and order.x[0] < w0 < order.x[-1]:
                    diff = abs(x0 - w0)
                    idx = i
            if idx < 0 or (idx == i and diff > 10.0):
                continue
            order = self.data[idx]


            # Make sure the model is bigger than this order
            if model.x[0] > order.x[0] - 5.0:
                model.x = np.r_[(order.x[0] - 5.0,), model.x]
                model.y = np.r_[(1.0,), model.y]
            if model.x[-1] < order.x[-1] + 5.0:
                model.x = np.r_[model.x, (order.x[-1] + 5.0,)]
                model.y = np.r_[model.y, (1.0,)]
            model.cont = np.ones(model.x.size)

            # Rotationally broaden model
            xgrid = np.arange(model.x[0], model.x[-1], 0.001)
            model = FittingUtilities.RebinData(model, xgrid)
            model = Broaden.RotBroad(model, self.vsini * units.km.to(units.cm))

            # Find low point:
            idx = np.argmin(model.y)
            w0 = model.x[idx]
            idx = np.argmin(order.y / order.cont)
            x0 = order.x[idx]
            print "Model wavelength = %.5f" % w0
            print "Data wavelength = %.5f" % x0
            print "Velocity shift = %g km/s" % (3e5 * (x0 - w0) / w0)


            # Rebin data to constant (log) spacing
            start = np.log(order.x[0])
            end = np.log(order.x[-1])
            neworder = order.copy()
            neworder.x = np.logspace(start, end, order.size(), base=np.e)
            neworder = FittingUtilities.RebinData(order, neworder.x)

            # Rebin the model to the same spacing
            logspacing = np.log(neworder.x[1] / neworder.x[0])
            left = np.searchsorted(model.x, order.x[0] - 10)
            right = np.searchsorted(model.x, order.x[-1] + 10)
            right = min(right, model.size() - 2)
            left, right = 0, -1
            start = np.log(model.x[left])
            end = np.log(model.x[right])
            xgrid = np.exp(np.arange(start, end + logspacing * 1.1, logspacing))

            segment = FittingUtilities.RebinData(model, xgrid)
            plt.figure(3)
            plt.plot(neworder.x, neworder.y / neworder.cont)
            plt.plot(segment.x, segment.y / segment.cont)

            corr = Correlate.Correlate([neworder, ], [segment, ], debug=True)
            plt.figure(2)
            plt.plot(corr.x, corr.y)

            if not np.any(np.isnan(corr.y)):
                corrlist.append(corr)
                normalization += float(order.size())


                #fcn = interp1d(model.x, model.y, kind='linear', bounds_error=False, fill_value=1.0)

                #full_model.y *= fcn(full_model.x)
                #plt.plot(model.x, model.y)

        #plt.plot(full_model.x, full_model.y)
        #plt.show()

        #output = Correlate.GetCCF(self.data, full_model, vsini=0.0, resolution=60000, process_model=True, rebin_data=True, debug=True)
        #ccf = output["CCF"]
        #plt.plot(ccf.x, ccf.y)
        #idx = np.argmax(ccf.y)
        #print "Maximum CCF at %g km/s" %(ccf.x[idx])
        #plt.show()

        # Add up the individual CCFs (use the Maximum Likelihood method from Zucker 2003, MNRAS, 342, 1291)
        total = corrlist[0].copy()
        total.y = np.ones(total.size())
        for i, corr in enumerate(corrlist):
            correlation = spline(corr.x, corr.y, k=1)
            N = self.data[i].size()
            total.y *= np.power(1.0 - correlation(total.x) ** 2, float(N) / normalization)
        master_corr = total.copy()
        master_corr.y = 1.0 - np.power(total.y, 1.0 / float(len(corrlist)))

        idx = np.argmax(master_corr.y)
        rv = master_corr.x[idx]
        print "Radial velocity = %g km/s" % rv

        plt.figure(1)
        plt.plot(master_corr.x, master_corr.y, 'k-')
        plt.xlabel("Velocity (km/s)")
        plt.ylabel("CCF")
        plt.show()

        return rv


    def UserContinuum(self, spectrum):
        """
          This will let the user click twice to define continuum points, and
        will then fit a straight line through the points as the continuum vector.
        Expects a short segment of spectrum, such that the continuum is quite linear.
        """
        self.interactive_mode = "continuum"
        fig = plt.figure(1)
        cid = fig.canvas.mpl_connect('button_press_event', self.mouseclick)
        self.clicks = []
        plt.plot(spectrum.x, spectrum.y)
        plt.draw()
        plt.waitforbuttonpress()
        plt.waitforbuttonpress()
        fig.canvas.mpl_disconnect(cid)
        plt.cla()
        # Once we get here, the user has clicked twice
        for click in self.clicks:
            print click.xdata, "\t", click.ydata
        slope = (self.clicks[1].ydata - self.clicks[0].ydata) / (self.clicks[1].xdata - self.clicks[0].xdata)
        spectrum.cont = self.clicks[0].ydata + slope * (spectrum.x - self.clicks[0].xdata)
        return spectrum


    def UserVsini(self, spectrum):
        """
          This does a Fourier transform on the spectrum, and then lets
        the user click on the first minimum, which indicates the vsini of the star.
        """
        # Set up plotting
        self.interactive_mode = "vsini"
        fig = plt.figure(1)
        cid = fig.canvas.mpl_connect('button_press_event', self.mouseclick)

        # Make wavelength spacing uniform
        xgrid = np.linspace(spectrum.x[0], spectrum.x[-1], spectrum.size())
        spectrum = FittingUtilities.RebinData(spectrum, xgrid)
        extend = np.array(40 * spectrum.size() * [1, ])
        spectrum.y = np.r_[extend, spectrum.y, extend]

        # Do the fourier transorm and keep the positive frequencies
        fft = np.fft.fft(spectrum.y - 1.0)
        freq = np.fft.fftfreq(spectrum.y.size, d=spectrum.x[1] - spectrum.x[0])
        good = np.where(freq > 0)[0]
        fft = fft[good].real ** 2 + fft[good].imag ** 2
        freq = freq[good]

        # Plot inside a do loop, to let user try a few times
        done = False
        trials = []
        plt.loglog(freq, fft)
        plt.xlim((1e-2, 10))
        plt.draw()
        for i in range(10):
            plt.waitforbuttonpress()
            sigma_1 = self.click.xdata
            if self.click.button == 1:
                c = constants.c.cgs.value * units.cm.to(units.km)
                vsini = 0.66 * c / (spectrum.x.mean() * sigma_1)
                print "vsini = ", vsini, " km/s"
                trials.append(vsini)
                plt.cla()
            else:
                done = True
                break

        fig.canvas.mpl_disconnect(cid)
        if len(trials) == 1:
            return trials[0]

        print "\n"
        for i, vsini in enumerate(trials):
            print "\t[%i]: vsini = %.1f km/s" % (i + 1, vsini)
        inp = raw_input("\nWhich vsini do you want to use (choose from the options above)? ")
        return trials[int(inp) - 1]


    def mouseclick(self, event):
        """
          This is a generic mouseclick method. It will act differently
        based on what the value of self.interactive_mode is.
        """
        if self.interactive_mode == "continuum":
            if len(self.clicks) < 2:
                plt.plot(event.xdata, event.ydata, 'rx', markersize=12)
                plt.draw()
                self.clicks.append(event)


        elif self.interactive_mode == "vsini":
            self.click = event
        return


    def _resample(self, order):
        """
          Semi-private method to resample an order to a constant wavelength spacing
        """
        xgrid = np.linspace(order.x[0], order.x[-1], order.size())
        return FittingUtilities.RebinData(order, xgrid)
