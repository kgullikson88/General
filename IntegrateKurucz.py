import numpy as np
import pylab
from scipy.interpolate import griddata, interp1d
from scipy.integrate import simps
import os
from collections import defaultdict
import InterpolateKurucz
import Units
import DataStructures

homedir = os.environ["HOME"]
filterdir = "%s/Dropbox/School/Research/AstarStuff/TargetLists/AgeDetermination/FilterCurves/" %homedir

defaultfilters = {'U': "%sUband2.dat" %filterdir,
                  'B': "%sBband2.dat" %filterdir,
                  'V': "%sVband2.dat" %filterdir,
                  'R': "%sRband2.dat" %filterdir,
                  'I': "%sIband2.dat" %filterdir,
                  'J': "%sJband2.dat" %filterdir,
                  'H': "%sHband2.dat" %filterdir,
                  'K': "%sKband2.dat" %filterdir,
                  '2mass_J': "%s2mass_Jband.dat" %filterdir,
                  '2mass_H': "%s2mass_Hband.dat" %filterdir,
                  '2mass_K': "%s2mass_Kband.dat" %filterdir}


"""
  Class to integrate kurucz models in various photometric filters,
    for various temperatures, surface gravities, and metallicities.
  Tvals, gvals, and metalvals are all 1-D arrays of the temperature,
    log(g), and metallicities to integrate
  filterfiles is a dictionary with the key being the name of the band
    and the entry being the filename for the throughput in two-column
    format with wavelength (in nm) and throughput as the two columns.
  fluxtype is the type of flux. If Fnu, the flux will be converted to
    erg/s/cm2/Hz before integrating. Otherwise, the flux will be
    integrated as erg/s/cm^2/angstrom
  unitconv can be used to perform unit conversions by a constant factor.
    i.e. converting from erg/s/cm2/Hz to Jansky requires unitconv=1e23
"""
class Integrate:
  def __init__(self, kurucz=None, Tvals=np.arange(5000, 30000, 500),
               gvals=np.arange(3.0, 5.0, 0.1),
               metalvals=np.arange(-1.0, 0.5, 0.1),
               filterfiles=defaultfilters, fluxtype="Fnu", unitconv=1.0):
    if kurucz == None:
      kurucz = InterpolateKurucz.Models()
      
    #Read in filter files and store as DataStructures.xypoints
    self.filters = {}
    for band in filterfiles:
      x, y = np.loadtxt(filterfiles[band], unpack=True)
      self.filters[band] = DataStructures.xypoint(x=x, y=y)

    Temperatures = []
    Gravities = []
    Metallicities = []
    Fluxes = defaultdict(list)

    for T in Tvals:
      print "Integrating for T = %g" %T
      for logg in gvals:
        for metal in metalvals:
          spectrum = kurucz.GetInterpolatedSpectrum(T, logg, metal)
          spectrum.x *= Units.nm/Units.angstrom
          if fluxtype == "Fnu":
            spectrum.y *= Units.angstrom/Units.cm * (spectrum.x*Units.cm/Units.nm)**2 / Units.c * unitconv
          elif fluxtype == "Flambda":
            spectrum.y *= unitconv
          else:
            print "Warning! fluxtype given ( %s ) is not recognized!" %fluxtype
            print "Please use either 'Fnu' or 'Flambda' "

          Temperatures.append(T)
          Gravities.append(logg)
          Metallicities.append(metal)
          for color in self.filters:
            Fluxes[color].append(self.GetFlux(spectrum, self.filters[color]))

    #Done integrating. Now make interpolator functions
    print "Done Integrating filters!"
    self.Interpolators = {}
    for color in Fluxes:
      self.Interpolators[color] = lambda Temp, grav, feonh, fluxes=Fluxes[color]: griddata((Temperatures, Gravities, Metallicities), fluxes, (Temp, grav, feonh), method='linear')

    #Make kurucz a class variable in case the user wants to use it
    self.kurucz_models = kurucz
    
          
          
  #Function to integrate a spectrum across some bandbass
  #Both spectrum and band are expected to be Datastructures.xypoint objects
  def GetFlux(self, spectrum, band):
    bandpass = interp1d(band.x, band.y, bounds_error=False, fill_value=0.0)
    flux = simps(spectrum.y*bandpass(spectrum.x), x=spectrum.x) / simps(bandpass(spectrum.x), x=spectrum.x)
    return flux

          


if __name__ == "__main__":
  integrate = Integrate(Tvals=np.arange(15000, 17000, 200), gvals=np.arange(4.0, 4.3, 0.1))
  interpolators = integrate.Interpolators
  T = 16050.
  logg = 4.14
  metal = 0.13
  print integrate.Interpolators['V'](T, logg, metal)
  kurucz = integrate.kurucz
  spectrum = kurucz.GetInterpolatedSpectrum(T, logg, metal)
  spectrum.x *= Units.nm/Units.angstrom
  spectrum.y *= Units.angstrom/Units.cm * (spectrum.x*Units.cm/Units.nm)**2 / Units.c
  print integrate.GetFlux(spectrum, integrate.filters['V'])
