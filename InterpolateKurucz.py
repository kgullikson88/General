import numpy
from scipy.interpolate import UnivariateSpline
import os
from collections import defaultdict
import DataStructures
import pyfits
import pylab

homedir = os.environ["HOME"] + "/"
model_directory = homedir + "Dropbox/School/Research/AstarStuff/TargetLists/AgeDetermination/Kurucz_Models"

class Models:
  def __init__(self, modeldir=model_directory):
    self.model_dict = defaultdict(str)   #Dictionary of all models in modeldir, with temperature as key
    self.read_dict = defaultdict(list)   #Dictionary of all models already read in.
                                         #The list is the wavelength, followed by a dictionary
                                         #of flux values for each value of log(g)
    self.logg_grid = numpy.arange(0.0, 5.5, 0.5)  #grid of log(g) values for the default Kurucz grid.
                                                  #Warning! May not be true if a different grid is used!
    allfiles = os.listdir(modeldir)
    for fname in allfiles:
      if fname.startswith("ckp00") and fname.endswith(".fits"):
        temperature = float(fname.split("_")[-1].split(".fits")[0])
        self.model_dict[temperature] = fname

  #Function to read in a model with temperature T
  def ReadFile(self, T, logg):
    #Make sure it is not already read in
    if T in self.read_dict.keys():
      if logg in self.read_dict[T][1].keys():
        print "Model with T = %g and log(g) = %g already read. Skipping..." %(T, logg)
        return 0
      else:
        hdulist = pyfits.open(self.model_dict[T])
        data = hdulist[1].data
        hdulist.close()
        logg_str = "g%i" %(logg*10.)
        try:
          flux = data[logg_str]
          self.read_dict[T][1][logg] = flux
        except KeyError:
          print "Error! log(g) = %g does not exist in file %s!"
          return -1
    else:
      #This means this temperature has not yet been read in
      fname = self.model_dict[T]
      if fname == "":
        #This temperature does not exist in the model
        print "Error! No model found with T = %g" %T
      else:
        hdulist = pyfits.open(fname)
        data = hdulist[1].data
        hdulist.close()
        wave = data['wavelength']
        d = defaultdict(numpy.ndarray)
        logg_str = "g%i" %(logg*10.)
        try:
          flux = data[logg_str]
          d[logg] = flux
          self.read_dict[T] = [wave, d]
        except KeyError:
          print "Error! log(g) = %g does not exist in file %s!"
          return -1

    #If we get here, everything was successful.
    return 0


  #Function to retrieve data, returning as an xypoint
  def GetSpectrum(self, T, logg):
    retval = self.ReadFile(T, logg)
    if retval == 0:
      wave = self.read_dict[T][0]
      flux = self.read_dict[T][1][logg]
      return DataStructures.xypoint(x=wave, y=flux)
    else:
      return -1
  

  #Function to linearly interpolate to a given Temperature and log(g)
  def GetInterpolatedSpectrum(self, T, logg):
    #First, find the closest two temperatures in self.model_dict
    Tclosest = 9e9
    Tsecond = 0
    Temperatures = sorted(self.model_dict.keys())
    for temp in Temperatures:
      if numpy.abs(T-temp) < numpy.abs(Tclosest-temp):
        Tclosest = temp
    i = 0
    while Tsecond < Tclosest:
      Tsecond = Temperatures[i]
      i += 1
    if Tclosest > T and i > 1:
      Tsecond = Temperatures[i-2]
      #for temp in Temperatures:
      #if numpy.abs(T-temp) < numpy.abs(Tsecond-temp) and temp != Tclosest:
      #  Tsecond = temp

    #Do the same thing for log(g)
    gclosest = 9e9
    gsecond = 9e10
    for g in self.logg_grid:
      if numpy.abs(logg-g) < numpy.abs(gclosest-g):
        gclosest=g
    for g in self.logg_grid:
      if numpy.abs(logg-g) < numpy.abs(gsecond-g) and g != gclosest:
        gsecond=g


    #For each temperature, we will interpolate to the requested log(g) first
    #Do the closest temperature first
    spec1 = self.GetSpectrum(Tclosest, gclosest)
    spec2 = self.GetSpectrum(Tclosest, gsecond)
    if type(spec1) != int and type(spec2) != int:
      #This means everything went fine with GetSpectrum
      spectrum_T1 = spec1.copy()
      spectrum_T1.y = (spec2.y - spec1.y)/(gsecond - gclosest)*(logg - gclosest) + spec1.y
    else:
      return -1

      
    #And the second closest temperature
    spec1 = self.GetSpectrum(Tsecond, gclosest)
    spec2 = self.GetSpectrum(Tsecond, gsecond)
    if type(spec1) != int and type(spec2) != int:
      #This means everything went fine with GetSpectrum
      spectrum_T2 = spec1.copy()
      spectrum_T2.y = (spec2.y - spec1.y)/(gsecond - gclosest)*(logg - gclosest) + spec1.y
    else:
      return -1

    spectrum = spectrum_T1.copy()
    if numpy.all(spectrum_T1.x == spectrum_T2.x):
      spectrum.y = (spectrum_T1.y - spectrum_T2.y)/(Tclosest - Tsecond) * (T - Tclosest) + spectrum_T1.y
    else:
      return -1


    return spectrum
