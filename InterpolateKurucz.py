import numpy as np
from scipy.interpolate import UnivariateSpline
import os
from collections import defaultdict
from astropy.io import fits as pyfits
import pylab
import DataStructures
import Units

homedir = os.environ["HOME"] + "/"
model_directory = homedir + "Dropbox/School/Research/AstarStuff/TargetLists/AgeDetermination/Kurucz_Models"

class Models:
  def __init__(self, modeldir=model_directory, debug=False):
    self.model_dict = defaultdict(lambda : defaultdict(str))   #Dictionary of all models in modeldir,
                                                               #with temperature and metallicity as key
    self.read_dict = defaultdict(lambda : defaultdict(str))    #Dictionary of all models already read in.
                                                               #The list is the wavelength, followed by
                                                               #a dictionary of flux values for each
                                                               #value of log(g)
    self.logg_grid = np.arange(0.0, 5.5, 0.5)  #grid of log(g) values for the default Kurucz grid.
                                                  #Warning! May not be true if a different grid is used!
    self.modeldir = modeldir
    self.debug = debug
    allfiles = os.listdir(modeldir)
    for fname in allfiles:
      if fname.startswith("ck") and fname.endswith(".fits"):
        temperature = float(fname.split("_")[-1].split(".fits")[0])
        metallicity_string = fname[2:5]
        metallicity = float(metallicity_string[1:])/10.0
        if "m" in metallicity_string:
          metallicity *= -1
        self.model_dict[temperature][metallicity] = fname

  #Function to read in a model with temperature T
  def ReadFile(self, T, logg, metal):
    #Make sure it is not already read in
    if T in self.read_dict.keys() and metal in self.read_dict[T].keys():
      if logg in self.read_dict[T][metal][1].keys():
        if self.debug:
          print "Model with T = %g and log(g) = %g already read. Skipping..." %(T, logg)
        return 0
      else:
        hdulist = pyfits.open(self.modeldir + "/" + self.model_dict[T][metal])
        data = hdulist[1].data
        hdulist.close()
        logg_str = "g%.2i" %(logg*10.)
        try:
          flux = data[logg_str]
          self.read_dict[T][metal][1][logg] = flux
        except KeyError:
          print "Error! log(g) = %g does not exist in file %s!" %(logg, self.modeldir + "/" + self.model_dict[T][metal])
          return -1
    else:
      #This means this temperature and metallicity has not yet been read in
      fname = self.modeldir + "/" + self.model_dict[T][metal]
      if self.model_dict[T][metal] == "":
        #This temperature does not exist in the model
        print T, metal
        print "Error! No model found with T = %g and [Z/H] = " %(T, metal)
        return -1
      else:
        hdulist = pyfits.open(fname)
        data = hdulist[1].data
        hdulist.close()
        wave = data['wavelength']
        d = defaultdict(np.ndarray)
        logg_str = "g%.2i" %(logg*10.)
        try:
          flux = data[logg_str]
          d[logg] = flux
          self.read_dict[T][metal] = [wave, d]
        except KeyError:
          print "Error! log(g) = %g does not exist in file %s!" %(logg, fname)
          return -1

    #If we get here, everything was successful.
    return 0


  #Function to retrieve data, returning as an xypoint
  def GetSpectrum(self, T, logg, metal):
    retval = self.ReadFile(T, logg, metal)
    if retval == 0:
      wave = self.read_dict[T][metal][0]
      flux = self.read_dict[T][metal][1][logg]
      return DataStructures.xypoint(x=wave, y=flux)
    else:
      return retval
  

  #Function to linearly interpolate to a given Temperature and log(g)
  def GetInterpolatedSpectrum(self, T, logg, metal=0.0):
    #First, find the closest two temperatures in self.model_dict
    Tclosest = 9e9
    Tsecond = 0
    Temperatures = sorted(self.model_dict.keys())
    for temp in Temperatures:
      if np.abs(T-temp) < np.abs(Tclosest-temp):
        Tclosest = temp
    i = 0
    while Tsecond < Tclosest:
      Tsecond = Temperatures[i]
      i += 1
    if Tclosest > T and i > 1:
      Tsecond = Temperatures[i-2]
    elif Tclosest == Temperatures[-1]:
      Tsecond = Temperatures[-2]
    elif Tclosest == Temperatures[0]:
      Tsecond = Temperatures[1]
      #for temp in Temperatures:
      #if np.abs(T-temp) < np.abs(Tsecond-temp) and temp != Tclosest:
      #  Tsecond = temp

    #Do the same thing for log(g)
    gclosest = 9e9
    gsecond =0
    for g in self.logg_grid:
      if np.abs(logg-g) < np.abs(gclosest-g):
        gclosest=g
    i = 0
    while gsecond < gclosest:
      gsecond = self.logg_grid[i]
      i += 1
    if gclosest > logg and i > 1:
      gsecond = self.logg_grid[i-2]
    elif gclosest == self.logg_grid[-1]:
      gsecond = self.logg_grid[-2]
    elif gclosest == self.logg_grid[0]:
      gsecond = self.logg_grid[1]

    #And again for metallicity
    metalclosest = 9e9
    metalsecond = -9e9
    metals = sorted(self.model_dict[Tclosest].keys())
    if self.debug:
      print "Metals list: "
      print metals
    for z in metals:
      if np.abs(metal - z) < np.abs(metalclosest - z):
        metalclosest = z
    i = 0
    while metalsecond < metalclosest:
      metalsecond = metals[i]
      i += 1
    if metalclosest > metal and i > 1:
      metalsecond = metals[i-2]
    elif metalclosest == metals[-1]:
      metalsecond = metals[-2]
    elif metalclosest == metals[0]:
      metalsecond = metals[1]

    if self.debug:
      print "T = %g\tlog(g) = %g\t[Z/H] = %g" %(T, logg, metal)
      print "[%g, %g]\t[%g, %g]\t[%g, %g]" %(Tclosest, Tsecond, gclosest, gsecond, metalclosest, metalsecond)

    #For each temperature and metallicity, we will interpolate to the requested log(g) first
    #Do the closest temperature and metallicity first
    spec1 = self.GetSpectrum(Tclosest, gclosest, metalclosest)
    spec2 = self.GetSpectrum(Tclosest, gsecond, metalclosest)
    if logg == gclosest:
      spectrum_T1_Z1 = spec1.copy()
    elif type(spec1) != int and type(spec2) != int:
      #This means everything went fine with GetSpectrum
      spectrum_T1_Z1 = spec1.copy()
      spectrum_T1_Z1.y = (spec2.y - spec1.y)/(gsecond - gclosest)*(logg - gclosest) + spec1.y
      if gsecond == gclosest:
        print "log(g) values the same1: %g, %g" %(gsecond, logg)
    else:
      return -1

    #Closest Temperature, second closest metallicity
    spec1 = self.GetSpectrum(Tclosest, gclosest, metalsecond)
    spec2 = self.GetSpectrum(Tclosest, gsecond, metalsecond)
    if logg == gclosest:
      spectrum_T1_Z2 = spec1.copy()
    elif type(spec1) != int and type(spec2) != int:
      #This means everything went fine with GetSpectrum
      spectrum_T1_Z2 = spec1.copy()
      spectrum_T1_Z2.y = (spec2.y - spec1.y)/(gsecond - gclosest)*(logg - gclosest) + spec1.y
      if gsecond == gclosest:
        print "log(g) values the same2: %g" %gsecond
    else:
      return -1
      
    #And the second closest temperature with closest metallicity
    spec1 = self.GetSpectrum(Tsecond, gclosest, metalclosest)
    spec2 = self.GetSpectrum(Tsecond, gsecond, metalclosest)
    if logg == gclosest:
      spectrum_T2_Z1 = spec1.copy()
    elif type(spec1) != int and type(spec2) != int:
      #This means everything went fine with GetSpectrum
      spectrum_T2_Z1 = spec1.copy()
      spectrum_T2_Z1.y = (spec2.y - spec1.y)/(gsecond - gclosest)*(logg - gclosest) + spec1.y
      if gsecond == gclosest:
        print "log(g) values the same3: %g" %gsecond
    else:
      return -1

    #Finally, the second closest temperature with the second closest metallicity
    spec1 = self.GetSpectrum(Tsecond, gclosest, metalsecond)
    spec2 = self.GetSpectrum(Tsecond, gsecond, metalsecond)
    if logg == gclosest:
      spectrum_T2_Z2 = spec1.copy()
    elif type(spec1) != int and type(spec2) != int:
      #This means everything went fine with GetSpectrum
      spectrum_T2_Z2 = spec1.copy()
      spectrum_T2_Z2.y = (spec2.y - spec1.y)/(gsecond - gclosest)*(logg - gclosest) + spec1.y
      if gsecond == gclosest:
        print "log(g) values the same4: %g" %gsecond
    else:
      return -1

    #Now, interpolate to the requested metallicity
    spectrum_T1 = spectrum_T1_Z1.copy()
    if metalclosest != metal and np.all(spectrum_T1_Z1.x == spectrum_T1_Z2.x):
      spectrum_T1.y = (spectrum_T1_Z1.y - spectrum_T1_Z2.y)/(metalclosest - metalsecond) * (metal - metalclosest) + spectrum_T1_Z1.y
      if metalsecond == metalclosest:
        print "[Fe/H] values the same1: %g" %metalsecond
    elif np.any(spectrum_T1_Z1.x != spectrum_T1_Z2.x):
      print "Wavelength grid not the same!"
      return -1

    spectrum_T2 = spectrum_T2_Z1.copy()
    if metalclosest != metal and np.all(spectrum_T2_Z1.x == spectrum_T2_Z2.x):
      spectrum_T2.y = (spectrum_T2_Z1.y - spectrum_T2_Z2.y)/(metalclosest - metalsecond) * (metal - metalclosest) + spectrum_T2_Z1.y
      if metalsecond == metalclosest:
        print "[Fe/H] values the same2: %g" %metalsecond
    elif np.any(spectrum_T2_Z1.x != spectrum_T2_Z2.x):
      print "Wavelength grid not the same!"
      return -1
      

    spectrum = spectrum_T1.copy()
    if T == Tclosest:
      return spectrum
    if np.all(spectrum_T1.x == spectrum_T2.x):
      spectrum.y = (spectrum_T1.y - spectrum_T2.y)/(Tclosest - Tsecond) * (T - Tclosest) + spectrum_T1.y
      if Tsecond == Tclosest:
        print "T values the same1: %g" %Tsecond
    else:
      return -1

    #spec1 = self.GetSpectrum(Tclosest, gclosest, metalclosest)
    #spec2 = self.GetSpectrum(Tsecond, gsecond, metalsecond)
    #pylab.plot(spectrum.x, spectrum.y, label="Interpolated")
    #pylab.plot(spec1.x, spec1.y, label="Closest match")
    #pylab.plot(spec2.x, spec2.y, label="Second closest")
    #pylab.legend(loc='best')
    #pylab.show()
    return spectrum



  #Function to linearly interpolate to a given Temperature and log(g)
  def GetClosestSpectrum(self, T, logg, metal=0.0):
    #First, find the closest temperature in self.model_dict
    Tclosest = 9e9
    Temperatures = sorted(self.model_dict.keys())
    for temp in Temperatures:
      if np.abs(T-temp) < np.abs(Tclosest-temp):
        Tclosest = temp

    #Do the same thing for log(g)
    gclosest = 9e9
    for g in self.logg_grid:
      if np.abs(logg-g) < np.abs(gclosest-g):
        gclosest=g

    #And again for metallicity
    metalclosest = 9e9
    metals = sorted(self.model_dict[Tclosest].keys())
    for z in metals:
      if np.abs(metal - z) < np.abs(metalclosest - z):
        metalclosest = z

    if self.debug:
      print "T = %g\tlog(g) = %g\t[Z/H] = %g" %(Tcloses, gclosest, metalclosest)

    return self.GetSpectrum(Tclosest, gclosest, metalclosest)
