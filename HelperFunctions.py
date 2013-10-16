"""
  Just a set of helper functions that I use often

"""
import os
import csv
from astropy.io import fits as pyfits
import pySIMBAD as sim
import DataStructures
from collections import defaultdict
import SpectralTypeRelations
import numpy
from scipy.misc import factorial
from scipy.optimize import fminbound, fmin, brent, golden, minimize_scalar

#Ensure a directory exists. Create it if not
def ensure_dir(f):
  d = os.path.dirname(f)
  if not os.path.exists(d):
    os.makedirs(d)
    
    
#Get star data from SIMBAD
def GetStarData(starname):
  link = sim.buildLink(starname)
  star = sim.simbad(link)
  return star


#Check to see if the given star is a binary in the WDS catalog
#if so, return the most recent separation and magnitude of all components
WDS_location = "%s/Dropbox/School/Research/AstarStuff/TargetLists/WDS_MagLimited.csv" %(os.environ["HOME"])
def CheckMultiplicityWDS(starname):
  if type(starname) == str:
    star = GetStarData(starname)
  elif isinstance(starname, sim.simbad):
    star = starname
  else:
    print "Error! Unrecognized variable type in HelperFunctions.CheckMultiplicity!"
    return False
    
  all_names = star.names()

  #Check if one of them is a WDS name
  WDSname = ""
  for name in all_names:
    if "WDS" in name[:4]:
      WDSname = name
  if WDSname == "":
    return False
    
  #Get absolute magnitude of the primary star, so that we can determine 
  #   the temperature of the secondary star from the magnitude difference
  MS = SpectralTypeRelations.MainSequence()
  p_Mag = MS.GetAbsoluteMagnitude(star.SpectralType()[:2], 'V')
  

  #Now, check the WDS catalog for this star
  searchpart = WDSname.split("J")[-1].split("A")[0]
  infile = open(WDS_location, 'rb')
  lines = csv.reader(infile, delimiter=";")
  components = defaultdict(lambda : defaultdict())
  for line in lines:
    if searchpart in line[0]:
      sep = float(line[9])
      mag_prim = float(line[10])
      component = line[2]
      try:
        mag_sec = float(line[11])
        s_Mag = p_Mag + (mag_sec - mag_prim)   #Absolute magnitude of the secondary
        s_spt = MS.GetSpectralType(MS.AbsMag, s_Mag)   #Spectral type of the secondary
      except ValueError:
        mag_sec = "Unknown"
        s_spt = "Unknown"
      components[component]["Separation"] = sep
      components[component]["Primary Magnitude"] = mag_prim
      components[component]["Secondary Magnitude"] = mag_sec
      components[component]["Secondary SpT"] = s_spt
  return components
      



#Check to see if the given star is a binary in the SB9 catalog
#if so, return some orbital information about all the components
SB9_location = "%s/Dropbox/School/Research/AstarStuff/TargetLists/SB9public" %(os.environ["HOME"])
def CheckMultiplicitySB9(starname):
  #First, find the record number in SB9
  infile = open("%s/Alias.dta" %SB9_location)
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
  infile = open("%s/Main.dta" %SB9_location)
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
        companion["Magnitude"] = Unknown #TODO: work out from blackbody
      companion["SpT"] = spt2

  #Finally, get orbit information for our star (Use the most recent publication)
  infile = open("%s/Orbits.dta" %SB9_location)
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



"""
  A function to determine the error bars from binomial statistics.
  Follows Burgasser et al 2003, ApJ 586, 512
  
  n is the number observed
  N is the sample size
"""
def BinomialErrors(n, N, debug=False, tol=0.001):
  n = int(n)
  N = int(N)
  p0 = float(n)/float(N)  #Observed probability
  guess_errors = numpy.sqrt(n*p0*(1.0-p0)/float(N))
  
  func = lambda x: numpy.sum([factorial(N+1)/(factorial(i)*factorial(N+1-i)) * x**i * (1.0-x)**(N+1-i) for i in range(0, n+1)])
  lower_errfcn = lambda x: numpy.abs(func(x) - 0.84)
  upper_errfcn = lambda x: numpy.abs(func(x) - 0.16)

  #Find the best fit. Need to do various tests for convergence
  converged = False
  methods = ['Bounded', 'Brent', 'Golden']
  i = 0
  while not converged and i < len(methods):
    result = minimize_scalar(lower_errfcn, bracket=(0.0, p0), bounds=(0.0, p0), method=methods[i])
    if result.fun < tol and result.x > 0.0 and result.x < p0:
      converged = True
      lower = result.x
    i += 1
  if not converged:
    print "BinomialErrors Warning! Fit did not succeed for lower bound. Using Gaussian approximation"
    lower = p0 - max(0.0, p0-guess_errors)

  #Now, do the same thing for the upper interval
  i = 0
  converged = False
  while not converged and i < len(methods):
    result = minimize_scalar(upper_errfcn, bracket=(p0, 1.0), bounds=(p0, 1.0), method=methods[i])
    if result.fun < tol and result.x > p0 and result.x < 1.0:
      converged = True
      upper = result.x
    i += 1
  if not converged:
    print "BinomialErrors Warning! Fit did not succeed for upper bound. Using Gaussian approximation"
    upper = p0 + min(1.0, p0+guess_errors)

  if debug:
    print "n = %i, N = %i\np0 = %.5f\tlower = %.5f\tupper=%.5f\n" %(n, N, p0, lower, upper)
  
  return lower, upper




"""
  The following series of functions will read in a fits file
  I think this works for all instruments, though maybe just HET...
"""
def ReadFits(datafile, errors=False, extensions=False, x=None, y=None, cont=None, debug=False):
  if debug:
    print "Reading in file %s: " %datafile

  if extensions:
    #This means the data is in fits extensions, with one order per extension
    #At least x and y should be given (and should be strings to identify the field in the table record array)
    if type(x) != str:
      x = raw_input("Give name of the field which contains the x array: ")
    if type(y) != str:
      y = raw_input("Give name of the field which contains the y array: ")
    orders = []
    hdulist = pyfits.open(datafile)
    if cont == None:
      if not errors:
        for i in range(1,len(hdulist)):
          data = hdulist[i].data
          xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y))
          orders.append(xypt)
      else:
        if type(errors) != str:
          errors = raw_input("Give name of the field which contains the errors/sigma array: ")
        for i in range(1,len(hdulist)):
          data = hdulist[i].data
          xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), err=data.field(errors))
          orders.append(xypt)
    elif type(cont) == str:
      if not errors:
        for i in range(1,len(hdulist)):
          data = hdulist[i].data
          xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), cont=data.field(cont))
          orders.append(xypt)
      else:
        if type(errors) != str:
          errors = raw_input("Give name of the field which contains the errors/sigma array: ")
        for i in range(1,len(hdulist)):
          data = hdulist[i].data
          xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), cont=data.field(cont), err=data.field(errors))
          orders.append(xypt)

  else:
    #Data is in multispec format rather than in fits extensions
    #Call Rick White's script
    retdict = multispec.readmultispec(datafile, quiet=not debug)
  
    #Check if wavelength units are in angstroms (common, but I like nm)
    hdulist = pyfits.open(datafile)
    header = hdulist[0].header
    hdulist.close()
    wave_factor = 1.0   #factor to multiply wavelengths by to get them in nanometers
    for key in sorted(header.keys()):
      if "WAT1" in key:
        if "label=Wavelength"  in header[key] and "units" in header[key]:
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
      wave = retdict['wavelen'][i]*wave_factor
      if errors == False:
        flux = retdict['flux'][i]
        err = numpy.ones(flux.size)*1e9
        err[flux > 0] = numpy.sqrt(flux[flux > 0])
      else:
        if type(errors) != int:
          errors = int(raw_input("Enter the band number (in C-numbering) of the error/sigma band: "))
        flux = retdict['flux'][0][i]
        err = retdict['flux'][errors][i]
      cont = FindContinuum.Continuum(wave, flux, lowreject=2, highreject=4)
      orders.append(DataStructures.xypoint(x=wave, y=flux, err=err , cont=cont))
  return orders





  

  
