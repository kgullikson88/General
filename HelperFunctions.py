"""
  Just a set of helper functions that I use often

"""
import os
import csv
import pySIMBAD as sim
from collections import defaultdict
import SpectralTypeRelations
import numpy
from scipy.misc import factorial
from scipy.optimize import fminbound

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
def BinomialErrors(n, N):
  n = int(n)
  N = int(N)
  p0 = float(n)/float(N)  #Observed probability
  #guess_errors = numpy.sqrt(n*p0*(1.0-p0)/float(N))
  
  func = lambda x: numpy.sum([factorial(N+1)/(factorial(i)*factorial(N+1-i)) * x**i * (1.0-x)**(N+1-i) for i in range(1, n+1)])
  lower_errfcn = lambda x: numpy.abs(func(x) - 0.84)
  upper_errfcn = lambda x: numpy.abs(func(x) - 0.16)
  
  lower = fminbound(lower_errfcn, 0, p0)
  upper = fminbound(upper_errfcn, p0, 1)
  
  return lower, upper











  

  
