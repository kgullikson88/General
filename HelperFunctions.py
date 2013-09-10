"""
  Just a set of helper functions that I use often

"""
import os
import csv
import pySIMBAD as sim
from collections import defaultdict
import SpectralTypeRelations


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
      print sep, mag_prim, mag_sec
      components[component]["Separation"] = sep
      components[component]["Primary Magnitude"] = mag_prim
      components[component]["Secondary Magnitude"] = mag_sec
      components[component]["Secondary SpT"] = s_spt
  return components
      
