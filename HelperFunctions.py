"""
  Just a set of helper functions that I use often

"""
import os
import csv
import pySIMBAD as sim
from collections import defaultdict


#Ensure a directory exists. Create it if not
def ensure_dir(f):
  d = os.path.dirname(f)
  if not os.path.exists(d):
    os.makedirs(d)


#Check to see if the given star is a binary in the WDS catalog
#if so, return the most recent separation and magnitude of all components
WDS_location = "%s/Dropbox/School/Research/AstarStuff/TargetLists/WDS_MagLimited.csv" %(os.environ["HOME"])
def CheckMultiplicityWDS(starname):
  link = sim.buildLink(starname)
  star = sim.simbad(link)
  all_names = star.names()

  #Check if one of them is a WDS name
  WDSname = ""
  for name in all_names:
    if "WDS" in name[:4]:
      WDSname = name
  if WDSname == "":
    return False
  searchpart = WDSname.split("J")[-1].split("A")[0]

  #Now, check the WDS catalog for this star
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
      except ValueError:
        mag_sec = "Unknown"
      print sep, mag_prim, mag_sec
      components[component]["Separation"] = sep
      components[component]["Primary Magnitude"] = mag_prim
      components[component]["Secondary Magnitude"] = mag_sec
  return components
      
