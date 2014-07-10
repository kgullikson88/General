"""
  This script goes through the stars observed, and searches for both known and new
  companions to each target. Right now, it only does known companions automatically
"""

import pySIMBAD as sim
import sys
import os
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
import matplotlib.pyplot as plt
import numpy
import HelperFunctions
import SpectralTypeRelations
from astropy.io import fits as pyfits
import StarData
from astropy import units, constants
from astropy.io import ascii
from re import search



"""
Make some class initializations to use
"""
MS = SpectralTypeRelations.MainSequence()
PMS = SpectralTypeRelations.PreMainSequence(pms_tracks_file="%s/Dropbox/School/Research/Stellar_Evolution/Baraffe_Tracks.dat" %(os.environ["HOME"]), track_source="Baraffe")
PMS2 = SpectralTypeRelations.PreMainSequence()


"""
  The following dictionary shows the new detections from my data.
  It includes both brand-new detections, and detections of the secondary
    star lines in known SB1s.
  The key is the star name, and the value is the estimated temperature
"""
NewDetections = {"HIP 16244":  5700,  #Start of TS23 candidates
                 "HIP 109521": 5000,
                 "HR 545":     4850,
                 "HIP 3478":   6200,
                 "HIP 39847":  6600,
                 "HIP 12332":  5700,
                 "HIP 14764":  6300,
                 "HIP 22833":  5300,

                 "HIP 41039":  5600,  #Start of CHIRON candidates
                 "HIP 74117":  6100,
                 "HR 5605":    6100,
                 "HIP 79404":  5700,
                 "HIP 109139": 6600,
                 "HIP 93225":  5800,
                 "HIP 38593":  5800,
                 "HIP 51362":  6000,
                 "HIP 37297":  6100,
                 "HIP 26126":  6100,
                 "HIP 32607":  5200,
                 "HIP 79199":  5700,

                 "HIP 77336":  6500,  #Start of HET candidates
                 "HIP 93393":  3800
                 }

#The total number of stars observed. Need to edit this often!
numstars = 292




def GetMass(spt, age):
  """
  Returns the mass of the system in solar masses
  spt: Spectral type of the star
  age: age, in years, of the system
  """

  #Get temperature
  end = search("[0-9]", spt).end()
  T = MS.Interpolate(MS.Temperature, spt[:end])

  # Determine which tracks to use
  if spt[0] == "O" or spt[0] == "B" or spt[0] == "A" or spt[0] == "F":
    return PMS2.GetFromTemperature(age, T, key="Mass")
  else:
    return PMS.GetFromTemperature(age, T, key="Mass")


def GetAge(sptlist):
  """
  Returns the age of the system, in years, given a list
  of spectral types. It determines the age as the 
  main sequence lifetime of the earliest-type star
  """
  lowidx = 999
  for spt in sptlist:
    if "I" in spt:
      #Pre-main sequence. Just use the age of an early O star, 
      # which is ~ 1Myr
      lowidx = 1
      break
    end = search("[0-9]", spt).end()
    idx = MS.SpT_To_Number(spt[:end])
    if idx < lowidx:
      lowidx = idx

  spt = MS.Number_To_SpT(lowidx)
  Tprim = MS.Interpolate(MS.Temperature, spt)
  age = PMS2.GetMainSequenceAge(Tprim, key="Temperature")
  return age




def GetKnownMultiples(multiple_file, maxsep=200.0, starlist=None):
  """
  Given a multiplicity file and an optional list of stars,
  return the mass ratios of all of the systems
  """
  systems = ascii.read(multiple_file)[20:]
  multiple_systems = {}
  massratios = []
  for primary in systems:
    starname = primary[0]
    if starlist is not None and starname not in starlist:
      continue
    spt = primary[1]
    plx = float(primary[2])/1000.0   #Parallax in arcsec
    numcomponents = int(primary[4])
    print starname, numcomponents
    known_stars = [spt,]
    for comp in range(numcomponents):
      spt = primary[6+4*comp]
      sep = primary[7+4*comp]
      if "?" in sep:
        sep = 0.1
      else:
        sep = float(sep)
      if not "?" in spt and sep/plx <= maxsep:
        known_stars.append(spt)

    print known_stars, '\n'

    if len(known_stars) > 1:
      #Get age of the system
      age = GetAge(known_stars)
      M1 = GetMass(known_stars[0], age)
      multiple_systems[starname] = [[M1, age],]
      for obj in known_stars[1:]:
        M2 = GetMass(obj, age)
        q = min(M2/M1, M1/M2)
        massratios.append(q)
        multiple_systems[starname].append([obj, q])


  return massratios, multiple_systems



def GetSpectralType(name, star_file):
  """
  Given a star name and a file containing name and spectral type,
  return the spectral type of the star
  """
  stars = ascii.read(star_file)[20:]
  found = False
  for star in stars:
    if star[0] == name:
      found = True
      spt = star[1]
  if not found:
    spt = raw_input("Star %s not found! What is it's spectral type? " %name)
  return spt





def MakeDistribution():
  """
  Find the known multiples for each instrument. 
  Then, add in the new data (from dictionary above for now)
  """

  companion_file = "%s/Dropbox/School/Research/AstarStuff/TargetLists/Multiplicity.csv" %(os.environ["HOME"])
  ts23_file = "%s/Dropbox/School/Research/AstarStuff/TargetLists/Observed_TS23.csv" %(os.environ["HOME"])
  het_file = "%s/Dropbox/School/Research/AstarStuff/TargetLists/Observed_HET.csv" %(os.environ["HOME"])
  chiron_file = "%s/Dropbox/School/Research/AstarStuff/TargetLists/Observed_CHIRON.csv" %(os.environ["HOME"])

  all_massratios = []
  new_massratios = []
  for instrument in [ts23_file, het_file, chiron_file]:
    #Get the objects observed with the instrument
    observed = ascii.read(instrument)[5:]
    stars = []
    for star in observed:
      starname = star[1]
      print starname
      try:
        test = int(starname)
        stars.append("HIP %i" %test)
      except ValueError:
        stars.append(starname)
    known_massratios, known_systems = GetKnownMultiples(companion_file, starlist = stars)

    for name in known_systems.keys():
      if name in NewDetections.keys():
        # Get the mass ratio for the new system
        M1, age = known_systems[name][0]
        spt = MS.GetSpectralType(MS.Temperature, NewDetections[name], interpolate=True)
        M2 = GetMass(spt, age)

        #Overlap between known systems and the new stuff. Ask what to do.
        print "System %s has both known and new companions listed" %name
        print "What do you want to do?"
        for i in range(1,len(known_systems[name])):
          print "  [%i]: Replace %s companion with new detection (%i K)" %(i, known_systems[name][i][0], NewDetections[name])
        print "  [a]: Add new detection to system"
        inp = raw_input(" ")
        if inp.lower() == "a":
          new_massratios.append(min(M1/M2, M2/M1))
          NewDetections[name] = 0  #Do this as a flag to not add this companion again
        else:
          inp = int(inp)
          known_systems[name][inp][1] = min(M1/M2, M2/M1)
          NewDetections[name] = 0

      # Now, just add the mass-ratios in known_systems to all_massratios
      for companion in known_systems[name][1:]:
        all_massratios.append(companion[1])

  # Go through the new detections, adding them to new_massratios
  # Ignore anything with 0 temperature (set as such above)
  for name in NewDetections.keys():
    if abs(NewDetections[name]) > 1:
      p_spt = GetSpectralType(name, companion_file)
      age = GetAge([p_spt])
      M1 = GetMass(p_spt, age)
      s_spt = MS.GetSpectralType(MS.Temperature, NewDetections[name], interpolate=True)
      M2 = GetMass(s_spt, age)
      new_massratios.append(min(M1/M2, M2/M1))


  #Save the mass-ratios
  numpy.savetxt("Known_Massratios.txt", numpy.transpose(all_massratios))
  numpy.savetxt("New_Massratios.txt", numpy.transpose(new_massratios))

  return all_massratios, new_massratios

  

def GetDistribution():
  """
  Just reads in the distribution saved in MakeDistribution
  """
  known = numpy.loadtxt("Known_Massratios.txt")
  new = numpy.loadtxt("New_Massratios.txt")
  return known, new









if __name__ == "__main__":
  #known_massratios, new_massratios = MakeDistribution()
  known_massratios, new_massratios = GetDistribution()

  #Finally, plot
  bins = numpy.arange(0.0, 1.1, 0.1)
  fig = plt.figure(dpi=120)
  ax = fig.add_subplot(111)
  qdist = [known_massratios, new_massratios]
  ax.hist(qdist,
          bins=bins, 
          label=("Known binary systems", "New candidate binary systems"), 
          stacked=True, 
          color=['DarkSlateBlue', 'SaddleBrown'],
          rwidth=1.0)
  
  #Make error bars
  nums = numpy.zeros(bins.size - 1)
  for i in range(len(qdist)):
    nums += numpy.histogram(qdist[i], bins=bins)[0]
  lower = []
  upper = []
  for n in nums:
    pl, pu = HelperFunctions.BinomialErrors(n, numstars, debug=False)
    lower.append(pl*numpy.sqrt(numstars))
    upper.append(pu*numpy.sqrt(numstars))
  ax.errorbar(bins[:-1]+0.05, 
              nums, 
              yerr=[lower, upper], 
              fmt=None, 
              ecolor='0.0', 
              elinewidth=2, 
              capsize=5)


  #Add the histogram from Bate 2014 binaries with M > 1.5Msun
  #bate = [1,4,2,5,9]
  #batebins = [0.1, 0.3, 0.5, 0.7, 0.9]
  #batewidth = 0.2
  #ax.bar(numpy.array(batebins)-batewidth/2, bate, batewidth, color='none', edgecolor='red', lw=3)

  #Labels
  ax.set_xlabel(r"$M_s/M_p$", fontsize=15)
  ax.set_ylabel("Number", fontsize=15)
  ax.set_ylim((0, 15))
  ax.set_title("Mass Ratio Distribution Within 200 AU", fontsize=20)
  leg = ax.legend(loc='best', fancybox=True)
  leg.get_frame().set_alpha(0.5)
  plt.show()
