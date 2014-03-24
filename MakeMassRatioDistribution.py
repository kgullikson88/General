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


"""
  The following dictionary shows the new detections from my data.
  It includes both brand-new detections, and detections of the secondary
    star lines in known SB1s.
  The key is the star name, and the value is the estimated temperature
"""
NewDetections = {"HIP 60009": [5500,],
                 "HIP 82673": [6000,],
                 "HIP 88116": [6000,],
                 "HIP 110838": [4400,],
                 "HIP 93393": [3800,],
                 "HR 545": [5000,],
                 "HIP 22958": [6200,]
                 }

#Do the same thing for known binaries not in WDS or SB9
KnownBinaries = {}





"""
  This function will search the WDS catalog for known companions within 'sep' arcseconds
"""
def GetWDSCompanions(starname, sep=5.0, MS=None):
  if MS == None:
    MS = SpectralTypeRelations.MainSequence()
  companions = HelperFunctions.CheckMultiplicityWDS(starname)
  companion_info = []
  if companions:
    for configuration in companions:
      component = companions[configuration]
      if component["Separation"] < sep:
        s_spt = component["Secondary SpT"]
        if s_spt == "Unknown":
          print "Warning! Star %s has a companion with unknown magnitude/spectral type in WDS!" %starname
        else:
          mass = MS.Interpolate(MS.Mass, s_spt)
          companion_info.append((component["Separation"], mass))
  return companion_info


"""
   This function searches the SB9 catalog for spectroscopic companions
   The return type is determined by what information is given in the database,
     but always consists of an integer followed by a float

     -If no companion exists, the return values are 0,0
     -If both K1 and K2 are known (double-lined binary):
        -integer returned is 1, float returned is the mass ratio
     -If only K1 is known (the case for most):
        -integer returned is 2, and the float is the mass function f(M2)=M2 (sin(i))^2 / (1+1/q)^2
"""
def GetSB9Companions(starname, MS=None):
  if MS == None:
    MS = SpectralTypeRelations.MainSequence()
  companion = HelperFunctions.CheckMultiplicitySB9(starname)
  if not companion:
    return 0, 0
  if companion["K1"] != "Unknown" and companion["K2"] != "Unknown":
    q = companion["K1"] / companion["K2"]
    return 1, q
  elif companion["K1"] != "Unknown":
    K1 = companion["K1"]
    P = companion["Period"]
    mass_fcn = (K1*units.km.to(units.cm))**3 * (P*units.day.to(units.second)) / (2*numpy.pi*constants.G.cgs.value)
    return 2, mass_fcn * units.gram.to(units.solMass)


"""
    This function gets the mass-ratio lists from the chiron data directory.
"""
def GetCHIRONdist(datadir="CHIRON_data/", MS=None):
  if not datadir.endswith("/"):
    datadir = datadir + "/"
  dirlist = ["%s%s" %(datadir, d) for d in os.listdir(datadir) if d.startswith("201")]

  if MS == None:
    MS = SpectralTypeRelations.MainSequence()

  multiplicity = 0.0
  numstars = 0.0
  mass_ratios = []
  new_massratios = []
  stars = []
  for directory in dirlist:
    starlist = [f for f in os.listdir(directory) if f.startswith("H") and f.endswith("-0.fits")]
    for star in starlist:
      #First, get the known companions
      multiple = False
      sb = False
      header = pyfits.getheader("%s/%s" %(directory, star))
      starname = header['OBJECT']
      print starname
      stardata = StarData.GetData(starname)
      primary_mass = MS.Interpolate(MS.Mass, stardata.spectype[:2])
      distance = 1000.0 / stardata.par
      known_companions = GetWDSCompanions(starname, MS=MS, sep=200.0/distance)
      code, value = GetSB9Companions(starname)
      if len(known_companions) > 0:
        multiple = True
        for comp in known_companions:
          print "\tq = %g" %(comp[1]/(primary_mass))
          mass_ratios.append(comp[1]/primary_mass)
      if code == 1:
        sb = True
        multiple = True
        q = value
        wds = False
        for comp in known_companions:
          if abs(q-comp[1]) < 0.1 and comp[0] < 4.0:
            wds = True
        if not wds:
          mass_ratios.append(q)
        else:
          print "Spectroscopic binary found which may match a WDS companion."
          usr = raw_input("Use both (y or n)? ")
          if "y" in usr:
            mass_ratios.append(q)
        print "Spectroscopic companion with q = %g" %q
      elif code == 2:
        print "Single-lined spectroscopic companion to %s found! Double-lined in my data?" %starname
        multiple = True

      #Put in known binaries that aren't in WDS or SB9
      if starname in KnownBinaries:
        for T in KnownBinaries[starname]:
          spt = MS.GetSpectralType(MS.Temperature, T)
          mass = MS.Interpolate(MS.Mass, spt)
          new_q = mass/primary_mass
          massratios.append(new_q)
          multiple = True
          
          

      #Now, put in my data
      if starname in NewDetections:
        for T in NewDetections[starname]:
          spt = MS.GetSpectralType(MS.Temperature, T)
          mass = MS.Interpolate(MS.Mass, spt)
          new_q = mass/primary_mass
          previously_known = False
          for comp in known_companions:
            if abs(new_q - comp[1]) < 0.1 and comp[0] < 4.0:
              previously_known = True
          if sb and abs(new_q - q) < 0.1:
            previously_known = True
        if not previously_known:
          new_massratios.append(new_q)
          multiple = True

      #Keep track of total binary fraction
      if multiple:
        multiplicity += 1
        stars.append(starname)
      numstars += 1.0


  #Return mass_ratio lists
  mass_ratios = [min(q, 1.0) for q in mass_ratios]
  return mass_ratios, new_massratios, multiplicity, numstars, stars



"""
    This function returns the mass-ratio lists from HET data (same format as for CHIRON data)
"""
def GetHETdist(datadir="HET_data/", MS=None):
  if not datadir.endswith("/"):
    datadir = datadir + "/"
  dirlist = ["%s%s" %(datadir, d) for d in os.listdir(datadir) if d.startswith("201")]

  if MS == None:
    MS = SpectralTypeRelations.MainSequence()

  multiplicity = 0.0
  numstars = 0.0
  mass_ratios = []
  new_massratios = []
  stars = []
  for directory in dirlist:
    starlist = [f for f in os.listdir(directory) if ((f.startswith("HR") and not f.startswith("HRS")) or f.startswith("HIP")) and f.endswith("-0.fits")]
    for star in starlist:
      #First, get the known companions
      multiple = False
      sb = False
      header = pyfits.getheader("%s/%s" %(directory, star))
      starname = header['OBJECT'].split()[0].replace("_"," ")
      print starname
      stardata = StarData.GetData(starname)
      primary_mass = MS.Interpolate(MS.Mass, stardata.spectype[:2])
      distance = 1000.0 / stardata.par
      known_companions = GetWDSCompanions(starname, MS=MS, sep=200.0/distance)
      code, value = GetSB9Companions(starname)
      if len(known_companions) > 0:
        multiple = True
        for comp in known_companions:
          print "\tq = %g" %(comp[1]/(primary_mass))
          mass_ratios.append(comp[1]/primary_mass)
      if code == 1:
        sb = True
        multiple = True
        q = value
        wds = False
        for comp in known_companions:
          if abs(q-comp[1]) < 0.1 and comp[0] < 4.0:
            wds = True
        if not wds:
          mass_ratios.append(q)
        else:
          print "Spectroscopic binary found which may match a WDS companion."
          usr = raw_input("Use both (y or n)? ")
          if "y" in usr:
            mass_ratios.append(q)
        print "Spectroscopic companion with q = %g" %q
      elif code == 2:
        print "Single-lined spectroscopic companion to %s found! Double-lined in my data?" %starname
        multiple = True

      #Put in known binaries that aren't in WDS or SB9
      if starname in KnownBinaries:
        for T in KnownBinaries[starname]:
          spt = MS.GetSpectralType(MS.Temperature, T)
          mass = MS.Interpolate(MS.Mass, spt)
          new_q = mass/primary_mass
          massratios.append(new_q)
          multiple = True

      #Now, put in my data
      if starname in NewDetections:
        for T in NewDetections[starname]:
          spt = MS.GetSpectralType(MS.Temperature, T)
          mass = MS.Interpolate(MS.Mass, spt)
          new_q = mass/primary_mass
          previously_known = False
          for comp in known_companions:
            if abs(new_q - comp[1]) < 0.1 and comp[0] < 4.0:
              previously_known = True
          if sb and abs(new_q - q) < 0.1:
            previously_known = True
        if not previously_known:
          new_massratios.append(new_q)
          multiple = True

      #Keep track of total binary fraction
      if multiple:
        multiplicity += 1
        stars.append(starname)
      numstars += 1.0


  #Make some plots
  mass_ratios = [min(q, 1.0) for q in mass_ratios]
  return mass_ratios, new_massratios, multiplicity, numstars, stars




"""
    This function returns the mass-ratio lists from TS23 data (same format as for CHIRON data)
"""
def GetTS23dist(datadir="McDonaldData/", MS=None):
  if not datadir.endswith("/"):
    datadir = datadir + "/"
  dirlist = ["%s%s" %(datadir, d) for d in os.listdir(datadir) if d.startswith("201") and not d.startswith("201301")]

  if MS == None:
    MS = SpectralTypeRelations.MainSequence()

  multiplicity = 0.0
  numstars = 0.0
  mass_ratios = []
  new_massratios = []
  stars = []
  for directory in dirlist:
    starlist = [f for f in os.listdir(directory) if (f.startswith("HR") or f.startswith("HIP")) and f.endswith("-0.fits")]
    for star in starlist:
      #First, get the known companions
      multiple = False
      sb = False
      header = pyfits.getheader("%s/%s" %(directory, star))
      starname = header['OBJECT']
      print starname
      stardata = StarData.GetData(starname)
      primary_mass = MS.Interpolate(MS.Mass, stardata.spectype[:2])
      distance = 1000.0 / stardata.par
      known_companions = GetWDSCompanions(starname, MS=MS, sep=200.0/distance)
      code, value = GetSB9Companions(starname)
      if len(known_companions) > 0:
        multiple = True
        for comp in known_companions:
          print "\tq = %g" %(comp[1]/(primary_mass))
          mass_ratios.append(comp[1]/primary_mass)
      if code == 1:
        sb = True
        multiple = True
        q = value
        wds = False
        for comp in known_companions:
          if abs(q-comp[1]) < 0.1 and comp[0] < 4.0:
            wds = True
        if not wds:
          mass_ratios.append(q)
        else:
          print "Spectroscopic binary found which may match a WDS companion."
          usr = raw_input("Use both (y or n)? ")
          if "y" in usr:
            mass_ratios.append(q)
        print "Spectroscopic companion with q = %g" %q
      elif code == 2:
        print "Single-lined spectroscopic companion to %s found! Double-lined in my data?" %starname
        multiple = True

      #Put in known binaries that aren't in WDS or SB9
      if starname in KnownBinaries:
        for T in KnownBinaries[starname]:
          spt = MS.GetSpectralType(MS.Temperature, T)
          mass = MS.Interpolate(MS.Mass, spt)
          new_q = mass/primary_mass
          massratios.append(new_q)
          multiple = True

      #Now, put in my data
      if starname in NewDetections:
        for T in NewDetections[starname]:
          spt = MS.GetSpectralType(MS.Temperature, T)
          mass = MS.Interpolate(MS.Mass, spt)
          new_q = mass/primary_mass
        previously_known = False
        for comp in known_companions:
          if abs(new_q - comp[1]) < 0.1 and comp[0] < 4.0:
            previously_known = True
          if sb and abs(new_q - q) < 0.1:
            previously_known = True
        if not previously_known:
          new_massratios.append(new_q)
          multiple = True

      #Keep track of total binary fraction
      if multiple:
        multiplicity += 1
        stars.append(starname)
      numstars += 1.0


  #Make some plots
  mass_ratios = [min(q, 1.0) for q in mass_ratios]
  return mass_ratios, new_massratios, multiplicity, numstars, stars



def main1():
  dirlist = []
  chiron_only = False
  het_only = False
  ts23_only = False
  color = False
  for arg in sys.argv[1:]:
    if "-chiron" in arg:
      chiron_only = True
    elif "-het" in arg:
      het_only = True
    elif "-ts23" in arg:
      ts23_only = True
    elif "-color" in arg:
      color = True
    else:
      dirlist.append(arg)
  if len(dirlist) == 0:
    dirlist = [d for d in os.listdir("./") if d.startswith("2013")]

  MS = SpectralTypeRelations.MainSequence()

  #Get data
  ch_mass_ratios, ch_new_massratios, ch_num_multiple, ch_numstars, ch_stars = GetCHIRONdist(MS=MS)
  het_mass_ratios, het_new_massratios, het_num_multiple, het_numstars, het_stars = GetHETdist(MS=MS)
  ts_mass_ratios, ts_new_massratios, ts_num_multiple, ts_numstars, ts_stars = GetTS23dist(MS=MS)
  if het_only:
    mass_ratios = het_mass_ratios
    new_massratios = het_new_massratios
    num_multiple = het_num_multiple
    numstars = het_numstars
  elif chiron_only:
    mass_ratios = ch_mass_ratios
    new_massratios = ch_new_massratios
    num_multiple = ch_num_multiple
    numstars = ch_numstars
  elif ts23_only:
    mass_ratios = ts_mass_ratios
    new_massratios = ts_new_massratios
    num_multiple = ts_num_multiple
    numstars = ts_numstars
  else:
    mass_ratios = het_mass_ratios
    new_massratios = het_new_massratios
    num_multiple = het_num_multiple
    numstars = het_numstars
    stars = het_stars
    for i, q in enumerate(ch_mass_ratios):
      if ch_stars[i] not in stars:
        mass_ratios.append(q)
      else:
        print "Duplicate star found: %s" %ch_stars[i]
    for i, q in enumerate(ts_mass_ratios):
      if ts_stars[i] not in stars:
        mass_ratios.append(q)
      else:
        print "Duplicate star found: %s" %ts_stars[i]
    for i, q in enumerate(ch_new_massratios):
      if ch_stars[i] not in stars:
        new_massratios.append(q)
      else:
        print "Duplicate star found: %s" %ch_stars[i]
    for i, q in enumerate(ts_new_massratios):
      if ts_stars[i] not in stars:
        new_massratios.append(q)
      else:
        print "Duplicate star found: %s" %ts_stars[i]
    num_multiple += ch_num_multiple + ts_num_multiple
    numstars += ch_numstars + ts_numstars
  

  print "Multiplicity fraction = %g" %(num_multiple/numstars)
  bins = numpy.arange(0.0, 1.1, 0.1)
  if len(new_massratios) > 0:
    mass_ratios = [mass_ratios, new_massratios]
  else:
    mass_ratios = [mass_ratios, []]
  
  print "\n\n***  All mass ratios  ***"
  for idx in range(2):
    for q in mass_ratios[idx]:
      print q
    print "\n\n\n\n#New detections:"

  print "\n"
  
  #Plot
  plt.figure(1)

  
  if color:
    plt.hist(mass_ratios, bins=bins, color=['chocolate','deepskyblue'], histtype='barstacked', label=["Known companions", "New companions"], rwidth=1)
  else:
    plt.hist(mass_ratios, bins=bins, color=['0.25','0.5'], histtype='barstacked', label=["Known companions", "New companions"], rwidth=1)
  plt.legend(loc='best', fancybox=True)
  #Make error bars
  nums = numpy.zeros(bins.size-1)
  for i in range(len(mass_ratios)):
    nums += numpy.histogram(mass_ratios[i], bins=bins)[0]
  lower = []
  upper = []
  for n in nums:
    pl, pu = HelperFunctions.BinomialErrors(n, numstars, debug=False)
    lower.append(pl*numpy.sqrt(numstars))
    upper.append(pu*numpy.sqrt(numstars))
  #p = nums/nums.sum()
  #errors = nums*p*(1.0-p)
  plt.errorbar(bins[:-1] + 0.05, nums, yerr=[lower,upper], fmt=None, ecolor='0.0', elinewidth=2, capsize=5)
  """
  if len(new_massratios) > 0:
    y,edges = numpy.histogram(new_massratios, bins=bins)
    print y
    print edges
    plt.bar(bins[:-1], y, bottom=numpy.array(height), color='green', align='edge')
    #plt.hist(new_massratios, bins=bins, bottom=height, color='green')
  """
  plt.xlabel(r"$\rm M_s/M_p$", fontsize=17)
  plt.ylabel("Number", fontsize=17)
  plt.title("Mass Ratio Distribution for Companions within 200 AU", fontsize=20)
  
  #plt.figure(2)
  #plt.hist(mass_ratios, bins=bins, color=['gray','green'], cumulative=True, normed=True, histtype='step', linewidth=2, stacked=True)
  #plt.plot(bins, bins, 'k--', linewidth=2)
  #plt.xlabel(r"$\rm M_s/M_p$")
  #plt.ylabel("Cumulative Frequency")
  plt.show()
      

import re   #Move to top!
from collections import defaultdict   #Is this already imported?
import warnings
def ParseConfiguration(config, plx=10.0, vmag=5.5):
  # First, find the subsystems using open/close parentheses
  # Find the open and close parentheses
  paropen = numpy.array([m.start() for m in re.finditer('\(', config)])
  parclose = numpy.array([m.start() for m in re.finditer('\)', config)])

  #Each subsystem is enclosed in parentheses.
  subsystems = defaultdict(list)
  numopen = 0
  systems = []
  for i, char in enumerate(config):
    if i in paropen:
      numopen += 1
      subsystems[numopen].append(i)
    elif i in parclose:
      subsystems[numopen].append(i)
      numopen -= 1


  # Determine the mass ratios of everything we know
  massratios = []
  for level in sorted(subsystems.keys())[::-1]:
    for i in range(0, len(subsystems[level]), 2):
      start = subsystems[level][i]+1
      end = subsystems[level][i+1]
      sysconfig = config[start:end]
      if "WD" in sysconfig.upper():
        continue
      print level, ": ", sysconfig
      if "(" in sysconfig and "?;" not in sysconfig:
        if sysconfig.find("(") < sysconfig.find("+"):
          p_mass = float(sysconfig.split("(")[0].split(")")[0])
          if sysconfig.find("+") < sysconfig.split(")")[1].find("("):
            s_mass = float(sysconfig.split("(")[1].split(")")[1])
          else:
            secondary = sysconfig.split("+")[1].split(";")[0].strip()
            s_spt = GetSpectralType(secondary, plx, vmag)
            s_mass = MS.Interpolate(MS.Mass, s_spt)

        else:
          s_mass = float(sysconfig.split("(")[1].split(")")[0])
          primary = sysconfig.split("+")[0].split(";")[0].strip()
          p_spt = GetSpectralType(primary, plx, vmag)
          p_mass = MS.Interpolate(MS.Mass, p_spt)

        massratios.append(s_mass/p_mass)
        seg = config.split(sysconfig)
        n = len(sysconfig)
        config = "%s%.3f%s%s" %(seg[0], p_mass + s_mass, "0"*(n-5), seg[1])
        print "NEW: ", config

      elif "?;" not in sysconfig:
        if "sim" in sysconfig:
          p_spt = GetSpectralType(primary, plx, vmag)
          s_spt = p_spt
        else:
          #Get the primary and secondary stars
          primary = sysconfig.split("+")[0].strip().strip(":")
          secondary = sysconfig.split("+")[1].split(";")[0].strip()
          p_spt = GetSpectralType(primary, plx, vmag)
          idx = primary.find(p_spt)
          if idx != 0:
            p_mag = float(primary[:idx])
          else:
            p_mag = vmag
          s_spt = GetSpectralType(secondary, plx, vmag, primary_spt=p_spt, primary_mag=p_mag)
        
        p_mass = MS.Interpolate(MS.Mass, p_spt)
        s_mass = MS.Interpolate(MS.Mass, s_spt)
        massratios.append(s_mass/p_mass)
        seg = config.split(sysconfig)
        n = len(sysconfig)
        config = "%s%.3f%s%s" %(seg[0], p_mass + s_mass, "0"*(n-5), seg[1])
        print "NEW: ", config
  return massratios



def GetSpectralType(star, plx, sysmag, primary_spt=None, primary_mag=None):
  found = False
  for typ in ["O", "B", "A", "F", "G", "K", "M"]:
    if typ in star:
      found = True
      idx = star.find(typ)
      spt = star[idx:]

      if "I" in spt:
        warnings.warn("The spectral type is not main sequence: %s" %spt)
      
      #Check for any other letters:
      m = re.search("[A-Za-z]", spt[1:])
      if m != None:
        idx = m.start()+1
        spt = spt[:idx]


  if found:
    return spt.split("V")[0]
  else:
    #Only a magnitude is given.
    #Check to see if the user gave the primary spectral type/magnitude
    if primary_mag != None and primary_spt != None:
      absmag = MS.Interpolate(MS.AbsMag, primary_spt)
      absmag2 = primary_mag - 5*numpy.log10(1000.0/plx) + 5
      reddening = absmag2 - absmag 
    else:
      warnings.warn("Cannot determine V band redenning! Setting to 0")
      reddening = 0.0

    # Convert this magnitude to absolute magnitude using distance
    mag = float(star)
    absmag = mag - 5*numpy.log10(1000.0/plx) + 5 - reddening
    return MS.GetSpectralType(MS.AbsMag, absmag, interpolate=True)

  







def ReadSampleFile(samplefile, startcol=10, endcol=544, namecol=1, configcol=2, obscol=41, parcol=4, magcol=5, delimiter="|"):
  infile = open(samplefile)
  lines = infile.readlines()
  infile.close()
  massratios = []
  for line in lines[startcol:endcol]:
    segments = line.split(delimiter)
    name = segments[namecol].strip("'").strip()
    configuration = segments[configcol].strip("'").strip()
    observed = segments[obscol].strip("'").strip()
    try:
      parallax = float(segments[parcol].strip("'").strip())
    except ValueError:
      parallax = 50.0
      warnings.warn("Parallax not parseable: %s" %(segments[parcol].strip("'").strip()))
    try:
      vmag = float(segments[magcol].strip("'").strip())
    except ValueError:
      vmag = 5.5
      warnings.warn("V band magnitude not parseable: %s" %(segments[magcol].strip("'").strip()))
    
    print name
    print configuration
    #try:
    if observed == "1" or observed == "0":
      mrs = ParseConfiguration(configuration, plx=parallax, vmag=vmag)
      for q in mrs:
        massratios.append(q)
    #except ValueError:
    #  print configuration



  plt.hist(q)
    



def main2():
  """
    A new version of the driver function, which reads in the full sample spreadsheet
    that contains multiplicity information from Eggleton and Tokovinin 2008
  """
  sampledatafile = "%s/Dropbox/School/Research/AstarStuff/TargetLists/FullSample.csv" %(os.environ['HOME'])
  known_systems = ReadSampleFile(sampledatafile)










if __name__ == "__main__":
  MS = SpectralTypeRelations.MainSequence()
  main2()  
