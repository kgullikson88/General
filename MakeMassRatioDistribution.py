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
NewDetections = {#"HIP 67782": [3900,],
                 "HIP 77336": [6500,],
                 #"HIP 85379": [6700,],
                 #               "HIP 72154": [3500,5700],
                 #"HIP 72515": [5700,],
                 "HIP 93393": [3800,],
#"HIP 92312": [5000,],
#"HIP 96840": [3500,5200],
#"HIP 100069": [3200,],
                 #               "HIP 8704": [3500,],
                 #"HIP 105972": [7600,],
                 #               "HIP 116582": [3200,6700],    THIS MIGHT BE A FOREGROUND BINARY!
                 "HIP 2548": [6500,],
                 #               "HIP 17527": [3500,],
                 #               "HIP 97870": [3300,],
                 #"HIP 13165": [3500,],
                 #"HIP 14143": [3500,],#7300],
                 #"HIP 20430": [5800,],
                 #               "HIP 105282": [3700,3700],
                 #"HIP 105282": [3700,],
                 #               "HIP 8016": [3500,3500],
                 "HIP 14043": [6200,],
                 #               "HIP 58590": [3800,],
                 "HIP 82673": [6000,],
                 #               "HIP 87108": [3500,4400],
                 #               "HIP 104139": [5000,],
                                 "HIP 95241": [4100,],
                 #               "HIP 116247": [3400,],
                 #               "HIP 117452": [4700,],
                 #               "HIP 60009": [3300,5500],
                 "HIP 60009": [5500,],
                 #               "HIP 63724": [3400,],
                 #               "HIP 79404": [3800,6000],
                 #               "HIP 92855": [4000,5800],
                 "HIP 112029": [6300,],
                 #               "HIP 76600": [5600,],
                 #               "HIP 77516": [3500,],
                 #               "HIP 78820": [4000,],
                 "HIP 88816": [6400,],
                 #               "HIP 80883": [3700,],
                 #               "HIP 78554": [3400,],
                 "HIP 15444": [6100,],
                 "HIP 20789": [5400,],
                 "HR 545":    [5000,],
                 "HIP 5132":  [3700,],
                 }

#Do the same thing for known binaries not in WDS or SB9
KnownBinaries = {"HIP 76267": [5800,]
                 }


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
      numstars += 1.0


  #Return mass_ratio lists
  mass_ratios = [min(q, 1.0) for q in mass_ratios]
  return mass_ratios, new_massratios, multiplicity, numstars



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
      numstars += 1.0


  #Make some plots
  mass_ratios = [min(q, 1.0) for q in mass_ratios]
  return mass_ratios, new_massratios, multiplicity, numstars




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
  for directory in dirlist:
    starlist = [f for f in os.listdir(directory) if (f.startswith("HR") or f.startswith("HIP")) and f.endswith("-0.fits")]
    for star in starlist:
      #First, get the known companions
      multiple = False
      sb = False
      header = pyfits.getheader("%s/%s" %(directory, star))
      print star
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
      numstars += 1.0


  #Make some plots
  mass_ratios = [min(q, 1.0) for q in mass_ratios]
  return mass_ratios, new_massratios, multiplicity, numstars



if __name__ == "__main__":
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
  ch_mass_ratios, ch_new_massratios, ch_num_multiple, ch_numstars = GetCHIRONdist(MS=MS)
  het_mass_ratios, het_new_massratios, het_num_multiple, het_numstars = GetHETdist(MS=MS)
  ts_mass_ratios, ts_new_massratios, ts_num_multiple, ts_numstars = GetTS23dist(MS=MS)
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
    for q in ch_mass_ratios:
      mass_ratios.append(q)
    for q in ts_mass_ratios:
      mass_ratios.append(q)
    for q in ch_new_massratios:
      new_massratios.append(q)
    for q in ts_new_massratios:
      new_massratios.append(q)
    num_multiple += ch_num_multiple + ts_num_multiple
    numstars += ch_numstars + ts_numstars
  

  print "Multiplicity fraction = %g" %(num_multiple/numstars)
  bins = numpy.arange(0.0, 1.1, 0.1)
  plt.figure(1)
  if len(new_massratios) > 0:
    mass_ratios = [mass_ratios, new_massratios]
  else:
    mass_ratios = [mass_ratios, []]
  
  print "\n*** Mass Ratios found: ***"
  for idx in [0,1]:
    for q in mass_ratios[idx]:
      print q
  print "\n"

  
  if color:
    plt.hist(mass_ratios, bins=bins, color=['chocolate','deepskyblue'], histtype='barstacked', label=["Known companions", "Candidate companions"], rwidth=1)
  else:
    plt.hist(mass_ratios, bins=bins, color=['0.25','0.5'], histtype='barstacked', label=["Known companions", "Candidate companions"], rwidth=1)
  plt.legend(loc='best')
  #Make error bars
  nums = numpy.zeros(bins.size-1)
  for i in range(len(mass_ratios)):
    nums += numpy.histogram(mass_ratios[i], bins=bins)[0]
  lower = []
  upper = []
  for n in nums:
    pl, pu = HelperFunctions.BinomialErrors(n, numstars, debug=True)
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
  plt.xlabel(r"$\rm M_s/M_p$", fontsize=20)
  plt.ylabel("Number", fontsize=20)
  plt.title("Mass Ratio Distribution for Companions within 200 AU", fontsize=30)
  
  #plt.figure(2)
  #plt.hist(mass_ratios, bins=bins, color=['gray','green'], cumulative=True, normed=True, histtype='step', linewidth=2, stacked=True)
  #plt.plot(bins, bins, 'k--', linewidth=2)
  #plt.xlabel(r"$\rm M_s/M_p$")
  #plt.ylabel("Cumulative Frequency")
  plt.show()
      
