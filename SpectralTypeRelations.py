from collections import defaultdict
from scipy.interpolate import UnivariateSpline
import numpy
import DataStructures
import Units

#Provides relations temperature, luminosity, radius, and mass for varius spectral types
#Data comes from Carroll and Ostlie book, or interpolated from it
#ALL RELATIONS ARE FOR MAIN SEQUENCE ONLY!

"""
  Usage:
         Make instance of class (currently only MainSequence class available
         call instance.Interpolate(instance.dict, SpT) where dict is the name of the dictionary you want to interpolate (Temperature, Radius, or Mass) and SpT is the spectral type of what you wish to interpolate to.
  
"""



class MainSequence:
  def __init__(self):
    self.Temperature = defaultdict(float)
    self.Radius = defaultdict(float)
    self.Mass = defaultdict(float)
    self.Lifetime = defaultdict(float)
    self.Temperature['O5'] = 42000
    self.Temperature['O6'] = 39500
    self.Temperature['O7'] = 35800
    self.Temperature['O8'] = 35800
    self.Temperature['B0'] = 30000
    self.Temperature['B1'] = 25400
    self.Temperature['B2'] = 20900
    self.Temperature['B3'] = 18800
    self.Temperature['B5'] = 15200
    self.Temperature['B6'] = 13700
    self.Temperature['B7'] = 12500
    self.Temperature['B8'] = 11400
    self.Temperature['B9'] = 10500
    self.Temperature['A0'] = 9800
    self.Temperature['A1'] = 9400
    self.Temperature['A2'] = 9020
    self.Temperature['A5'] = 8190
    self.Temperature['A8'] = 7600
    self.Temperature['F0'] = 7300
    self.Temperature['F2'] = 7050
    self.Temperature['F5'] = 6650
    self.Temperature['F8'] = 6250
    self.Temperature['G0'] = 5940
    self.Temperature['G2'] = 5790
    self.Temperature['G8'] = 5310
    self.Temperature['K0'] = 5150
    self.Temperature['K1'] = 4990
    self.Temperature['K3'] = 4690
    self.Temperature['K4'] = 4540
    self.Temperature['K5'] = 4410
    self.Temperature['K7'] = 4150
    self.Temperature['M0'] = 3840
    self.Temperature['M1'] = 3660
    self.Temperature['M2'] = 3520
    self.Temperature['M3'] = 3400
    self.Temperature['M4'] = 3290
    self.Temperature['M5'] = 3170
    self.Temperature['M6'] = 3030
    self.Temperature['M7'] = 2860

    self.Radius['O5'] = 13.4
    self.Radius['O6'] = 12.2
    self.Radius['O7'] = 11.0
    self.Radius['O8'] = 10.0
    self.Radius['B0'] = 6.7
    self.Radius['B1'] = 5.2
    self.Radius['B2'] = 4.1
    self.Radius['B3'] = 3.8
    self.Radius['B5'] = 3.2
    self.Radius['B6'] = 2.9
    self.Radius['B7'] = 2.7
    self.Radius['B8'] = 2.5
    self.Radius['B9'] = 2.3
    self.Radius['A0'] = 2.2
    self.Radius['A1'] = 2.1
    self.Radius['A2'] = 2.0
    self.Radius['A5'] = 1.8
    self.Radius['A8'] = 1.5
    self.Radius['F0'] = 1.4
    self.Radius['F2'] = 1.3
    self.Radius['F5'] = 1.2
    self.Radius['F8'] = 1.1
    self.Radius['G0'] = 1.06
    self.Radius['G2'] = 1.03
    self.Radius['G8'] = 0.96
    self.Radius['K0'] = 0.93
    self.Radius['K1'] = 0.92
    self.Radius['K3'] = 0.86
    self.Radius['K4'] = 0.83
    self.Radius['K5'] = 0.80
    self.Radius['K7'] = 0.74
    self.Radius['M0'] = 0.63
    self.Radius['M1'] = 0.56
    self.Radius['M2'] = 0.48
    self.Radius['M3'] = 0.41
    self.Radius['M4'] = 0.35
    self.Radius['M5'] = 0.29
    self.Radius['M6'] = 0.24
    self.Radius['M7'] = 0.20


    self.Mass['O5'] = 60
    self.Mass['O6'] = 37
    self.Mass['O8'] = 23
    self.Mass['B0'] = 17.5
    self.Mass['B5'] = 7.6
    self.Mass['B6'] = 5.9
    self.Mass['B8'] = 3.8
    self.Mass['A0'] = 2.9
    self.Mass['A5'] = 2.0
    self.Mass['F0'] = 1.6
    self.Mass['F5'] = 1.4
    self.Mass['G0'] = 1.05
    self.Mass['K0'] = 0.79
    self.Mass['K5'] = 0.67
    self.Mass['M0'] = 0.51
    self.Mass['M2'] = 0.40
    self.Mass['M5'] = 0.21


    self.Lifetime['O9.1'] = 8
    self.Lifetime['B1.1'] = 11
    self.Lifetime['B2.5'] = 16
    self.Lifetime['B4.2'] = 26
    self.Lifetime['B5.3'] = 43
    self.Lifetime['B6.7'] = 94
    self.Lifetime['B7.7'] = 165
    self.Lifetime['B9.7'] = 350
    self.Lifetime['A1.6'] = 580
    self.Lifetime['A5'] = 1100
    self.Lifetime['A8.4'] = 1800
    self.Lifetime['F2.6'] = 2700
    
  def SpT_To_Number(self, SpT):
    basenum = float(SpT[1:])
    SpectralClass = SpT[0]
    if SpectralClass == "O":
      return basenum
    elif SpectralClass == "B":
      return basenum+10
    elif SpectralClass == "A":
      return basenum+20
    elif SpectralClass == "F":
      return basenum+30
    elif SpectralClass == "G":
      return basenum+40
    elif SpectralClass == "K":
      return basenum+50
    elif SpectralClass == "M":
      return basenum+60
    else:
      print "Something weird happened! Spectral type = ", SpT
      return -1

  def Number_To_SpT(self, number):
    tens_index = 0
    num = float(number)
    while num > 0:
      num -= 10
      tens_index += 1
    tens_index = tens_index - 1
    if tens_index == 0:
      spt_class = "O"
    elif tens_index == 1:
      spt_class = "B"
    elif tens_index == 2:
      spt_class = "A"
    elif tens_index == 3:
      spt_class = "F"
    elif tens_index == 4:
      spt_class = "G"
    elif tens_index == 5:
      spt_class = "K"
    elif tens_index == 6:
      spt_class = "M"
    subclass = str(number - 10*tens_index)
    return spt_class + subclass
    
  def Interpolate(self,dictionary, SpT):
    #First, we must convert the relations above into a monotonically increasing system
    #Just add ten when we get to each new spectral type
    relation = DataStructures.xypoint(len(dictionary))
    
    xpoints = []
    ypoints = []
    for key, index in zip(dictionary, range(len(dictionary))):
      #Convert key to a number
      xpoints.append(self.SpT_To_Number(key))
      ypoints.append(dictionary[key])
      
    sorting_indices = [i[0] for i in sorted(enumerate(xpoints), key=lambda x:x[1])]
    for index in range(len(dictionary)):
      i = sorting_indices[index]
      relation.x[index] = xpoints[i]
      relation.y[index] = ypoints[i]
    
    RELATION = UnivariateSpline(relation.x, relation.y, s=0)

    spnum = self.SpT_To_Number(SpT)
    if spnum > 0:
      return RELATION(spnum)
    else:
      return spnum

  def GetSpectralType(self, dictionary, value):
    #Returns the spectral type that is closest to the value (within 0.1 subtypes)
    testgrid = numpy.arange(self.SpT_To_Number("O1"), self.SpT_To_Number("M9"), 0.1)
    besttype = "O1"
    best_difference = 9e9
    for num in testgrid:
      spt = self.Number_To_SpT(num)
      difference = numpy.abs(value - self.Interpolate(dictionary, spt))
      if difference < best_difference:
        best_difference = difference
        besttype = spt
    return besttype

########################################################
########               Pre-Main Sequence         #######
########################################################
import os
homedir = os.environ["HOME"] + "/"
tracksfile = homedir + "Dropbox/School/Research/Summer2011/EvolutionaryTracks.dat"

class PreMainSequence:
  def __init__(self, pms_tracks_file=tracksfile):
    import numpy
    import Units
    infile = open(pms_tracks_file)
    lines = infile.readlines()
    infile.close()

    #self.Mass = defaultdict(float)
    #self.Radius = defaultdict(float)
    #self.Age = defaultdict(float)
    M = []   # mass (solar masses)
    t = []   # age (Myr)
    T = []   # temperature
    R = []   # radius (solar radii)

    for line in lines:
      if not line.startswith("#"):
        if line != "\n":
          columns = line.split()
          mass = float(columns[0])
          age = 10**float(columns[1])
          temperature = 10**float(columns[3])
          g = 10**float(columns[4])
          radius = numpy.sqrt(Units.G*mass*Units.Msun/g)/Units.Rsun
          M.append(mass)
          t.append(age/1e6)
          T.append(temperature)
          R.append(radius)

    from scipy.interpolate import SmoothBivariateSpline, interp2d, griddata
    
    self.Mass = SmoothBivariateSpline(T, t, M, kx=1, ky=1, s=0)
    self.Radius = SmoothBivariateSpline(T, t, R, kx=1, ky=1, s=0)
    self.AgeFromTemperatureAndMass = SmoothBivariateSpline(T, M, t, kx=1, ky=1, s=0)
    #self.Mass = interp2d(T, t, M)
    #self.Radius = interp2d(T, t, R)
    #self.AgeFromTemperatureAndMass = interp2d(T, M, t)

    #Todo: These interpolation functions not working in simple tests
    #   (z=x^2 + y^2). Find what I am doing wrong or something that works!
    #import pylab
    
    self.MS = MainSequence()

  def Interpolate(self, SpT, yvar, value):
    #Get the temperature from the spectral type (use main sequence relations)
    temperature = self.MS.Interpolate(self.MS.Temperature, SpT)
    
    if type(value) == str:
      if "mass" in value or "Mass" in value:
        fcn = self.Mass
      elif "radius" in value or "Radius" in value:
        fcn = self.Radius
      elif "age" in value or "Age" in value:
        fcn = self.AgeFromMassAndTemperature
      elif "temperature" in value or "Temperature" in value:
        return temperature
      else:
        print "Error! Unknown value: %s" %value
        return
    else:
      fcn = value


    return fcn(temperature, yvar)
          
      
      
if __name__ == "__main__":
  sptr = MainSequence()
  pms = PreMainSequence()
  for spt in ["K9", "K5", "K0", "G5", "G0"]:
    temp = sptr.Interpolate(sptr.Temperature, spt)
    radius = sptr.Interpolate(sptr.Radius, spt)
    print "%s:  T=%g\tR=%g" %(spt, temp, radius)
    print pms.Interpolate(spt, 1000, "radius")

    
    
    
    
    
    
    
    
    
    
    
    
    
