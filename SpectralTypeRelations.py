from collections import defaultdict
from scipy.interpolate import UnivariateSpline
import numpy
import DataStructures
from astropy import constants

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
    self.BC = defaultdict(float)
    self.BmV = defaultdict(float)   #B-V color
    self.UmV = defaultdict(float)   #U-V color
    self.VmJ = defaultdict(float)   #V-J color
    self.VmH = defaultdict(float)   #V-H color
    self.VmK = defaultdict(float)   #V-K color
    self.AbsMag = defaultdict(float)  #Absolute Magnitude in V band
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

    #From Kenyon & Hartmann 1995
    self.BmV['B0'] = -0.3
    self.BmV['B1'] = -0.26
    self.BmV['B2'] = -0.22
    self.BmV['B3'] = -0.19
    self.BmV['B4'] = -0.16
    self.BmV['B5'] = -0.14
    self.BmV['B6'] = -0.13
    self.BmV['B7'] = -0.11
    self.BmV['B8'] = -0.09
    self.BmV['B9'] = -0.06
    self.BmV['A0'] = 0.0
    self.BmV['A1'] = 0.03
    self.BmV['A2'] = 0.06
    self.BmV['A3'] = 0.09
    self.BmV['A4'] = 0.12
    self.BmV['A5'] = 0.14
    self.BmV['A6'] = 0.16
    self.BmV['A7'] = 0.19
    self.BmV['A8'] = 0.23
    self.BmV['A9'] = 0.27
    self.BmV['F0'] = 0.31
    self.BmV['F1'] = 0.33
    self.BmV['F2'] = 0.35
    self.BmV['F3'] = 0.37
    self.BmV['F4'] = 0.39
    self.BmV['F5'] = 0.42
    self.BmV['F6'] = 0.46
    self.BmV['F7'] = 0.50
    self.BmV['F8'] = 0.52
    self.BmV['F9'] = 0.55
    self.BmV['G0'] = 0.58
    self.BmV['G1'] = 0.60
    self.BmV['G2'] = 0.62
    self.BmV['G3'] = 0.63
    self.BmV['G4'] = 0.64
    self.BmV['G5'] = 0.66
    self.BmV['G6'] = 0.68
    self.BmV['G7'] = 0.71
    self.BmV['G8'] = 0.73
    self.BmV['G9'] = 0.78
    self.BmV['K0'] = 0.82
    self.BmV['K1'] = 0.85
    self.BmV['K2'] = 0.89
    self.BmV['K3'] = 0.97
    self.BmV['K4'] = 1.07
    self.BmV['K5'] = 1.16
    self.BmV['K6'] = 1.27
    self.BmV['K7'] = 1.38
    self.BmV['M0'] = 1.41
    self.BmV['M1'] = 1.8
    self.BmV['M2'] = 1.52
    self.BmV['M3'] = 1.55
    self.BmV['M4'] = 1.60
    self.BmV['M5'] = 1.82
    self.BmV['M6'] = 2.00

    #From Kenyon & Hartmann 1995
    self.UmV['B0'] = -1.38
    self.UmV['B1'] = -1.23
    self.UmV['B2'] = -1.08
    self.UmV['B3'] = -0.94
    self.UmV['B4'] = -0.84
    self.UmV['B5'] = -0.72
    self.UmV['B6'] = -0.61
    self.UmV['B7'] = -0.50
    self.UmV['B8'] = -0.40
    self.UmV['B9'] = -0.17
    self.UmV['A0'] = 0.00
    self.UmV['A1'] = 0.05
    self.UmV['A2'] = 0.10
    self.UmV['A3'] = 0.15
    self.UmV['A4'] = 0.20
    self.UmV['A5'] = 0.24
    self.UmV['A6'] = 0.26
    self.UmV['A7'] = 0.28
    self.UmV['A8'] = 0.30
    self.UmV['A9'] = 0.32
    self.UmV['F0'] = 0.34
    self.UmV['F1'] = 0.36
    self.UmV['F2'] = 0.37
    self.UmV['F3'] = 0.38
    self.UmV['F4'] = 0.39
    self.UmV['F5'] = 0.40
    self.UmV['F6'] = 0.44
    self.UmV['F7'] = 0.49
    self.UmV['F8'] = 0.55
    self.UmV['F9'] = 0.59
    self.UmV['G0'] = 0.64
    self.UmV['G1'] = 0.64
    self.UmV['G2'] = 0.64
    self.UmV['G3'] = 0.71
    self.UmV['G4'] = 0.79
    self.UmV['G5'] = 0.86
    self.UmV['G6'] = 0.90
    self.UmV['G7'] = 0.95
    self.UmV['G8'] = 1.00
    self.UmV['G9'] = 1.13
    self.UmV['K0'] = 1.27
    self.UmV['K1'] = 1.44
    self.UmV['K2'] = 1.52
    self.UmV['K3'] = 1.80
    self.UmV['K4'] = 2.01
    self.UmV['K5'] = 2.22
    self.UmV['K6'] = 2.43
    self.UmV['K7'] = 2.64
    self.UmV['M0'] = 2.66
    self.UmV['M1'] = 2.74
    self.UmV['M2'] = 2.69
    self.UmV['M3'] = 2.65
    self.UmV['M4'] = 2.89
    self.UmV['M5'] = 3.07
    self.UmV['M6'] = 3.33

    #From Kenyon & Hartmann 1995
    self.VmJ['B0'] = -0.70
    self.VmJ['B1'] = -0.61
    self.VmJ['B2'] = -0.55
    self.VmJ['B3'] = -0.45
    self.VmJ['B4'] = -0.40
    self.VmJ['B5'] = -0.35
    self.VmJ['B6'] = -0.32
    self.VmJ['B7'] = -0.29
    self.VmJ['B8'] = -0.26
    self.VmJ['B9'] = -0.14
    self.VmJ['A0'] = 0.00
    self.VmJ['A1'] = 0.06
    self.VmJ['A2'] = 0.12
    self.VmJ['A3'] = 0.18
    self.VmJ['A4'] = 0.25
    self.VmJ['A5'] = 0.30
    self.VmJ['A6'] = 0.34
    self.VmJ['A7'] = 0.39
    self.VmJ['A8'] = 0.45
    self.VmJ['A9'] = 0.50
    self.VmJ['F0'] = 0.54
    self.VmJ['F1'] = 0.58
    self.VmJ['F2'] = 0.63
    self.VmJ['F3'] = 0.69
    self.VmJ['F4'] = 0.76
    self.VmJ['F5'] = 0.83
    self.VmJ['F6'] = 0.87
    self.VmJ['F7'] = 0.98
    self.VmJ['F8'] = 1.00
    self.VmJ['F9'] = 1.03
    self.VmJ['G0'] = 1.05
    self.VmJ['G1'] = 1.08
    self.VmJ['G2'] = 1.09
    self.VmJ['G3'] = 1.11
    self.VmJ['G4'] = 1.15
    self.VmJ['G5'] = 1.16
    self.VmJ['G6'] = 1.18
    self.VmJ['G7'] = 1.27
    self.VmJ['G8'] = 1.28
    self.VmJ['G9'] = 1.30
    self.VmJ['K0'] = 1.43
    self.VmJ['K1'] = 1.53
    self.VmJ['K2'] = 1.63
    self.VmJ['K3'] = 1.79
    self.VmJ['K4'] = 1.95
    self.VmJ['K5'] = 2.13
    self.VmJ['K6'] = 2.25
    self.VmJ['K7'] = 2.37
    self.VmJ['M0'] = 2.79
    self.VmJ['M1'] = 3.00
    self.VmJ['M2'] = 3.24
    self.VmJ['M3'] = 3.78
    self.VmJ['M4'] = 4.38
    self.VmJ['M5'] = 5.18
    self.VmJ['M6'] = 6.27

    #From Kenyon & Hartmann 1995
    self.VmH['B0'] = -0.81
    self.VmH['B1'] = -0.71
    self.VmH['B2'] = -0.65
    self.VmH['B3'] = -0.53
    self.VmH['B4'] = -0.47
    self.VmH['B5'] = -0.41
    self.VmH['B6'] = -0.37
    self.VmH['B7'] = -0.34
    self.VmH['B8'] = -0.31
    self.VmH['B9'] = -0.16
    self.VmH['A0'] = 0.00
    self.VmH['A1'] = 0.06
    self.VmH['A2'] = 0.13
    self.VmH['A3'] = 0.21
    self.VmH['A4'] = 0.28
    self.VmH['A5'] = 0.36
    self.VmH['A6'] = 0.41
    self.VmH['A7'] = 0.47
    self.VmH['A8'] = 0.54
    self.VmH['A9'] = 0.61
    self.VmH['F0'] = 0.47
    self.VmH['F1'] = 0.73
    self.VmH['F2'] = 0.79
    self.VmH['F3'] = 0.87
    self.VmH['F4'] = 0.97
    self.VmH['F5'] = 1.06
    self.VmH['F6'] = 1.17
    self.VmH['F7'] = 1.27
    self.VmH['F8'] = 1.30
    self.VmH['F9'] = 1.33
    self.VmH['G0'] = 1.36
    self.VmH['G1'] = 1.39
    self.VmH['G2'] = 1.41
    self.VmH['G3'] = 1.44
    self.VmH['G4'] = 1.47
    self.VmH['G5'] = 1.52
    self.VmH['G6'] = 1.58
    self.VmH['G7'] = 1.66
    self.VmH['G8'] = 1.69
    self.VmH['G9'] = 1.73
    self.VmH['K0'] = 1.88
    self.VmH['K1'] = 2.00
    self.VmH['K2'] = 2.13
    self.VmH['K3'] = 2.33
    self.VmH['K4'] = 2.53
    self.VmH['K5'] = 2.74
    self.VmH['K6'] = 2.88
    self.VmH['K7'] = 3.03
    self.VmH['M0'] = 3.48
    self.VmH['M1'] = 3.67
    self.VmH['M2'] = 3.91
    self.VmH['M3'] = 4.40
    self.VmH['M4'] = 4.98
    self.VmH['M5'] = 5.80
    self.VmH['M6'] = 6.93

    #From Kenyon & Hartmann 1995
    self.VmK['B0'] = -0.93
    self.VmK['B1'] = -0.81
    self.VmK['B2'] = -0.74
    self.VmK['B3'] = -0.61
    self.VmK['B4'] = -0.55
    self.VmK['B5'] = -0.57
    self.VmK['B6'] = -0.43
    self.VmK['B7'] = -0.39
    self.VmK['B8'] = -0.35
    self.VmK['B9'] = -0.18
    self.VmK['A0'] = 0.00
    self.VmK['A1'] = 0.07
    self.VmK['A2'] = 0.14
    self.VmK['A3'] = 0.22
    self.VmK['A4'] = 0.30
    self.VmK['A5'] = 0.8
    self.VmK['A6'] = 0.44
    self.VmK['A7'] = 0.50
    self.VmK['A8'] = 0.57
    self.VmK['A9'] = 0.64
    self.VmK['F0'] = 0.70
    self.VmK['F1'] = 0.76
    self.VmK['F2'] = 0.82
    self.VmK['F3'] = 0.91
    self.VmK['F4'] = 1.01
    self.VmK['F5'] = 1.10
    self.VmK['F6'] = 1.21
    self.VmK['F7'] = 1.32
    self.VmK['F8'] = 1.35
    self.VmK['F9'] = 1.38
    self.VmK['G0'] = 1.41
    self.VmK['G1'] = 1.44
    self.VmK['G2'] = 1.46
    self.VmK['G3'] = 1.49
    self.VmK['G4'] = 1.53
    self.VmK['G5'] = 1.58
    self.VmK['G6'] = 1.64
    self.VmK['G7'] = 1.72
    self.VmK['G8'] = 1.76
    self.VmK['G9'] = 1.80
    self.VmK['K0'] = 1.96
    self.VmK['K1'] = 2.09
    self.VmK['K2'] = 2.22
    self.VmK['K3'] = 2.42
    self.VmK['K4'] = 2.63
    self.VmK['K5'] = 2.85
    self.VmK['K6'] = 3.00
    self.VmK['K7'] = 3.16
    self.VmK['M0'] = 3.65
    self.VmK['M1'] = 3.87
    self.VmK['M2'] = 4.11
    self.VmK['M3'] = 4.65
    self.VmK['M4'] = 5.26
    self.VmK['M5'] = 6.12
    self.VmK['M6'] = 7.30

    #From Allen's Astrophysical Quantities and Binney & Merrifield (marked with 'BM')
    self.AbsMag['O5'] = -5.7
    self.AbsMag['O8'] = -4.9    #BM
    self.AbsMag['O9'] = -4.5
    self.AbsMag['B0'] = -4.0
    self.AbsMag['B2'] = -2.45
    self.AbsMag['B3'] = -1.6    #BM
    self.AbsMag['B5'] = -1.2
    self.AbsMag['B8'] = -0.25
    self.AbsMag['A0'] = 0.65
    self.AbsMag['A2'] = 1.3
    self.AbsMag['A5'] = 1.95
    self.AbsMag['F0'] = 2.7
    self.AbsMag['F2'] = 3.6
    self.AbsMag['F5'] = 3.5
    self.AbsMag['F8'] = 4.0
    self.AbsMag['G0'] = 4.4
    self.AbsMag['G2'] = 4.7
    self.AbsMag['G5'] = 5.1
    self.AbsMag['G8'] = 5.5
    self.AbsMag['K0'] = 5.9
    self.AbsMag['K2'] = 6.4
    self.AbsMag['K5'] = 7.35
    self.AbsMag['M0'] = 8.8
    self.AbsMag['M2'] = 9.9
    self.AbsMag['M5'] = 12.3
    
    
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

  def GetAbsoluteMagnitude(self, spt, color):
    Vmag = self.Interpolate(self.AbsMag, spt)
    if color == "V":
      return Vmag
    else:
      if color == "U" or color == "B":
        string = "color_diff = self.Interpolate(self.%smV, spt)" %color
        exec(string)
        return color_diff + Vmag
      elif color == "J" or color == "H" or color == "K":
        string = "color_diff = self.Interpolate(self.Vm%s, spt)" %color
        exec(string)
        return Vmag - color_diff
      else:
        raise ValueError("Color %s not known!" %color)

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

    G = constants.G.cgs.value
    M_sun = constants.M_sun.cgs.value
    R_sun = constants.R_sun.cgs.value

    for line in lines:
      if not line.startswith("#"):
        if line != "\n":
          columns = line.split()
          mass = float(columns[0])
          age = 10**float(columns[1])
          temperature = 10**float(columns[3])
          g = 10**float(columns[4])
          radius = numpy.sqrt(G * mass*M_sun / g) / R_sun
          #radius = numpy.sqrt(Units.G*mass*Units.Msun/g)/Units.Rsun
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

    
    
    
    
    
    
    
    
    
    
    
    
    
