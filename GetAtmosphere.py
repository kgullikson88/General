"""
  This file provides some functions to generate a profile from
  a timestamp and list of files with appropriate filenames
"""

from astropy.time import Time
import numpy
import matplotlib.pyplot as plt

def GetAtmosphereFiles(filenames, datestr, timestr):
  """
    This finds the two best atmosphere files, given a date and time of
    observation. Generally:
      datestr = header['DATE-OBS']
          and
      timestr = header['UT']

    The list of filenames MUST have the following naming convention:
      GDAS_atmosphere_YYYY-MM-DD_HH.dat

    Returns the two files bracketing the observation date/time,
      the julian date of the observation, and the julian dates
      of the atmosphere profiles in the bracketing files
  """

  #Get julian date of the observation
  obstime = Time("%s %s" %(datestr, timestr), format='iso', scale='utc').jd

  #Get the julian dates for each file from the filename
  fileDict = {}
  for fname in filenames:
    datestr = fname.split("_")[2]
    hour = fname.split("_")[3].split(".")[0]
    t = Time("%s %s:00:00" %(datestr, hour), format='iso', scale='utc').jd
    fileDict[t] = fname

  #Sort the times by the distance from the observation time
  times = sorted(fileDict.keys(), key=lambda t: (t-obstime)**2)

  goodtimes = times[:2]
  if goodtimes[0] < goodtimes[1]:
    first = goodtimes[0]
    last = goodtimes[1]
  else:
    first = goodtimes[1]
    last = goodtimes[2]

  return fileDict[first], fileDict[last], first, last, obstime


def InterpolateAtmosphere(firstfile, lastfile, t1, t2, t):
  """
    This takes the output of GetAtmosphereFiles, reads in the files,
    and interpolates the values as a function of pressure

    Returns: the pressure, temperature, and dew point at the obstime
  """
  Pres1,height1,Temp1,dew1 = numpy.loadtxt(firstfile, usecols=(0,1,2,3), unpack=True)
  Pres2,height2,Temp2,dew2 = numpy.loadtxt(lastfile, usecols=(0,1,2,3), unpack=True)
  

  #Sometimes, the first pressure will be different (because it hits the ground level)
  if abs(Pres1[0] - Pres2[0]) > 1e-3:
    Pres = Pres1[1:]
    height1 = height1[1:]
    height2 = height2[1:]
    Temp1 = Temp1[1:]
    Temp2 = Temp2[1:]
    dew1 = dew1[1:]
    dew2 = dew2[1:]
  
  #Set up output arrays
  Pres = Pres1
  height = height1
  Temp = Temp1
  dew = dew1

  #Now, interpolate the temperature, height, and dewpoint at each pressure
  for i, P in enumerate(Pres):
    height[i] = (height1[i] - height2[i])/(t1 - t2) * (t - t1) + height1[i]
    Temp[i] = (Temp1[i] - Temp2[i])/(t1-t2) * (t - t1) + Temp1[i]
    dew[i] = (dew1[i] - dew2[i])/(t1-t2) * (t-t1) + dew1[i]
  return Pres, height, Temp, dew


def GetProfile(filenames, datestr, timestr):
  """
    This function will call GetAtmosphereFiles to find the
      GDAS profiles bracketing the observation time given
      by datestr and timestr. Generally:

        datestr = header['DATE-OBS']
          and
        timestr = header['UT']
    Returns: pressure, height, temperature, and relative humidity
             at several layers, interpolated to the observation time
  """
  #Get the interpolated values
  P, Z, T, D = InterpolateAtmosphere(*GetAtmosphereFiles(filenames, datestr, timestr))
  
  #Sort by height
  sorter = numpy.argsort(Z)
  Z = Z[sorter]
  P = P[sorter]
  T = T[sorter]
  D = D[sorter]

  #Convert dew point temperature to ppmv
  #formulas and constants come from http://www.vaisala.com/Vaisala%20Documents/Application%20notes/Humidity_Conversion_Formulas_B210973EN-F.pdf
  Pw = 6.116441 * 10**(7.591386*D/(D + 240.7263))
  h2o = Pw / (P-Pw) * 1e6

  #Convert height to km, and temperature to kelvin
  Z /= 1000.0
  T += 273.15
  return Z, P, T, h2o


