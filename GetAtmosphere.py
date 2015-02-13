"""
  This file provides some functions to generate a profile from
  a timestamp and list of files with appropriate filenames
"""

from astropy.time import Time
import numpy as np
import MakeModel
from scipy.interpolate import InterpolatedUnivariateSpline as spline


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

    # Get julian date of the observation
    obstime = Time("%s %s" % (datestr, timestr), format='iso', scale='utc').jd

    #Get the julian dates for each file from the filename
    fileDict = {}
    for fname in filenames:
        datestr = fname.split("_")[2]
        hour = fname.split("_")[3].split(".")[0]
        t = Time("%s %s:00:00" % (datestr, hour), format='iso', scale='utc').jd
        fileDict[t] = fname

    #Sort the times by the distance from the observation time
    times = sorted(fileDict.keys(), key=lambda t: (t - obstime) ** 2)

    goodtimes = times[:2]
    if goodtimes[0] < goodtimes[1]:
        first = goodtimes[0]
        last = goodtimes[1]
    else:
        first = goodtimes[1]
        last = goodtimes[0]

    return fileDict[first], fileDict[last], first, last, obstime


def InterpolateAtmosphere(firstfile, lastfile, t1, t2, t):
    """
      This takes the output of GetAtmosphereFiles, reads in the files,
      and interpolates the values as a function of pressure

      Returns: the pressure, temperature, and dew point at the obstime
    """
    Pres1, height1, Temp1, dew1 = np.loadtxt(firstfile, usecols=(0, 1, 2, 3), unpack=True)
    Pres2, height2, Temp2, dew2 = np.loadtxt(lastfile, usecols=(0, 1, 2, 3), unpack=True)

    # Interpolate
    P1 = spline(height1, Pres1)
    T1 = spline(height1, Temp1)
    D1 = spline(height1, dew1)
    P2 = spline(height2, Pres2)
    T2 = spline(height2, Temp2)
    D2 = spline(height2, dew2)

    # Make a new height grid
    firstval = max(height1[0], height2[0])
    lastval = min(height1[-1], height2[-1])
    Z = np.logspace(np.log10(firstval), np.log10(lastval), height1.size)
    P = (P1(Z) - P2(Z)) / (t1 - t2) * (t - t1) + P1(Z)
    T = (T1(Z) - T2(Z)) / (t1 - t2) * (t - t1) + T1(Z)
    D = (D1(Z) - D2(Z)) / (t1 - t2) * (t - t1) + D1(Z)
    return P, Z, T, D


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
    # Get the interpolated values
    P, Z, T, D = InterpolateAtmosphere(*GetAtmosphereFiles(filenames, datestr, timestr))

    #Sort by height
    sorter = np.argsort(Z)
    Z = Z[sorter]
    P = P[sorter]
    T = T[sorter]
    D = D[sorter]

    #Convert dew point temperature to ppmv
    Pw = np.zeros(D.size)
    for i, dewpoint in enumerate(D):
        Pw[i] = MakeModel.VaporPressure(dewpoint + 273.15)
    h2o = Pw / (P - Pw) * 1e6

    #Convert height to km, and temperature to kelvin
    Z /= 1000.0
    T += 273.15
    return Z, P, T, h2o


