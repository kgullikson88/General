import numpy
import sys
import subprocess
import ConvertMIPASto_lblrtm_format as convert
import scipy.interpolate
from scipy.signal import fftconvolve
import os
import DataStructures
import MakeTape5
#import Units
from astropy import constants, units
from collections import defaultdict
import lockfile
import struct
from pysynphot import observation
from pysynphot import spectrum

homedir = os.environ['HOME']
TelluricModelingDirRoot = "%s/School/Research/aerlbl_v12.2/" %homedir
NumRunDirs = 4


#Dectionary giving the number to molecule name for LBLRTM
MoleculeNumbers = {1: "H2O",
                   2: "CO2",
		   3: "O3",
		   4: "N2O",
		   5: "CO",
		   6: "CH4",
		   7: "O2",
		   8: "NO",
		   9: "SO2",
		   10: "NO2",
		   11: "NH3",
		   12: "HNO3",
		   13: "OH",
		   14: "HF",
		   15: "HCl",
		   16: "HBr",
		   17: "HI",
		   18: "ClO",
		   19: "OCS",
		   20: "H2CO",
		   21: "HOCl",
		   22: "N2",
		   23: "HCN",
		   24: "CH3Cl",
		   25: "H2O2",
		   26: "C2H2",
		   27: "C2H6",
		   28: "PH3",
		   29: "COF2",
		   30: "SF6",
		   31: "H2S",
		   32: "HCOOH",
		   33: "HO2",
		   34: "O",
		   35: "ClONO2",
		   36: "NO+",
		   37: "HOBr",
		   38: "C2H4",
		   39: "CH3OH"}
                   

"""
This is the main code to generate a telluric absorption spectrum.
The pressure, temperature, etc... can be adjusted all the way 
on the bottom of this file.
"""
def Main(pressure=795.0, temperature=283.0, lowfreq=4000, highfreq=4600, angle=45.0, humidity=50.0, co2=368.5, o3=3.9e-2, n2o=0.32, co=0.14, ch4=1.8, o2=2.1e5, no=1.1e-19, so2=1e-4, no2=1e-4, nh3=1e-4, hno3=5.6e-4, lat=30.6, alt=2.1, wavegrid=None, resolution=None, nmolecules=12, save=False, libfile=None, debug=False):

    #Determine output filename
    found = False
    for i in range(1,NumRunDirs+1):
      test = "%srundir%i/" %(TelluricModelingDirRoot, i)
      lock = lockfile.FileLock(test)
      if not lock.is_locked():
        TelluricModelingDir = test
        ModelDir = "%sOutputModels/" %TelluricModelingDir
        lock.acquire()
        found = True
        break
    if not found:
      print "Un-locked directory not found!"
      sys.exit()
    print "Telluric Modeling Directory: %s" %TelluricModelingDir
    print "Model Directory: %s" %ModelDir

    #Output filename
    model_name = ModelDir + "transmission"+"-%.2f" %pressure + "-%.2f" %temperature + "-%.1f" %humidity + "-%.1f" %angle + "-%.2f" %(co2) + "-%.2f" %(o3*100) + "-%.2f" %ch4 + "-%.2f" %(co*10)
  
    #Convert from relative humidity to concentration (ppm)
    #formulas and constants come from http://www.scribd.com/doc/53758450/Humidity-Conversion-Formulas-B210973EN-B-Lores
    Psat = 6.1162*10**(7.5892*(temperature-273.15)/(240.71+(temperature-273.15)))
    Pw = Psat*humidity/100.0
    h2o = Pw/(pressure - Pw)*1e6


    Atmosphere = defaultdict(list)
    indices = {}
    
    #Read in MIPAS_atmosphere_profile
    if debug:
      print "Generating new atmosphere profile"
    filename = TelluricModelingDir + "MIPAS_atmosphere_profile"
    infile = open(filename)
    lines = infile.readlines()
    for i in range(len(lines)):
        line = lines[i]
	if line.startswith("*") and "END" not in line:
	  if line.find("HGT") > 0 and line.find("[") > 0:
            numlevels = int(lines[i-1].split()[0])
	    indices["Z"] = i
          elif line.find("PRE") > 0 and line.find("[") > 0:
            indices["P"] = i
          elif line.find("TEM") > 0 and line.find("[") > 0:
            indices["T"] = i
	  else:
	    molecule = line.split("*")[-1].split()[0]
	    indices[molecule] = i
    infile.close()
    #Determine the number of lines that follow each header
    levelsperline = 5.0
    linespersection = int(numlevels/levelsperline + 0.9)
    

    #Fill atmosphere structure
    layers = []
    for j in range(indices["Z"]+1, indices["Z"]+1+linespersection):
      line = lines[j]
      levels = line.split()
      [layers.append(float(level)) for level in levels]
    #Pressure
    for j in range(linespersection):
      line = lines[j+indices["P"]+1]
      levels = line.split()
      for i, level in enumerate(levels):
	Atmosphere[layers[int(j*levelsperline+i)]].append(float(level))
    #Temperature
    for j in range(linespersection):
      line = lines[j+indices["T"]+1]
      levels = line.split()
      for i, level in enumerate(levels):
	Atmosphere[layers[int(j*levelsperline+i)]].append(float(level))
	Atmosphere[layers[int(j*levelsperline+i)]].append([])
    #Abundances
    for k in range(1, nmolecules+1):
      for j in range(linespersection):
	line = lines[j+indices[MoleculeNumbers[k]]+1]
	levels = line.split()
	for i, level in enumerate(levels):
	  Atmosphere[layers[int(j*levelsperline+i)]][2].append(float(level))


    #Now, scale the abundances from those at 'alt' km
    #  (Linearly interpolate)
    keys = sorted(Atmosphere.keys())
    lower = max(0, numpy.searchsorted(keys, alt)-1)
    upper = min(lower + 1, len(keys)-1)
    scale_values = list(Atmosphere[lower])
    scale_values[2] = list(Atmosphere[lower][2])
    scale_values[0] = (Atmosphere[upper][0]-Atmosphere[lower][0]) / (keys[upper]-keys[lower]) * (alt-keys[lower]) + Atmosphere[lower][0]
    scale_values[1] = (Atmosphere[upper][1]-Atmosphere[lower][1]) / (keys[upper]-keys[lower]) * (alt-keys[lower]) + Atmosphere[lower][1]
    for mol in range(len(scale_values[2])):
      scale_values[2][mol] = (Atmosphere[upper][2][mol]-Atmosphere[lower][2][mol]) / (keys[upper]-keys[lower]) * (alt-keys[lower]) + Atmosphere[lower][2][mol]

    #Do the actual scaling
    #scale_values = Atmosphere[2]
    pressure_scalefactor = pressure/scale_values[0]
    temperature_scalefactor = temperature/scale_values[1]
    for layer in layers:
      Atmosphere[layer][0] *= pressure_scalefactor
      Atmosphere[layer][1] *= temperature_scalefactor
      Atmosphere[layer][2][0] *= h2o/scale_values[2][0]
      Atmosphere[layer][2][1] *= co2/scale_values[2][1]
      Atmosphere[layer][2][2] *= o3/scale_values[2][2]
      Atmosphere[layer][2][3] *= n2o/scale_values[2][3]
      Atmosphere[layer][2][4] *= co/scale_values[2][4]
      Atmosphere[layer][2][5] *= ch4/scale_values[2][5]
      Atmosphere[layer][2][6] *= o2/scale_values[2][6]
      Atmosphere[layer][2][2] *= no/scale_values[2][6]
      Atmosphere[layer][2][2] *= so2/scale_values[2][6]
      Atmosphere[layer][2][2] *= no2/scale_values[2][6]
      Atmosphere[layer][2][2] *= nh3/scale_values[2][6]
      Atmosphere[layer][2][2] *= hno3/scale_values[2][6]


    #Now, Read in the ParameterFile and edit the necessary parameters
    parameters = MakeTape5.ReadParFile(parameterfile=TelluricModelingDir + "ParameterFile")
    parameters[48] = "%.1f" %lat
    parameters[49] = "%.1f" %alt
    parameters[51] = "%.5f" %angle
    parameters[17] = lowfreq
    freq, transmission = numpy.array([]), numpy.array([])
    if (highfreq - lowfreq > 2000.0):
      while lowfreq + 2000.0 <= highfreq:
	parameters[18] = lowfreq + 2000.00000
	
	MakeTape5.WriteTape5(parameters, output=TelluricModelingDir + "TAPE5", atmosphere=Atmosphere)

	#Run lblrtm
        cmd = "cd " + TelluricModelingDir + ";sh runlblrtm_v3.sh"
	command = subprocess.check_call(cmd, shell=True)
        freq, transmission = ReadTAPE12(TelluricModelingDir, appendto=(freq, transmission))
        lowfreq = lowfreq + 2000.00001
	parameters[17] = lowfreq

    parameters[18] = highfreq
    MakeTape5.WriteTape5(parameters, output=TelluricModelingDir + "TAPE5", atmosphere=Atmosphere)

    #Run lblrtm
    cmd = "cd " + TelluricModelingDir + ";sh runlblrtm_v3.sh"
    command = subprocess.check_call(cmd, shell=True)
    freq, transmission = ReadTAPE12(TelluricModelingDir, appendto=(freq, transmission), debug=debug)

    #Convert from frequency to wavelength units
    wavelength = units.cm.to(units.nm)/freq

    #Correct for index of refraction of air (only done approximately):
    n = 1.00026
    wavelength /= n
    
    if save:
      print "All done! Output Transmission spectrum is located in the file below:"
      print model_name
      numpy.savetxt(model_name, numpy.transpose((wavelength[::-1], transmission[::-1])), fmt="%.8g")
      if libfile != None:
        infile = open(libfile, "a")
        infile.write(model_name + "\n")
        infile.close()

    #Unlock directory
    try:
      lock.release()
    except lockfile.NotLocked:
      print "Woah, the model directory was somehow unlocked prematurely!"
      
    if wavegrid != None:
      #Interpolate model to a constant wavelength grid
      #For now, just interpolate to the right spacing
      #Also, only keep the section near the chip wavelengths
      wavelength = wavelength[::-1]
      transmission = transmission[::-1]
      xspacing = (wavelength[-1] - wavelength[0])/float(wavelength.size-1)
      tol = 10  #Go 10 nm on either side of the chip
      left = numpy.searchsorted(wavelength, wavegrid[0]-tol)
      right = numpy.searchsorted(wavelength, wavegrid[-1]+tol)
      right = min(right, wavelength.size-1)

      Model = scipy.interpolate.UnivariateSpline(wavelength, transmission, s=0)
      model = DataStructures.xypoint(right-left+1)
      model.x = numpy.arange(wavelength[left], wavelength[right], xspacing)
      model.y = Model(model.x)
      model.cont = numpy.ones(model.x.size)

      return model

    return DataStructures.xypoint(x=wavelength[::-1], y=transmission[::-1])




#Function to read the output of LBLRTM, called TAPE12
def ReadTAPE12(directory, filename="TAPE12_ex", appendto=None, debug=False):
  if not directory.endswith("/"):
    directory = directory + "/"
  infile = open("%s%s" %(directory, filename), 'rb')
  content = infile.read()
  infile.close()

  offset = 1068   #WHY?!
  size = struct.calcsize('=ddfl')
  pv1,pv2,pdv,np = struct.unpack('=ddfl', content[offset:offset+size])
  v1 = pv1
  v2 = pv2
  dv = pdv
  if debug:
    print 'info: ',pv1,pv2,pdv,np
  npts = np
  spectrum = []
  while np > 0:
    offset += size + struct.calcsize("=4f")
    size = struct.calcsize("=%if" %np)
    temp1 = struct.unpack("=%if" %np, content[offset:offset+size])
    offset += size
    temp2 = struct.unpack("=%if" %np, content[offset:offset+size])
    npts += np
    junk = [spectrum.append(temp2[i]) for i in range(np)]

    offset += size + 8  #WHERE DOES 8 COME FROM?
    size = struct.calcsize('=ddfl')
    if len(content) > offset + size:
      pv1,pv2,pdv,np = struct.unpack('=ddfl', content[offset:offset+size])
      v2 = pv2
    else:
      break

  v = numpy.arange(v1, v2, dv)
  spectrum = numpy.array(spectrum)
  if v.size < spectrum.size:
      v = numpy.r_[v, v2+dv]
  if debug:
    print "v, spec size: ", v.size, spectrum.size

  if appendto != None and appendto[0].size > 0:
    old_v, old_spectrum = appendto[0], appendto[1]
    #Check for overlap (there shouldn't be any)
    last_v = old_v[-1]
    firstindex = numpy.searchsorted(v, last_v)
    v = numpy.r_[old_v, v[firstindex:]]
    spectrum = numpy.r_[old_spectrum, spectrum[firstindex:]]
  
  return v, spectrum



"""
The following functions are useful for the actual telluric modeling.
They are not used to actually create a telluric absorption spectrum.
"""

#This function rebins (x,y) data onto the grid given by the array xgrid
#  It is designed to rebin to a courser wavelength grid, but can also
#  interpolate to a finer grid
def RebinData(data,xgrid, synphot=True):
  if synphot:
    newdata = DataStructures.xypoint(x=xgrid)
    newdata.y = rebin_spec(data.x, data.y, xgrid)
    newdata.cont = rebin_spec(data.x, data.cont, xgrid)
    newdata.y[0] = data.y[0]
    newdata.y[-1] = data.y[-1]
    return newdata
  else:
    data_spacing = data.x[1] - data.x[0]
    grid_spacing = xgrid[1] - xgrid[0]
    newdata = DataStructures.xypoint(x=xgrid)
    if grid_spacing < 2.0*data_spacing:
      Model = scipy.interpolate.UnivariateSpline(data.x, data.y, s=0)
      Continuum = scipy.interpolate.UnivariateSpline(data.x, data.cont, s=0)
      newdata.y = Model(newdata.x)
      newdata.cont = Continuum(newdata.x)

    else:
      left = numpy.searchsorted(data.x, (3*xgrid[0]-xgrid[1])/2.0)
      for i in range(xgrid.size-1):
        right = numpy.searchsorted(data.x, (xgrid[i]+xgrid[i+1])/2.0)
        newdata.y[i] = numpy.mean(data.y[left:right])
        newdata.cont[i] = numpy.mean(data.cont[left:right])
        left = right
      right = numpy.searchsorted(data.x, (3*xgrid[-1]-xgrid[-2])/2.0)
      newdata.y[xgrid.size-1] = numpy.mean(data.y[left:right])
  
    return newdata
  


def rebin_spec(wave, specin, wavnew):
  spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
  f = numpy.ones(len(wave))
  filt = spectrum.ArraySpectralElement(wave, f)
  obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
  
  return obs.binflux


#This function reduces the resolution by convolving with a gaussian kernel
def ReduceResolution(data,resolution, cont_fcn=None, extend=True):
  centralwavelength = (data.x[0] + data.x[-1])/2.0
  xspacing = data.x[1] - data.x[0]   #NOTE: this assumes constant x spacing!
  FWHM = centralwavelength/resolution;
  sigma = FWHM/(2.0*numpy.sqrt(2.0*numpy.log(2.0)))
  left = 0
  right = numpy.searchsorted(data.x, 10*sigma)
  x = numpy.arange(0,10*sigma, xspacing)
  gaussian = numpy.exp(-(x-5*sigma)**2/(2*sigma**2))
  if extend:
    #Extend array to try to remove edge effects (do so circularly)
    before = data.y[-gaussian.size/2+1:]
    after = data.y[:gaussian.size/2]
    #extended = numpy.append(numpy.append(before, data.y), after)
    extended = numpy.r_[before, data.y, after]

    first = data.x[0] - float(int(gaussian.size/2.0+0.5))*xspacing
    last = data.x[-1] + float(int(gaussian.size/2.0+0.5))*xspacing
    x2 = numpy.linspace(first, last, extended.size) 
    
    conv_mode = "valid"

  else:
    extended = data.y.copy()
    x2 = data.x.copy()
    conv_mode = "same"

  newdata = data.copy()
  if cont_fcn != None:
    cont1 = cont_fcn(newdata.x)
    cont2 = cont_fcn(x2)
    cont1[cont1 < 0.01] = 1
  
    newdata.y = fftconvolve(extended*cont2, gaussian/gaussian.sum(), mode=conv_mode)/cont1

  else:
    newdata.y = fftconvolve(extended, gaussian/gaussian.sum(), mode=conv_mode)
    
  return newdata

#Just a convenince fcn which combines the above two
def ReduceResolutionAndRebinData(data,resolution,xgrid):
  data = ReduceResolution(data,resolution)
  return RebinData(data,xgrid)
  
  


if __name__ == "__main__":
  pressure = 796.22906
  temperature = 270.40
  humidity = 10.0
  angle = 40.8
  co2 = 368.5
  o3 = 0.039
  ch4 = 4.0
  co = 0.15
  o2 = 2.2e5
  #o2 = 4.4e5

  lowwave = 445
  highwave = 446
  lowwave = 700
  highwave = 800
  lowfreq = 1e7/highwave
  highfreq = 1e7/lowwave
  Main(pressure=pressure, temperature=temperature, humidity=humidity, lowfreq=lowfreq, highfreq=highfreq, angle=angle, o2=o2, alt=2.1, save=True)
          

