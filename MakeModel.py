import numpy
import sys
import subprocess
import ConvertMIPASto_lblrtm_format as convert
import scipy.interpolate
from scipy.signal import fftconvolve
import os
import DataStructures
import MakeTape5
import Units
from collections import defaultdict
import lockfile

homedir = os.environ['HOME']
#TelluricModelingDir = "%s/School/Research/aerlbl_v12.2/rundir2/" %homedir
TelluricModelingDirRoot = "%s/School/Research/aerlbl_v12.2/" %homedir
#ModelDir = "%sOutputModels/" %TelluricModelingDir
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
def Main(pressure=795.0, temperature=283.0, lowfreq=4000, highfreq=4600, angle=45.0, humidity=50.0, co2=268.5, o3=3.9e-2, n2o=0.32, co=0.14, ch4=1.8, o2=2.1e5, no=1.1e-19, so2=1e-4, no2=1e-4, nh3=1e-4, hno3=5.6e-4, wavegrid=None, resolution=None, nmolecules=12, save=False):

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

    #Now, scale the abundances from those at 2 km
    scale_values = Atmosphere[2]
    pressure_scalefactor = pressure/scale_values[0]
    temperature_scalefactor = temperature/scale_values[1]
    h2o_scalefactor = h2o/scale_values[2][0]
    co2_scalefactor = co2/scale_values[2][1]
    o3_scalefactor = o3/scale_values[2][2]
    co_scalefactor = co/scale_values[2][4]
    ch4_scalefactor = ch4/scale_values[2][5]
    o2_scalefactor = o2/scale_values[2][6]
    for layer in layers:
      Atmosphere[layer][0] *= pressure_scalefactor
      Atmosphere[layer][1] *= temperature_scalefactor
      Atmosphere[layer][2][0] *= h2o_scalefactor
      Atmosphere[layer][2][1] *= co2_scalefactor
      Atmosphere[layer][2][2] *= o3_scalefactor
      Atmosphere[layer][2][4] *= co_scalefactor
      Atmosphere[layer][2][5] *= ch4_scalefactor
      Atmosphere[layer][2][6] *= o2_scalefactor

    #Now, Read in the ParameterFile and edit the necessary parameters
    parameters = MakeTape5.ReadParFile(parameterfile=TelluricModelingDir + "ParameterFile")
    parameters[51] = "%.5f" %angle
    parameters[17] = lowfreq
    if (highfreq - lowfreq > 2000.0):
      while lowfreq + 2000.0 <= highfreq:
	parameters[18] = lowfreq + 2000.0
	
	MakeTape5.WriteTape5(parameters, output=TelluricModelingDir + "TAPE5", atmosphere=Atmosphere)

	#Run lblrtm
        cmd = "cd " + TelluricModelingDir + ";sh runlblrtm_v2.sh"
	command = subprocess.check_call(cmd, shell=True)
        lowfreq = lowfreq + 2000.0
	parameters[17] = lowfreq

    else:
      parameters[18] = highfreq
      MakeTape5.WriteTape5(parameters, output=TelluricModelingDir + "TAPE5", atmosphere=Atmosphere)

      #RUn lblrtm
      cmd = "cd " + TelluricModelingDir + ";sh runlblrtm_v2.sh"
      command = subprocess.check_call(cmd, shell=True)


    #Convert from frequency to wavelength units
    #freq2wave.Fix("FullSpectrum.freq")
    wavelength, transmission = FixTelluric(TelluricModelingDir + "FullSpectrum.freq", TelluricModelingDir)

    #Correct for index of refraction of air:
    """
    #Using Equation 32 from Owens paper (Saved in Dropbox/School/Research/opticsPaper02
    pressure *= Units.torr/Units.hPa
    temperature -= 273.15
    wavenumber = 1.0 / (wavelength*Units.cm/Units.nm)
    nminus1 = 1e-8 * (8342.13 + 2406030./(130. - 1.0/wavenumber**2) + 15997./(38.9 - 1.0/wavenumber**2)) * pressure/720.775 * (1 + pressure*(0.817 - 0.133*(temperature))*1e-6)/(1+0.0036610*(temperature))
    n = nminus1 + 1
    """
    n = 1.0003
    wavelength /= n
    
    if "FullSpectrum.freq" in os.listdir(TelluricModelingDir):
      cmd = "rm " + TelluricModelingDir + "FullSpectrum.freq"
      command = subprocess.check_call(cmd, shell=True)
    
    if save:
      print "All done! Output Transmission spectrum is located in the file below:"
      print model_name    
      numpy.savetxt(model_name, numpy.transpose((wavelength, transmission)), fmt="%.8g")

    #Unlock directory
    lock.release()
      
    if wavegrid != None:
      #Interpolate model to a constant wavelength grid
      #For now, just interpolate to the right spacing
      #Also, only keep the section near the chip wavelengths
      wavelength = wavelength[::-1]
      transmission = transmission[::-1]
      xspacing = (wavelength[-1] - wavelength[0])/float(wavelength.size)
      tol = 10  #Go 10 nm on either side of the chip
      left = numpy.searchsorted(wavelength, wavegrid[0]-tol)
      right = numpy.searchsorted(wavelength, wavegrid[-1]+tol)
      right = min(right, wavelength.size-1)

      Model = scipy.interpolate.UnivariateSpline(wavelength, transmission, s=0)
      model = DataStructures.xypoint(right-left+1)
      model.x = numpy.arange(wavelength[left], wavelength[right], xspacing)
      #model.x = numpy.copy(wavegrid)
      model.y = Model(model.x)
      model.cont = numpy.ones(model.x.size)

      return model

    return wavelength, transmission


def FixTelluric(filename, TelluricModelingDir):
  wavenumber, transmission = numpy.loadtxt(filename,unpack=True)
  wavelength = 1e4/wavenumber
  outfile = open(TelluricModelingDir + "FullSpectrum.wave", "w")
  for i in range(wavelength.size):
    outfile.write(str(wavelength[i]) + "\t" + str(transmission[i]) + "\n")
  outfile.close()
  return wavelength*1000.0, transmission


"""
The following functions are useful for the actual telluric modeling.
They are not used to actually create a telluric absorption spectrum.
"""

#This function rebins (x,y) data onto the grid given by the array xgrid
def RebinData(data,xgrid):
  Model = scipy.interpolate.UnivariateSpline(data.x, data.y, s=0)
  Continuum = scipy.interpolate.UnivariateSpline(data.x, data.cont, s=0)
  newdata = DataStructures.xypoint(xgrid.size)
  newdata.x = numpy.copy(xgrid)
  newdata.y = Model(newdata.x)
  newdata.cont = Continuum(newdata.x)

  left = numpy.searchsorted(data.x, (3*xgrid[0]-xgrid[1])/2.0)
  for i in range(xgrid.size-1):
    right = numpy.searchsorted(data.x, (xgrid[i]+xgrid[i+1])/2.0)
    newdata.y[i] = numpy.mean(data.y[left:right])
    left = right
  right = numpy.searchsorted(data.x, (3*xgrid[-1]-xgrid[-2])/2.0)
  newdata.y[xgrid.size-1] = numpy.mean(data.y[left:right])
  
  return newdata

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
    extended = numpy.append(numpy.append(before, data.y), after)

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
  
    #newdata.y = numpy.convolve(extended*cont2, gaussian/gaussian.sum(), mode=conv_mode)/cont1
    newdata.y = fftconvolve(extended*cont2, gaussian/gaussian.sum(), mode=conv_mode)/cont1

  else:
    #newdata.y = numpy.convolve(extended, gaussian/gaussian.sum(), mode=conv_mode)
    newdata.y = fftconvolve(extended, gaussian/gaussian.sum(), mode=conv_mode)
    
  return newdata

#Just a convenince fcn which combines the above two
def ReduceResolutionAndRebinData(data,resolution,xgrid):
  data = ReduceResolution(data,resolution)
  return RebinData(data,xgrid)
  
  


if __name__ == "__main__":
  pressure = 795.79
  temperature = 290.93
  humidity = 90.5155
  angle = 7.37
  lowfreq = 4000
  highfreq = 5000
  co2 = 368.5
  o3 = 0.04
  ch4 = 10.0
  co = 0.00
  o2 = 2.4e5

  lowwave = 700.0
  highwave = 800.0
  lowfreq = 1e7/highwave
  highfreq = 1e7/lowwave
  Main(pressure=pressure, temperature=temperature, humidity=humidity, lowfreq=lowfreq, highfreq=highfreq, angle=angle, co2=co2, o3=o3, ch4=ch4, co=co, o2=o2, save=True)
          

