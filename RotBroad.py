import pylab
import numpy
import Units
from scipy.interpolate import UnivariateSpline
import scipy.signal
import sys
import DataStructures
import SpectralTypeRelations
import FindContinuum


pi = numpy.pi


def CombineIntervals(intervals, overlap=0):
  
  iteration = 0
  for interval in intervals:
    lastindex = interval.x.size - overlap
    
    if iteration == 0:
      firstindex = 0
      master_x = interval.x[firstindex:lastindex]
      master_y = interval.y[firstindex:lastindex]
      master_cont = interval.cont[firstindex:lastindex]
    else:
      firstindex = numpy.searchsorted(interval.x, master_x[-1])+1
      master_x = numpy.append(master_x, interval.x[firstindex:lastindex])
      master_y = numpy.append(master_y, interval.y[firstindex:lastindex])
      master_cont = numpy.append(master_cont, interval.cont[firstindex:lastindex])
    iteration += 1

  output = DataStructures.xypoint(master_x.size)
  output.x = master_x.copy()
  output.y = master_y.copy()
  output.cont = master_cont.copy()

  #Scale continuum so the highest normalized flux = 1.0
  maxindex = numpy.argmax(output.y/output.cont)
  factor = output.y[maxindex]/output.cont[maxindex]
  output.cont *= factor

  return output


def ReadFile(filename):
  #Read in file
  x,y = numpy.loadtxt(filename, unpack=True)
  model = DataStructures.xypoint(x.size)
  model.x = x.copy()*Units.nm/Units.angstrom
  model.y = y.copy()
  #Read in continuum
  x,y = numpy.loadtxt(filename[:-1]+"17", unpack=True)
  cont_fcn = UnivariateSpline(x*Units.nm/Units.angstrom, y, s=0)
  model.cont = cont_fcn(model.x)

  return model


def Broaden(model, vsini, intervalsize=50.0, alpha=0.5, linear=False, findcont=False):  
  """
    model:           input filename of the spectrum. The continuum data is assumed to be in filename[:-1]+".17"
                     model can also be a DataStructures.xypoint containing the already-read model (must include continuum!)
    vsini:           the velocity (times sin(i) ) of the star
    intervalsize:    The size (in nm) of the interval to use for broadening. Since it depends on wavelength, you don't want to do all at once
    alpha:           Something to do with limb darkening...
    linear:          flag for if the x-spacing is already linear. If true, we don't need to make UnivariateSplines and linearize
    findcont:        flag to decide if the continuum needs to be found
  """

  if type(model) == str:
    model = ReadFile(model)

  if not linear:
    model_fcn = UnivariateSpline(model.x, model.y, s=0)
    if not findcont:
      cont_fcn = UnivariateSpline(model.x, model.cont, s=0)

  #Will convolve with broadening profile in steps, to keep accuracy
  #interval size is set as a keyword argument
  profilesize = -1
  firstindex = 0
  lastindex = 0
  intervals = []

  while lastindex < model.x.size - 1:
    lastindex = numpy.searchsorted(model.x, model.x[firstindex] + intervalsize) - 1
    interval = DataStructures.xypoint(lastindex - firstindex + 1)
    if linear:
      interval.x = model.x[firstindex:lastindex]
      interval.y = model.y[firstindex:lastindex]
      if not findcont:
        interval.cont = model.cont[firstindex:lastindex]
    else:
      interval.x = numpy.linspace(model.x[firstindex], model.x[lastindex], lastindex - firstindex + 1)
      interval.y = model_fcn(interval.x)
      if not findcont:
        interval.cont = cont_fcn(interval.x)

    if findcont:
      interval.cont = FindContinuum.Continuum(interval.x, interval.y)
    
    
    #Make broadening profile
    beta = alpha/(1-alpha)
    wave0 = interval.x[interval.x.size/2]
    zeta = wave0*vsini/Units.c
    xspacing = interval.x[1] - interval.x[0]
    wave = numpy.arange(wave0 - zeta, wave0 + zeta, xspacing)
    x = numpy.linspace(-1.0, 1.0, wave.size)
    flux = pi/2.0*(1.0 - 1.0/(1. + 2*beta/3.)*(2/pi*numpy.sqrt(1.-x**2) + beta/2*(1.-x**2)))
    profile = -(flux - flux.max())

    #Extend interval to reduce edge effects (basically turn convolve into circular convolution)
    before = interval.y[-profile.size/2+1:]
    after = interval.y[:profile.size/2]
    extended = numpy.append(numpy.append(before, interval.y), after)

    
    interval.y = scipy.signal.fftconvolve(extended, profile/profile.sum(), mode="valid")
    intervals.append(interval)

    if profile.size > profilesize:
      profilesize = profile.size
    firstindex = lastindex - 2*profile.size

  if len(intervals) > 1:
    return CombineIntervals(intervals, overlap=profilesize)
  else:
    return intervals[0]
  

def Test_fcn(model):

  model_fcn = UnivariateSpline(model.x, model.y, s=0)
  cont_fcn = UnivariateSpline(model.x, model.cont, s=0)

  print model_fcn(numpy.median(model.x))

  


if __name__ == "__main__":
  #filename = sys.argv[1]
  #SpT = sys.argv[2]
  #vsini = float(sys.argv[3]) #MUST BE GIVEN IN KM S^1

  filename = "BG19000g400v2.vis.7"
  fulldata = ReadFile(filename)
  left = numpy.searchsorted(fulldata.x, 850)
  right = numpy.searchsorted(fulldata.x, 950)
  data = DataStructures.xypoint(right-left+1)
  data.x = fulldata.x[left:right]
  data.y = fulldata.y[left:right]
  data.cont = fulldata.cont[left:right]
  spectrum = Broaden(data, 150*Units.cm/Units.km)
  pylab.plot(spectrum.x, spectrum.y/spectrum.cont)
  pylab.show()
  #outfilename = "Broadened_" + SpT + "_v%.0f.dat" %(vsini)
  #print "Outputting to ", outfilename
  #numpy.savetxt(outfilename, numpy.transpose((spectrum.x, spectrum.y/spectrum.cont)), fmt='%.8f\t%.8g')