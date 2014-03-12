"""
  This is a general function for all broadening. Importing this
  will let the user do rotational broadening, macroturbulent 
  broadening, and Gaussian broadening (reducing the resolution)
"""


from RotBroad_Fast import Broaden as RotBroad
from FittingUtilities import ReduceResolution
import numpy
from scipy.special import erf   #Error function
from astropy import constants, units
from scipy.signal import fftconvolve



def MacroBroad(data, vmacro, extend=True):
  """
    This broadens the data by a given macroturbulent velocity.
  It works for small wavelength ranges. I need to make a better
  version that is accurate for large wavelength ranges! Sorry
  for the terrible variable names, it was copied from 
  convol.pro in AnalyseBstar (Karolien Lefever)

  -data:   A DataStructures.xypoint instance storing the data to be 
           broadened. The data MUST be equally-spaced before calling
           this!

  -vmacro: The macroturbulent velocity, in km/s

  -extend: If true, the y-axis will be extended to avoid edge-effects
  """
  # Make the kernel
  c = constants.c.cgs.value * units.cm.to(units.km)
  sq_pi = numpy.sqrt(numpy.pi)
  lambda0 = numpy.median(data.x)
  xspacing = data.x[1] - data.x[0]
  mr = vmacro * lambda0 / c
  ccr = 2/(sq_pi * mr)

  px = numpy.arange(-data.size()/2, data.size()/2+1) * xspacing
  pxmr = abs(px) / mr
  profile = ccr * (numpy.exp(-pxmr**2) + sq_pi*pxmr*(erf(pxmr) - 1.0))

  # Extend the xy axes to avoid edge-effects, if desired
  if extend:
    
    before = data.y[-profile.size/2+1:]
    after = data.y[:profile.size/2]
    extended = numpy.r_[before, data.y, after]

    first = data.x[0] - float(int(profile.size/2.0+0.5))*xspacing
    last = data.x[-1] + float(int(profile.size/2.0+0.5))*xspacing
    x2 = numpy.linspace(first, last, extended.size) 
    
    conv_mode = "valid"

  else:
    extended = data.y.copy()
    x2 = data.x.copy()
    conv_mode = "same"

  # Do the convolution
  newdata = data.copy()
  newdata.y = fftconvolve(extended, profile/profile.sum(), mode=conv_mode)
    
  return newdata
  

  
