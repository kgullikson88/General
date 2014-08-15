import numpy
from scipy.interpolate import InterpolatedUnivariateSpline
cimport numpy
cimport cython
from libc.math cimport sqrt
from astropy import constants, units
import FittingUtilities

DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t


"""
  Here is the main, optimized function. It is stolen shamelessly from
  the avsini.c implementation given in the SPECTRUM code distribution.

  A wrapper called 'Broaden' with my standard calls is below
"""
@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.ndarray[DTYPE_t, ndim=1] convolve(numpy.ndarray[DTYPE_t, ndim=1] y,
                                             numpy.ndarray[DTYPE_t, ndim=1] ys,
                                             long num,
                                             long nd,
                                             double st,
                                             double dw,
                                             double vsini,
                                             double u):
  cdef double beta, gam, w, dlc, c1, c2, dv, r1, r2, v
  cdef long i, n1, n2, n
  cdef double clight = 299791.0
  cdef DTYPE_t f, t, s

  beta = (1.0-u)/(1.0-u/3.0)
  gam = u/(1.0-u/3.0)

  #End effect
  n1 = nd + 1
  ys[1:n1] = y[1:n1]
  n2 = num - nd - 1
  ys[n2:num+1] = y[n2:num+1]
  if vsini < 0.5:
    return y

  #Convolve with rotation profile
  w = st + (n1 - 1)*dw
  for n in range(n1, n2+1):
    w = w+dw
    s = 0.0
    t = 0.0
    dlc = w*vsini/clight
    c1 = 0.63661977*beta/dlc;
    c2 = 0.5*gam/dlc;
    dv = dw/dlc;

    for i in range(-nd, nd+1):
      v = i*dv
      r2 = 1.0 - v**2
      if r2 > 0.0:
        f = c1*sqrt(r2) + c2*r2
        t += f
        s += f*y[n+i]
    ys[n] = s/t
  
  return ys



"""
  This is the wrapper function. The user should call this!
"""

def Broaden(model, vsini, epsilon=0.5, linear=False, findcont=False):  
  """
    model:           xypoint instance with the data (or model) to be broadened
    vsini:           the velocity (times sin(i) ) of the star, in cm/s
    epsilon:          Linear limb darkening. I(u) = 1-epsilon + epsilon*u
    linear:          flag for if the x-spacing is already linear. If true, we don't need to linearize
    findcont:        flag to decide if the continuum needs to be found
  """

  if not findcont:
    cont_fcn = InterpolatedUnivariateSpline(model.x, model.cont)

  if not linear:
    x = numpy.linspace(model.x[0], model.x[-1], model.size())
    model = FittingUtilities.RebinData(model, x)
    if not findcont:
      model.cont = cont_fcn(model.x)
    else:
      model.cont = FittingUtilities.Continuum(model.x, model.y, lowreject=1.5, highreject=10)
  elif findcont:
    model.cont = FittingUtilities.Continuum(model.x, model.y, lowreject=1.5, highreject=10)

  #Need to prepare a few more variables before calling 'convolve'
  broadened = model.copy()
  num = model.size()
  ys = numpy.ones(num)
  start = model.x[0]
  dwave = model.x[1] - model.x[0]

  s2 = (start + num*dwave)*vsini/(dwave*constants.c.cgs.value)
  vsini *= units.cm.to(units.km)
  nd = s2 + 5.5

  broadened.y = convolve(model.y, ys, num, nd, start, dwave, vsini, epsilon)
  return broadened
