"""
Just a set of useful functions that I use often while fitting
"""
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline as smoother
import DataStructures

#Define bounding functions:
# lower bound:            lbound(boundary_value, parameter)
# upper bound:            ubound(boundary_value, parameter)
# lower and upper bounds: bound([low, high], parameter)
# fixed parameter:        fixed(fixed_value, parameter)
lbound = lambda p, x: 1e8*numpy.sqrt(p-x) + 1e-3*(p-x) if (x<p) else 0
ubound = lambda p, x: 1e8*numpy.sqrt(x-p) + 1e-3*(x-p) if (x>p) else 0
bound  = lambda p, x: lbound(p[0],x) + ubound(p[1],x)
fixed  = lambda p, x: bound((p,p), x)


#CCImprove
"""
  Improve the wavelength solution by a constant shift
"""
def CCImprove(data, model, be_safe=True, tol=0.2, debug=False):
  ycorr = numpy.correlate(data.y/data.cont-1.0, model.y/model.cont-1.0, mode="full")
  xcorr = numpy.arange(ycorr.size)
  maxindex = ycorr.argmax()
  lags = xcorr - (data.y.size-1)
  distancePerLag = (data.x[-1] - data.x[0])/float(data.x.size)
  offsets = -lags*distancePerLag

  if numpy.abs(offsets[maxindex]) < tol or not be_safe:
    if debug:
      return offsets[maxindex], DataStructures.xypoint(x=offsets, y=ycorr)
    else:
      return offsets[maxindex]
  else:
    return 0.0


"""
  This function fits the continuum spectrum by iteratively removing
points over one standard deviation below the mean, which are assumed
to be absorption lines.
"""
def Continuum(x, y, fitorder=3, lowreject=1, highreject=3, numiter=10000, function="poly"):
  done = False
  x2 = numpy.copy(x)
  y2 = numpy.copy(y)
  iteration = 0
  while not done and iteration < numiter:
    numiter += 1
    done = True
    if function == "poly":
      fit = numpy.poly1d(numpy.polyfit(x2 - x2.mean(), y2, fitorder))
    elif function == "spline":
      fit = smoother(x2, y2, s=fitorder)
    residuals = y2 - fit(x2 - x2.mean())
    mean = numpy.mean(residuals)
    std = numpy.std(residuals)
    badpoints = numpy.where(numpy.logical_or((residuals - mean) < -lowreject*std, residuals - mean > highreject*std))[0]
    if badpoints.size > 0 and x2.size - badpoints.size > 5*fitorder:
      done = False
      x2 = numpy.delete(x2, badpoints)
      y2 = numpy.delete(y2, badpoints)
  return fit(x - x2.mean())
