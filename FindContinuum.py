import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as spline

"""
  This function fits the continuum spectrum by iteratively removing
points over one standard deviation below the mean, which are assumed
to be absorption lines.
"""
def Continuum(x, y, fitorder=3):
  done = False
  x2 = numpy.copy(x)
  y2 = numpy.copy(y)
  while not done:
    done = True
    fit = numpy.poly1d(numpy.polyfit(x2 - x2.mean(), y2, fitorder))
    residuals = y2 - fit(x2 - x2.mean())
    mean = numpy.mean(residuals)
    std = numpy.std(residuals)
    badpoints = numpy.where((residuals - mean) < -std)[0]
    if badpoints.size > 0 and x2.size - badpoints.size > 5*fitorder:
      done = False
      x2 = numpy.delete(x2, badpoints)
      y2 = numpy.delete(y2, badpoints)
  return fit(x - x2.mean())


"""
  This function does a boxcar smoothing instead of fitting to a polynomial as above.
"""
def IterativeBoxcarSmooth(x, y, bcwidth=50, numstd=2.0):
  boxcar = numpy.ones(bcwidth)/float(bcwidth)
  done = False
  sizes = []    #To check if it gets stuck
  x2 = numpy.array(x)
  y2 = numpy.array(y)
  print "Number of standard deviations = %.2g" %numstd
  while not done:
    after = y2[-boxcar.size/2+1:]
    before = y2[:boxcar.size/2]
    extended = numpy.append(numpy.append(before, y2), after)
    smoothed = numpy.convolve(extended, boxcar, mode='valid')
    residuals = y2/smoothed - 1.0
    std = numpy.std(residuals)
    if std < 1e-8:
      std = 1e-8
    badindices = numpy.where(numpy.logical_and(residuals < -numstd*std, residuals > numstd*std))[0]
    if badindices.size < 5 or (len(sizes) > 10 and numpy.all(size == badindices.size for size in sizes[-10:])):
      done = True
      break
    sizes.append(badindices.size)
    goodindices = numpy.where(numpy.logical_and(residuals >= -numstd*std, residuals <= numstd*std))[0]
    fcn = spline(x2[goodindices], y2[goodindices], k=1)
    y2 = fcn(order2.x)
  return smoothed
