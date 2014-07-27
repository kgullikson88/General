import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline as smoother

"""
  This function fits the continuum spectrum by iteratively removing
points over one standard deviation below the mean, which are assumed
to be absorption lines.
"""
def Continuum(x, y, fitorder=3, lowreject=1, highreject=3, numiter=10000, function="poly"):
  done = False
  x2 = np.copy(x)
  y2 = np.copy(y)
  iteration = 0
  while not done and iteration < numiter:
    numiter += 1
    done = True
    if function == "poly":
      fit = np.poly1d(np.polyfit(x2 - x2.mean(), y2, fitorder))
    elif function == "spline":
      fit = smoother(x2, y2, s=fitorder)
    residuals = y2 - fit(x2 - x2.mean())
    mean = np.mean(residuals)
    std = np.std(residuals)
    badpoints = np.where(np.logical_or((residuals - mean) < -lowreject*std, residuals - mean > highreject*std))[0]
    if badpoints.size > 0 and x2.size - badpoints.size > 5*fitorder:
      done = False
      x2 = np.delete(x2, badpoints)
      y2 = np.delete(y2, badpoints)
  return fit(x - x2.mean())


"""
  This function finds the parts of the spectrum that are close to the continuum,
and so contain neither absorption nor emission lines

DOES NOT WORK VERY WELL YET!
"""
def GetContinuumRegions(x, y, cont=None, lowreject=3, highreject=3):
  if cont == None:
    cont = Continuum(x,y)

  normalized = y/cont
  x2 = x.copy()
  done = False
  goodpoints = range(x.size)
  while not done:
    done = True
    normalized2 = normalized[goodpoints]
    std = np.std(normalized2)
    badpoints = np.where(np.logical_or(normalized2 - 1.0 < -lowreject*std, normalized2 - 1.0 > highreject*std))[0]
    goodpoints = np.where(np.logical_and(normalized - 1.0 > -lowreject*std, normalized - 1.0 < highreject*std))[0]
    if badpoints.size > 0 and goodpoints.size - badpoints.size > 10:
      done = False

  return goodpoints
  


"""
  This function does a boxcar smoothing instead of fitting to a polynomial as above.
"""
def IterativeBoxcarSmooth(x, y, bcwidth=50, numstd=2.0):
  boxcar = np.ones(bcwidth)/float(bcwidth)
  done = False
  sizes = []    #To check if it gets stuck
  x2 = np.array(x)
  y2 = np.array(y)
  print "Number of standard deviations = %.2g" %numstd
  while not done:
    after = y2[-boxcar.size/2+1:]
    before = y2[:boxcar.size/2]
    extended = np.append(np.append(before, y2), after)
    smoothed = np.convolve(extended, boxcar, mode='valid')
    residuals = y2/smoothed - 1.0
    std = np.std(residuals)
    if std < 1e-8:
      std = 1e-8
    badindices = np.where(np.logical_and(residuals < -numstd*std, residuals > numstd*std))[0]
    if badindices.size < 5 or (len(sizes) > 10 and np.all(size == badindices.size for size in sizes[-10:])):
      done = True
      break
    sizes.append(badindices.size)
    goodindices = np.where(np.logical_and(residuals >= -numstd*std, residuals <= numstd*std))[0]
    fcn = spline(x2[goodindices], y2[goodindices], k=1)
    y2 = fcn(order2.x)
  return smoothed
