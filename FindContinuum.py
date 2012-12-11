import numpy



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
