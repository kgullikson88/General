import numpy
import matplotlib.pyplot as plt
import FitsUtils
import pyfits
from astropy import units
import os
import scipy.signal
import DataStructures

"""
  Function to find the spectral lines, given a model spectrum
  spectrum:        An xypoint instance with the model
  tol:             The line strength needed to count the line (0 is a strong line, 1 is weak)
  linespacing:     The minimum spacing between two consecutive lines
"""
def FindLines(spectrum, tol=0.99, linespacing = 0.01, debug=False):
  distance = 0.01
  xspacing = float(max(spectrum.x) - min(spectrum.x))/float(spectrum.size())
  N = int( linespacing / xspacing + 0.5)
  lines = list(scipy.signal.argrelmin(spectrum.y, order=N)[0])
  if debug:
    plt.plot(spectrum.x, spectrum.y)
  for i in range(len(lines)-1, -1, -1):
    idx = lines[i]
    xval = spectrum.x[idx]
    yval = spectrum.y[idx]
    if yval < tol:
      plt.plot([xval, xval], [yval-0.01, yval-0.03], 'r-')
    else:
      lines.pop(i)
  plt.show()
  return numpy.array(lines)
#numpy.savetxt("Linelist4.dat", lines, fmt="%.8f")



if __name__ == "__main__":
  filename = os.environ["HOME"] + "/School/Research/Useful_Datafiles/Telluric_Visible.dat"
  filename = "tell.dat"
  print "Reading telluric model"
  x,trans = numpy.loadtxt(filename, unpack=True)
  model = DataStructures.xypoint(x=x[::-1], y=trans[::-1])
  FindLines(model, debug=True)
