import numpy
import astropysics.obstools as tools
from scipy.optimize import leastsq, fmin
import matplotlib.pyplot as plt
import SpectralTypeRelations

"""
Function to fit a Cardelli, Clayton, & Mathis (1989) Reddening law to observed colors
Must give a dictionary of known colors with the name of the band as the key and the
  apparent magnitude as the value.
Must also give a spectral type and distance to the star
Can give an optional initial guess for Rv, but it is probably not necessary.
"""


Color2Lambda = {"U": 3650.0,
                "B": 4450.0,
                "V": 5510.0,
                "R": 6580.0,
                "I": 8060.0,
                "J": 12422.5979208,
                "H": 16513.172133,
                "K": 21655.7671846}


def FitFunction(colors, SpT, distance, Rv=3.1):
  #Set initial parameters
  pars = [1.6*distance/1000.0,
          Rv,
          distance]

  MS = SpectralTypeRelations.MainSequence()
  for color in colors:
    if colors[color][0] != None:
      try:
        absmag = MS.GetAbsoluteMagnitude(SpT, color)
        colors[color][0] -= absmag
      except ValueError:
        colors[color][0] = None

  extinct = tools.CardelliExtinction(Rv=pars[1])
  extinct.A0 = pars[0]
  for color in colors:
    if colors[color][0] != None:
      plt.errorbar(colors[color][2], colors[color][0], yerr=colors[color][1])
      print "%g +/- %g\n%g" %(colors[color][0], colors[color][1], extinct.Alambda(Color2Lambda[color]) + 5.0*numpy.log10(pars[2]) - 5.0)
      plt.plot(colors[color][2], extinct.Alambda(Color2Lambda[color]) + 5.0*numpy.log10(pars[2]) - 5.0, 'ro')
  plt.show()

  pars, success = leastsq(ErrorFunction, pars, args=(colors, SpT))
  chisq = ErrorFunction(pars, colors, SpT)
  extinct = tools.CardelliExtinction(Rv=pars[1])
  extinct.A0 = pars[0]
  for color in colors:
    if colors[color][0] != None:
      plt.errorbar(colors[color][2], colors[color][0], yerr=colors[color][1])
      print "%g +/- %g\n%g" %(colors[color][0], colors[color][1], extinct.Alambda(Color2Lambda[color]) + 5.0*numpy.log10(pars[2]) - 5.0)
      plt.plot(colors[color][2], extinct.Alambda(Color2Lambda[color]) + 5.0*numpy.log10(pars[2]) - 5.0, 'ro')
  plt.show()

  print "Pars, X^2 = ", pars, chisq
  return pars


def ErrorFunction(pars, colors, SpT):
  A0, Rv, d = pars[0], pars[1], pars[2]

  difference = 0.0
  extinct = tools.CardelliExtinction(Rv=Rv)
  extinct.A0 = A0
  for color in colors:
    if colors[color][0] != None:
      difference += (colors[color][0] - extinct.Alambda(Color2Lambda[color]) - 5.0*numpy.log10(d) + 5.0)**2 / colors[color][1]**2

  print pars, difference

  return difference*numpy.ones(10)


if __name__ == "__main__":
  infile = open("../Fluxes3.csv")
  lines = infile.readlines()
  infile.close()

  #Set up colors dictionary
  colors = {'U': [None, 1e-4, 360., 1823.],
          'B': [None, 1e-4, 430., 4130.],
          'V': [None, 1e-4, 550., 3781.],
          'R': [None, 1e-4, 700., 2941.],
          'I': [None, 1e-4, 900., 2635.],
          'J': [None, 1e-4, 1250., 1603.],
          'H': [None, 1e-4, 1600., 1075.],
          'K': [None, 1e-4, 2220., 667.],
          '2mass_J': [None, 1e-4, 1235., 1594.],
          '2mass_H': [None, 1e-4, 1662., 1024.],
          '2mass_K': [None, 1e-4, 2159., 666.7]}

  for line in lines[4:]:
    columns = line.split(",")
    star = columns[1].strip('"').strip("'").strip()
    if len(star.split()) > 1:
      cat = star.split()[0].strip()
      num = star.split()[1].strip()
      star = cat + num
    if "HR" not in star and "HIP" not in star:
      sys.exit("What is this crap!? %s" %star)
    SpT = columns[2].strip('"').strip("'").strip()[:2]
    metallicity = columns[3].strip()
    U = float(columns[4])
    Uerr = float(columns[5])
    B = float(columns[6])
    Berr = float(columns[7])
    V = float(columns[8])
    Verr = float(columns[9])
    R = float(columns[10])
    Rerr = float(columns[11])
    I = float(columns[12])
    Ierr = float(columns[13])
    J = float(columns[14])
    Jerr = float(columns[15])
    H = float(columns[16])
    Herr = float(columns[17])
    K = float(columns[18])
    Kerr = float(columns[19])
    L = float(columns[20])
    Lerr = float(columns[21])
    Lp = float(columns[22])        #L' filter
    Lperr = float(columns[23])
    M = float(columns[24])
    Merr = float(columns[25])

    parallax = float(columns[-2])
    perr = float(columns[-1])
    distance = 1000.0/parallax
    distance_error = perr/parallax**2 * 1000.0

    #Fill colors dictionary. First initialize all the colors to None
    for color in colors:
      colors[color][0] = None
    if U != -9:
      colors['U'][0] = U
      colors['U'][1] = max(Uerr, 0.01)
    if B != -9:
      colors['B'][0] = B
      colors['B'][1] = max(Berr, 0.01)
    if V != -9:
      colors['V'][0] = V
      colors['V'][1] = max(Verr, 0.01)
    if R != -9:
      colors['R'][0] = R
      colors['R'][1] = max(Rerr, 0.01)
    if I != -9:
      colors['I'][0] = I
      colors['I'][1] = max(Ierr, 0.01)
    if J != -9:
      colors['J'][0] = J
      colors['J'][1] = max(Jerr, 0.01)
    if H != -9:
      colors['H'][0] = H
      colors['H'][1] = max(Herr, 0.01)
    if K != -9:
      colors['K'][0] = K
      colors['K'][1] = max(Kerr, 0.01)

    pars = FitFunction(colors, SpT, distance)
    sys.exit()
