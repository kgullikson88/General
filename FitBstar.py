import scipy
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import leastsq
from collections import defaultdict
import pyfits
import numpy
import sys
import os
import RotBroad
import Units
import DataStructures
import pylab


homedir = os.environ['HOME']
Bstarfiles = {#2.25: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g225v2.vis.7",
              #2.50: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g250v2.vis.7",
              #2.75: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g275v2.vis.7",
              #3.00: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g300v2.vis.7",
              3.25: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g325v2.vis.7",
              3.50: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g350v2.vis.7",
              3.75: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g375v2.vis.7",
              4.00: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g400v2.vis.7",
              4.25: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g425v2.vis.7",
              4.50: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g450v2.vis.7",
              4.75: homedir + "/School/Research/McDonaldData/BstarModels/BG19000g475v2.vis.7",
              13.25: homedir + "/School/Research/McDonaldData/BstarModels/BG17000g325v2.vis.7",
              13.50: homedir + "/School/Research/McDonaldData/BstarModels/BG17000g350v2.vis.7",
              13.75: homedir + "/School/Research/McDonaldData/BstarModels/BG17000g375v2.vis.7",
              14.00: homedir + "/School/Research/McDonaldData/BstarModels/BG17000g400v2.vis.7",
              14.25: homedir + "/School/Research/McDonaldData/BstarModels/BG17000g425v2.vis.7",
              14.50: homedir + "/School/Research/McDonaldData/BstarModels/BG17000g450v2.vis.7",
              14.75: homedir + "/School/Research/McDonaldData/BstarModels/BG17000g475v2.vis.7",
              23.25: homedir + "/School/Research/McDonaldData/BstarModels/BG15000g325v2.vis.7",
              23.50: homedir + "/School/Research/McDonaldData/BstarModels/BG15000g350v2.vis.7",
              23.75: homedir + "/School/Research/McDonaldData/BstarModels/BG15000g375v2.vis.7",
              24.00: homedir + "/School/Research/McDonaldData/BstarModels/BG15000g400v2.vis.7",
              24.25: homedir + "/School/Research/McDonaldData/BstarModels/BG15000g425v2.vis.7",
              24.50: homedir + "/School/Research/McDonaldData/BstarModels/BG15000g450v2.vis.7",
              24.75: homedir + "/School/Research/McDonaldData/BstarModels/BG15000g475v2.vis.7",}


bstar_fcns = defaultdict(scipy.interpolate.fitpack2.InterpolatedUnivariateSpline)
print "Reading in primary star models"
for key in Bstarfiles:
  filename = Bstarfiles[key]
  print "Reading model ", filename
  #star = RotBroad.ReadFile(filename)
  #bstar_fcns[key] = UnivariateSpline(star.x, star.y/star.cont, s=0)


#Define bounding functions:
# lower bound:            lbound(boundary_value, parameter)
# upper bound:            ubound(boundary_value, parameter)
# lower and upper bounds: bound([low, high], parameter)
# fixed parameter:        fixed(fixed_value, parameter)
lbound = lambda p, x: 1e4*numpy.sqrt(p-x) + 1e-3*(p-x) if (x<p) else 0
ubound = lambda p, x: 1e4*numpy.sqrt(x-p) + 1e-3*(x-p) if (x>p) else 0
bound  = lambda p, x: lbound(p[0],x) + ubound(p[1],x)
fixed  = lambda p, x: bound((p,p), x)


def ErrorFunction(pars, dat, stars):
  model, logg, chisq = FindBestGravity(dat, stars, pars[0], pars[1])
  #print "RV = ", pars[0]*Units.km/Units.cm, "km/s"
  #print "vsini: ", pars[1]*Units.km/Units.cm, "km/s"
  #print "X^2 = ", chisq[0]
  #pylab.plot(dat.x, dat.y)
  #pylab.plot(model.x, model.y)
  #pylab.show()

  return chisq

"""
  Main fitting function. Give an xypoint (data), and a
  dictionary of UnivariateSplines for the un-broadened B-star models.
  Will fit the radial velocity (RV), and rotational velocity vsini
     (Initial guesses for these can be given as optional arguments)
     Both should be given in km/s!!
  EDIT: now it fits a quadratic correction to the model wavelengths
"""
def Fit(data, models=bstar_fcns, RV=0.0, vsini=100.0):
  pars = [RV*Units.cm/Units.km, vsini*Units.cm/Units.km]
  #ErrorFunction = lambda pars, dat, stars: FindBestGravity(dat, stars, pars[0], pars[1])[2]
  pars, success = leastsq(ErrorFunction, pars, args=(data, models), epsfcn=0.1, maxfev=10000)

  print "******************************************"
  print "Done fitting!"
  print "Fit success code = ", success
  print "******************************************"
  print "Best-fit RV: ", pars[0]*Units.km/Units.cm, "km/s"
  print "Best-fit vsini: ", pars[1]*Units.km/Units.cm, "km/s"
  bestfit, best_logg, chisq = FindBestGravity(data, models, pars[0], pars[1])
  print "Best log(g) = ", best_logg
  print "X^2 = ", chisq[0]
  return bestfit


"""
  Function to make a model B star given a UnivariateSpline of it,
  the x-points you want points at, a radial velocity, and a rotational velocity
"""
def GetModel(spline, data, RV, vsini):
  #We first need to make an evenly-sample x grid that is wider than the final one, so edge effects are not an issue
  xgrid = data.x
  xspacing = (xgrid[-1] - xgrid[0])/float(xgrid.size - 1)
  first = 2*xgrid[0] - xgrid[-1]
  last = 2*xgrid[-1] - xgrid[0]
  x = numpy.arange(first, last, xspacing)

  z = RV/Units.c
  unbroadened = DataStructures.xypoint(x=x, y=spline(x*(1+z)), cont=numpy.ones(x.size))

  broadened = RotBroad.Broaden(unbroadened, vsini, linear=True, alpha=0.1)

  #Now, we must spline the broadened function onto the xgrid
  fcn = UnivariateSpline(broadened.x, broadened.y/broadened.cont, s=0)
  a,b,c = AdjustWaveScale(data, fcn)
  #retarray = DataStructures.xypoint(x=xgrid, y=fcn(xgrid), cont=numpy.ones(xgrid.size))
  retarray = DataStructures.xypoint(x=xgrid, y=fcn(a+b*xgrid+c*xgrid**2), cont=numpy.ones(xgrid.size))

  #Adjust model continuum
  corrected = data.y/(data.cont*retarray.y)
  temp = DataStructures.xypoint(x=data.x, y=corrected)
  done = False
  while not done:
    done = True
    contfcn = numpy.poly1d(numpy.polyfit(temp.x, temp.y, 2))
    residuals = temp.y - contfcn(temp.x)
    sigma = numpy.std(residuals)
    badindices = numpy.where(residuals < -2*sigma)[0]
    if badindices.size > 0 and badindices.size < temp.x.size:
      temp.x = numpy.delete(temp.x, badindices)
      temp.y = numpy.delete(temp.y, badindices)
      done = False
  retarray.cont = 1.0/contfcn(retarray.x)

  return retarray


"""
  Function to search through the dictionary of models with different surface gravities
  Returns the model that gives the best fit
"""
def FindBestGravity(data, models, RV, vsini):
  best_chisq = 1e100
  best_logg = 0
  add = 0.0
  if vsini*Units.km/Units.cm < 1.0:
    add = bound([1.0, 600], vsini*Units.km/Units.cm)
    vsini = 100.0*Units.cm/Units.km
  if vsini*Units.km/Units.cm > 150.0:
    add = bound([1.0, 150], vsini*Units.km/Units.cm)
    vsini = 100.0*Units.cm/Units.km
  for key in models.keys():
    star = GetModel(models[key], data, RV, vsini)
    chisq = numpy.sum((data.y-star.y/star.cont)**2/data.err**2)
    if chisq < best_chisq:
      best_chisq = chisq
      best_logg = key

  star = GetModel(models[best_logg], data, RV, vsini)
  return star, best_logg, (best_chisq+add)*numpy.ones(data.x.size)


def AdjustWaveScale(data, model, a=0., b=1., c=0.):
  pars = [a,b,c]
  wavefcn = lambda pars, model_fcn, dat: (dat.y - model_fcn(pars[0]+pars[1]*dat.x+pars[2]*dat.x**2))**2/dat.err
  pars, success = leastsq(wavefcn, pars, args=(model, data))

  return pars[0], pars[1], pars[2]



def main1():
  import FitsUtils
  infilename = "output-9.fits"
  orders = FitsUtils.MakeXYpoints(infilename)
  bcwidth = 20
  for i in range(20,51):
    done = False
    print "\nFitting order %i" %(i+1)
    while not done:
      boxcar = numpy.ones(bcwidth)/float(bcwidth)
      order = orders[i]
      #star = Fit(order, RV=200.0)
      after = order.y[-boxcar.size/2+1:]
      before = order.y[:boxcar.size/2]
      extended = numpy.append(numpy.append(before, order.y), after)
      star = DataStructures.xypoint(x=order.x, y=numpy.convolve(extended, boxcar, mode='valid'))

      pylab.figure(1)
      pylab.plot(order.x, order.y/order.cont)
      pylab.plot(star.x, star.y/star.cont)
      pylab.title("Order %i" %(i+1))
      pylab.show()

      inp = raw_input("Divide by fit? ")
      if "y" in inp or inp == "":
        orders[i].y /= star.y/star.cont
        FitsUtils.OutputFitsFile(infilename, orders)
        done = True
      elif "#" in inp:
        bcwidth = max(2, int(inp[1:]))
      elif "+" in inp or "-" in inp:
        bcwidth += int(inp)
        bcwidth = max(2, bcwidth)
        print "Boxcar width = ", bcwidth
      elif "n" in inp:
        break


def GetApproximateSpectrum(order, bcwidth=50, numstd=2.0):
  boxcar = numpy.ones(bcwidth)/float(bcwidth)
  gausswidth = bcwidth/2.0
  #xspacing = (order.y[-1] - order.y[0])/float(order.y.size - 1.0)
  #x = numpy.linspace(-bcwidth/2.0, bcwidth/2.0, bcwidth)
  #gaussian = numpy.exp(x**2 / (2*gausswidth**2))
  #gaussian = gaussian / gaussian.sum()
  done = False
  order2 = order.copy()
  sizes = []    #To check if it gets stuck
  while not done:
    after = order2.y[-boxcar.size/2+1:]
    before = order2.y[:boxcar.size/2]
    extended = numpy.append(numpy.append(before, order2.y), after)
    smoothed = numpy.convolve(extended, boxcar, mode='valid')
    #smoothed = numpy.convolve(extended, gaussian, mode='valid')
    residuals = order2.y/smoothed - 1.0
    std = numpy.std(residuals)
    if std < 1e-8:
      std = 1e-8
    badindices = numpy.where(numpy.logical_and(residuals < -numstd*std, residuals > numstd*std))[0]
    if badindices.size < 5 or (len(sizes) > 10 and numpy.all(size == badindices.size for size in sizes[-10:])):
      done = True
      break
    sizes.append(badindices.size)
    goodindices = numpy.where(numpy.logical_and(residuals >= -numstd*std, residuals <= numstd*std))[0]
    fcn = spline(order2.x[goodindices], order2.y[goodindices], k=1)
    order2.y = fcn(order2.x)
  star = order.copy()
  star.y = smoothed.copy()
  return star

def main2():
  import FitsUtils
  infilename = "Star_17_Vul_NDIT10-2.fits"
  orders = FitsUtils.MakeXYpoints(infilename)
  bstar = GetApproximateSpectrum(orders)
  for i in range(17,len(orders)):
    order = orders[i]
    star = bstar[i]
    pylab.plot(order.x, order.y)
    pylab.plot(star.x, star.y)
    pylab.show()
    
  

if __name__ == "__main__":
  main2()

  
