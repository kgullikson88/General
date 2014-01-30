"""
  This function will perform a bootstrap analysis to help
determine the CCF significance of a peak. 

  Inputs:
  -------
  -Nboot:      The number of bootstrap trials to do. Larger is 
                 better but takes longer
  -data:       A list of DataStructures.xypoint instances, containing
                 the data at each echelle order. The data should be
                 smoothed and the continuum should be fit accurately.
  -model_file: The filename of the model to cross-correlate against.
                 The model will be read in and processed once before 
                 doing the bootstrap analysis.
  -vsini:      The rotational velocity by which you want the model
                 broadened.
  -resolution: The detector resolution R=lambda/delta-lambda
  
  
  What it does:
  -------------
  Randomly sample from data with replacement, to make a version
    of the data with the same statistical properties but with any
    companion star signal scrambled up.
  Cross-correlate the fake data with the given model, and determine
    the maximum peak height.
  Repeat this process Nboot times, and return the peak height for 
    each sample in a list.
    
  What to do with the output:
  ---------------------------
  You can do one of the following:
    1: Determine the standard deviation of the peak heights to get
       the 1-sigma errors on CCF height from just noise.
    2: Make confidence intervals, which may be asymmetric.
    3: Plot the distribution itself, and estimate the probability
       of getting your CCF peak result from just noise. A low 
       probability means it is most likely a real signal.

"""

import numpy
from scipy.signal import fftconvolve
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import sys
import time
from astropy import units, constants
from multiprocessing import Pool
from functools import partial
import DataStructures
import FittingUtilities
import RotBroad_Fast as RotBroad
import HelperFunctions

minvel = -1000.0
maxvel = 1000.0

"""
  This function just processes the model to prepare for cross-correlation
"""
def Process(filename, data, vsini, resolution):
  
  #Read in the model
  print "Reading in the input model from %s" %filename
  x, y = numpy.loadtxt(filename, usecols=(0,1), unpack=True)
  x = x*units.angstrom.to(units.nm)
  y = 10**y
  left = numpy.searchsorted(x, data[0].x[0]-10)
  right = numpy.searchsorted(x, data[-1].x[-1]+10)
  model = DataStructures.xypoint(x=x[left:right], y=y[left:right])
  
  
  #Linearize the x-axis of the model
  print "Linearizing model"
  xgrid = numpy.linspace(model.x[0], model.x[-1], model.size())
  model = FittingUtilities.RebinData(model, xgrid)
  
  
  #Broaden
  print "Rotationally broadening model to vsini = %g km/s" %(vsini*units.cm.to(units.km))
  if vsini > 1.0*units.km.to(units.cm):
    model = RotBroad.Broaden(model, vsini, linear=True)
    
    
  #Reduce resolution
  print "Convolving to the detector resolution of %g" %resolution
  model = FittingUtilities.ReduceResolution(model, resolution) 
  
  
  # Rebin subsets of the model to the same spacing as the data
  model_orders = []
  for i, order in enumerate(data):
    sys.stdout.write("\rGenerating model subset for order %i in the input data" %(i+1))
    sys.stdout.flush()
    # Find how much to extend the model so that we can get maxvel range.
    dlambda = order.x[order.size()/2] * maxvel*1.5/3e5
    left = numpy.searchsorted(model.x, order.x[0] - dlambda)
    right = numpy.searchsorted(model.x, order.x[-1] + dlambda)
    
    # Figure out the log-spacing of the data
    start = numpy.log(order.x[0])
    end = numpy.log(order.x[-1])
    xgrid = numpy.logspace(start, end, order.size(), base=numpy.e)
    logspacing = numpy.log(xgrid[1]/xgrid[0])
    
    # Finally, space the model segment with the same log-spacing
    start = numpy.log(model.x[left])
    end = numpy.log(model.x[right])
    xgrid = numpy.exp(numpy.arange(start, end+logspacing, logspacing))
      
    segment = FittingUtilities.RebinData(model.copy(), xgrid)
    segment.cont = FittingUtilities.Continuum(segment.x, segment.y, lowreject=1.5, highreject=5, fitorder=2)
    model_orders.append(segment)
   
  
  return model_orders
  
  
  
  
  
"""
  This is the main function, which does the analysis
"""
def GetSamples(data, model_file, Nboot, vsini=10.0, resolution=50000):

  #First, read in and process the model
  model_orders = Process(model_file, data, vsini*units.km.to(units.cm), resolution)
  
  
  #Now, begin the bootstrap loop
  print "\n\nStarting loop over bootstrap trials. This could take a while..."
  output = numpy.zeros(Nboot)
  pool = Pool(processes=4)
  fcn = partial(GetCCFHeight, data, model_orders)
  #output = pool.map(fcn, range(Nboot), chunksize=Nboot/4)
  
  result = pool.imap_unordered(fcn, xrange(Nboot))
  #result = pool.map_async(fcn, xrange(Nboot))
  pool.close()
  
  
  while True:
    completed = result._index
    #completed = Nboot - result._number_left
    if (completed == Nboot): break
    #if result.ready(): break
    sys.stdout.write("\rExecution is %.2f%% done" %(100.0*completed/float(Nboot)))
    sys.stdout.flush()
    time.sleep(2)
  
  output = [r for r in result]
  #print output
  return output
  
    
  
def GetCCFHeight(data, model_orders, *args):
  #Copy the original data. We will overwrite this in a minute
  newdata = [order.copy() for order in data]
    
  #Randomly sample from each order with replacement to make fake data
  for j in range(len(newdata)):
    order = newdata[j]
    index = numpy.random.randint(0, order.size(), order.size())
    order.y = (order.y/order.cont)[index]
      
    #Resample to log-spacing
    start = numpy.log(order.x[0])
    end = numpy.log(order.x[-1])
    xgrid = numpy.logspace(start, end, order.size(), base=numpy.e)
    newdata[j] = FittingUtilities.RebinData(order, xgrid)
    
    
  #Now, cross-correlate the new data against the model
  corr = Correlate(newdata, model_orders)
  return numpy.max(corr.y)
  
    
    
    
    
"""
  This function does the actual correlation, in the same way
that we do it for the companion search.
"""    
def Correlate(data, model_orders):
  corrlist = []
  normalization = 0.0
  for ordernum, order in enumerate(data):
    #print "Cross-correlating order %i" %(ordernum+1)
    model = model_orders[ordernum]
    reduceddata = order.y
    reducedmodel = model.y/model.cont
    meandata = reduceddata.mean()
    meanmodel = reducedmodel.mean()
    data_rms = numpy.std(reduceddata)
    model_rms = numpy.std(reducedmodel)
    left = numpy.searchsorted(model.x, order.x[0])
    right = model.x.size - numpy.searchsorted(model.x, order.x[-1])
    delta = left - right
    
    #ycorr = numpy.correlate(reduceddata - meandata, reducedmodel - meanmodel, mode='valid')
    ycorr = fftconvolve((reduceddata - meandata), (reducedmodel - meanmodel)[::-1], mode='valid')
    xcorr = numpy.arange(ycorr.size)
    lags = xcorr - right
    distancePerLag = numpy.log(model.x[1] / model.x[0])
    offsets = -lags*distancePerLag
    velocity = offsets * constants.c.cgs.value * units.cm.to(units.km)
    corr = DataStructures.xypoint(velocity.size)
    corr.x = velocity[::-1]
    corr.y = ycorr[::-1]/(data_rms*model_rms*float(ycorr.size))
        
    # Only save part of the correlation
    left = numpy.searchsorted(corr.x, minvel)
    right = numpy.searchsorted(corr.x, maxvel)
    corr = corr[left:right]

    normalization += float(order.size())
    
    # Save correlation
    corrlist.append(corr.copy())

    
  # Add up the individual CCFs (use the Maximum Likelihood method from Zucker 2003, MNRAS, 342, 1291)
  total = corrlist[0].copy()
  total.y = numpy.ones(total.size())
  for i, corr in enumerate(corrlist):
    correlation = spline(corr.x, corr.y, k=1)
    N = data[i].size()
    total.y *= numpy.power(1.0 - correlation(total.x)**2, float(N)/normalization)
  master_corr = total.copy()
  master_corr.y = 1.0 - numpy.power(total.y, 1.0/float(len(corrlist)))
  
  return master_corr














