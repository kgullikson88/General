import subprocess
import matplotlib.pyplot as plt
import sys
import os
import numpy
from collections import defaultdict
from scipy.interpolate import UnivariateSpline
import scipy.signal
import DataStructures
from astropy import units, constants
import FittingUtilities
import RotBroad_Fast as RotBroad
import time
import FittingUtilities
import HelperFunctions
from astropy import units, constants


currentdir = os.getcwd() + "/"
homedir = os.environ["HOME"]
outfiledir = currentdir + "Cross_correlations/"
modeldir = homedir + "/School/Research/Models/Sorted/Stellar/Vband/"
minvel = -1000  #Minimum velocity to output, in km/s
maxvel = 1000  

HelperFunctions.ensure_dir(outfiledir)

model_list = [modeldir + "lte30-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte31-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte32-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte33-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte34-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte35-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte36-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte37-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte38-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte39-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte40-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte42-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte43-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte44-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte45-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte46-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte47-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte48-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte49-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte50-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte51-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte52-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte53-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte54-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte55-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte56-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte57-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte58-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte59-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte61-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte63-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte64-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte65-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte67-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte68-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte69-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte70-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte72-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted"]

star_list = []
temp_list = []
gravity_list = []
metallicity_list = []
for fname in model_list:
  temp = int(fname.split("lte")[-1][:2])*100
  gravity = float(fname.split("lte")[-1][3:7])
  metallicity = float(fname.split("lte")[-1][7:11])
  star_list.append(str(temp))
  temp_list.append(temp)
  gravity_list.append(gravity)
  metallicity_list.append(metallicity)



#A convenience function for legacy support
def PyCorr2(data, stars=star_list, temps=temp_list, models=model_list, model_fcns = None, gravities=gravity_list, metallicities=metallicity_list, corr_mode='valid', process_model=True, vsini=15*units.km.to(units.cm), resolution=100000, segments="all", save_output=True, outdir=outfiledir, outfilename=None, outfilebase="", debug=False):
  PyCorr(data, stars=star_list, temps=temp_list, models=model_list, model_fcns = None, gravities=gravity_list, metallicities=metallicity_list, corr_mode='valid', process_model=True, vsini=15*units.km.to(units.cm), resolution=100000, segments="all", save_output=True, outdir=outfiledir, outfilename=None, outfilebase="", debug=False)



"""
   Function to make a cross-correlation function out of echelle data.
   Expects a list of xypoints as input, and various optional inputs.
"""
def PyCorr(data, stars=star_list, temps=temp_list, models=model_list, model_fcns = None, gravities=gravity_list, metallicities=metallicity_list, corr_mode='valid', process_model=True, vsini=15*units.km.to(units.cm), resolution=100000, segments="all", save_output=True, outdir=outfiledir, outfilename=None, outfilebase="", debug=False):

  makefname = False
  if outfilename == None:
    makefname = True

  if model_fcns == None:
    model_fcns = [None for m in models]
    
  
  #Re-sample all orders of the data to logspacing
  if debug:
    print "Resampling data to log-spacing"
  for i, order in enumerate(data):
    if debug:
      print "Resampling order %i to log-spacing" %i
    start = numpy.log(order.x[0])
    end = numpy.log(order.x[-1])
    neworder = order.copy()
    neworder.x = numpy.logspace(start, end, order.size(), base=numpy.e)
    neworder = FittingUtilities.RebinData(order, neworder.x)
    data[i] = neworder    

    
  #3: Begin loop over model spectra
  returnlist = []
  for i in range(len(models)):
    star = stars[i]
    temp = temps[i]
    gravity = gravities[i]
    metallicity = metallicities[i]

    if makefname:
      try:
        vsini_str = vsini*units.cm.to(units.km)
      except TypeError:
        vsini_str = 0.0
      outfilename = "%s%s.%.0fkps_%sK%+.1f%+.1f" %(outdir, outfilebase, vsini_str, star, gravity, metallicity)

    #a: Read in file (or  rename if already read in: PREFERRABLE!)
    if isinstance(models[i], str):
      print "******************************\nReading file ", modelfile
      x,y = numpy.loadtxt(models[i], usecols=(0,1), unpack=True)
      x *= units.angstrom.to(units.nm)
      y = 10**y
      cont = numpy.ones(y.size)*y.max()
      model = DataStructures.xypoint(x=x, y=y, cont=cont)
    elif isinstance(models[i], DataStructures.xypoint):
      model = models[i].copy()
    else:
      sys.exit("Model #%i of unkown type in Correlate.PyCorr2!" %i) 

    if process_model:
      if debug:
        print "Processing model..."
      left = numpy.searchsorted(model.x, data[0].x[0] - 10.0)
      right = numpy.searchsorted(model.x, data[-1].x[-1] + 10.0)
      if left > 0:
        left -= 1
      x2 = model.x[left:right].copy()
      y2 = model.y[left:right].copy()
      cont2 = FittingUtilities.Continuum(x2, y2, fitorder=2)
      if model_fcns[i] == None:
        if debug:
          print "Interpolating model"
        MODEL = UnivariateSpline(x2,y2, s=0)
      else:
        if debug:
          print "Model already interpolated. Thanks!"
        MODEL = model_fcns[i]
      #CONT = UnivariateSpline(x2, cont2, s=0)
      
    
    #h: Cross-correlate
    corrlist = []
    normalization = 0.0
    for ordernum, order in enumerate(data):
      if process_model:
        left = max(0, numpy.searchsorted(model.x, 2*order.x[0] - order.x[-1])-1)
        right = min(model.x.size-1, numpy.searchsorted(model.x, 2*order.x[-1] - order.x[0]))
        if left > 0:
          left -= 1

        #b: Make wavelength spacing constant
        model2 = DataStructures.xypoint(right - left + 1)
        model2.x = numpy.linspace(model.x[left], model.x[right], right - left + 1)
        model2.y = MODEL(model2.x)
        model2.cont = FittingUtilities.Continuum(model2.x, model2.y, lowreject=1.5, highreject=5, fitorder=2)
        model2.cont[model2.cont < 1e-5] = 1e-5

        #d: Rotationally broaden
        if vsini != None and vsini > 1.0*units.km.to(units.cm):
          model2 = RotBroad.Broaden(model2, vsini, linear=True)
          if debug:
            print "After rotational broadening"
      
        #e: Convolve to detector resolution
        if resolution != None:
          model2 = FittingUtilities.ReduceResolution(model2.copy(), resolution, extend=False)
        if debug:
          print "After resolution decrease"

        #f: Rebin to the same spacing as the data
        logspacing = numpy.log(order.x[1]/order.x[0])
        start = numpy.log(model2.x[0])
        end = numpy.log(model2.x[-1])
        xgrid = numpy.exp(numpy.arange(start, end+logspacing, logspacing))
        #xgrid = numpy.arange(model2.x[0], model2.x[-1], order.x[1] - order.x[0])
        model2 = FittingUtilities.RebinData(model2.copy(), xgrid)
        model2.cont = FittingUtilities.Continuum(model2.x, model2.y, lowreject=1.5, highreject=5, fitorder=2)
        if debug:
          print "After rebinning"

      #Now, do the actual cross-correlation
      reduceddata = order.y/order.cont
      reducedmodel = model2.y/model2.cont
      meandata = reduceddata.mean()
      meanmodel = reducedmodel.mean()
      data_rms = numpy.sqrt(numpy.sum((reduceddata - meandata)**2)/float(reduceddata.size))
      model_rms = numpy.sqrt(numpy.sum((reducedmodel - meanmodel)**2)/float(reducedmodel.size))
      left = numpy.searchsorted(model2.x, order.x[0])
      right = model2.x.size - numpy.searchsorted(model2.x, order.x[-1])
      delta = left - right
      if debug:
        order.output("Corr_inputdata.dat")
        model2.output("Corr_inputmodel.dat")
    
      #ycorr = scipy.signal.fftconvolve((reduceddata - meandata), (reducedmodel - meanmodel)[::-1], mode=corr_mode)
      
      ycorr = numpy.correlate(reduceddata - meandata, reducedmodel - meanmodel, mode=corr_mode)
      xcorr = numpy.arange(ycorr.size)
      if corr_mode == 'valid':
        lags = xcorr - (model2.x.size + order.x.size + delta - 1.0)/2.0
        lags = xcorr - (0 + right)
      elif corr_mode == 'full':
        lags = xcorr - model2.x.size
      else:
        sys.exit("Sorry! corr_mode = %s not supported yet!" %corr_mode)
      #distancePerLag = model2.x[1] - model2.x[0]
      #offsets = -lags*distancePerLag
      #velocity = offsets*3e5 / numpy.median(order.x)   
      distancePerLag = numpy.log(model2.x[1] / model2.x[0])
      offsets = -lags*distancePerLag
      velocity = offsets * constants.c.cgs.value * units.cm.to(units.km)
      corr = DataStructures.xypoint(velocity.size)
      corr.x = velocity[::-1]
      #corr.y = ycorr[::-1]/model_rms
      corr.y = ycorr[::-1]/(data_rms*model_rms*float(ycorr.size))
      if debug:
        if numpy.any(numpy.isnan([corr.y[i] for i in range(corr.size())])):
          print "NaN found in correlation!"
          corr.output("badcorrelation.dat")
          print data_rms, model_rms, reduceddata.size
          sys.exit()
      elif numpy.any(numpy.isnan([corr.y[i] for i in range(corr.size())])):
        print "NaN found in correlation!"
        #Output the inputs that are causing the issue
        order.output("%s.badccf_data.order%i" %(outfilename, ordernum+1))
        model2.output("%s.badccf_model.order%i" %(outfilename, ordernum+1))
        
          
        
      #i: Only save part of the correlation
      left = numpy.searchsorted(corr.x, minvel)
      right = numpy.searchsorted(corr.x, maxvel)
      corr.x = corr.x[left:right]
      corr.y = corr.y[left:right]

      normalization += float(order.size())
      #k: Save correlation
      corrlist.append(corr.copy())
      if debug:
        print "Outputting single-order CCF to %s.order%i" %(outfilename, ordernum+1)
        numpy.savetxt("%s.order%i" %(outfilename, ordernum+1), numpy.transpose((corr.x, corr.y)))
        #corr.output("%s.order%i" %(outfilename, ordernum+1))

    #Add up the individual CCFs (use the Maximum Likelihood method from Zucker 2003, MNRAS, 342, 1291)
    total = corrlist[0].copy()
    total.y = numpy.ones(total.size())
    #N = orders[0].size()
    #total.y = numpy.power(1.0 - total.y**2, float(N)/normalization)
    #master_corr = corrlist[0]
    for i, corr in enumerate(corrlist):
      correlation = UnivariateSpline(corr.x, corr.y, s=0, k=1)
      N = data[i].size()
      total.y *= numpy.power(1.0 - correlation(total.x)**2, float(N)/normalization)
      #master_corr.y += correlation(master_corr.x)
    #master_corr.y /= normalization
    master_corr = total.copy()
    master_corr.y = 1.0 - numpy.power(total.y, 1.0/float(len(corrlist)))

    #Finally, output
    if save_output:
      print "Outputting to ", outfilename, "\n"
      numpy.savetxt(outfilename, numpy.transpose((master_corr.x, master_corr.y)), fmt="%.10g")
      returnlist.append(outfilename)
    else:
      returnlist.append(master_corr)

  return returnlist




"""
   Autocorrelate each model against itself, on the xgrids given by the data orders
"""

def AutoCorrelate(data, stars=star_list, temps=temp_list, models=model_list, gravities=gravity_list, metallicities=metallicity_list, corr_mode='valid', process_model=True, normalize=False, vsini=15*units.km.to(units.cm), resolution=100000, segments="all", save_output=True, outdir=outfiledir, outfilename=None, outfilebase="", debug=False):

  ensure_dir(outdir)

  makefname = False
  if outfilename == None and save_output:
    makefname = True
    
  #3: Begin loop over model spectra
  returnlist = []
  for i in range(len(models)):
    star = stars[i]
    temp = temps[i]
    gravity = gravities[i]
    metallicity = metallicities[i]

    if makefname:
      outfilename = "%s%s.%.0fkps_%sK%+.1f%+.1f" %(outdir, outfilebase, vsini*units.cm.to(units.km), star, gravity, metallicity)

    #a: Read in file (or  rename if already read in: PREFERRABLE!)
    if isinstance(models[i], str):
      print "******************************\nReading file ", modelfile
      x,y = numpy.loadtxt(models[i], usecols=(0,1), unpack=True)
      x *= units.angstrom.to(units.nm)
      y = 10**y
      cont = numpy.ones(y.size)*y.max()
      model = DataStructures.xypoint(x=x, y=y, cont=cont)
    elif isinstance(models[i], DataStructures.xypoint):
      model = models[i].copy()

    if process_model:
      left = numpy.searchsorted(model.x, data[0].x[0] - 10.0)
      right = numpy.searchsorted(model.x, data[-1].x[-1] + 10.0)
      if left > 0:
        left -= 1
      x2 = model.x[left:right].copy()
      y2 = model.y[left:right].copy()
      cont2 = FittingUtilities.Continuum(x2, y2, fitorder=5)
      MODEL = UnivariateSpline(x2,y2, s=0)
      CONT = UnivariateSpline(x2, cont2, s=0)
      
    
    #h: Cross-correlate
    corrlist = []
    for ordernum, order in enumerate(data):
      if process_model:
        left = numpy.searchsorted(model.x, order.x[0] - 10.0)
        right = numpy.searchsorted(model.x, order.x[-1] + 10.0)
        if left > 0:
          left -= 1

        #b: Make wavelength spacing constant
        model2 = DataStructures.xypoint(right - left + 1)
        model2.x = numpy.linspace(model.x[left], model.x[right], right - left + 1)
        model2.y = MODEL(model2.x)
        model2.cont = CONT(model2.x)

        #d: Rotationally broaden
        if vsini != None and vsini > 1.0*units.km.to(units.cm):
          model2 = RotBroad.Broaden(model2, vsini, linear=True)
        if debug:
          print "After rotational broadening"
          print model2.y
      
        #e: Convolve to detector resolution
        if resolution != None:
          model2 = FittingUtilities.ReduceResolution(model2.copy(), resolution, extend=False)
        if debug:
          print "After resolution decrease"
          print model.y

        #f: Rebin to the same spacing as the data
        xgrid = numpy.arange(model2.x[0], model2.x[-1], order.x[1] - order.x[0])
        modellong = FittingUtilities.RebinData(model2.copy(), xgrid)
        modelshort = FittingUtilities.RebinData(model2.copy(), order.x)
        if debug:
          print "After rebinning"
          print modelshort.y
          print modellong.y

      #Now, do the actual cross-correlation
      reducedshort = modelshort.y / modelshort.cont
      reducedlong = modellong.y / modellong.cont
      meanshort = numpy.mean(reducedshort)
      meanlong = numpy.mean(reducedlong)
      modelshort_rms = numpy.sqrt(numpy.sum((reducedshort-meanshort)**2))
      modellong_rms = numpy.sqrt(numpy.sum((reducedlong-meanlong)**2))
      left = numpy.searchsorted(modellong.x, modelshort.x[0])
      right = modellong.x.size - numpy.searchsorted(modellong.x, modelshort.x[-1])
      delta = left - right
      if debug:
        modellong.output("Corr_inputmodellong.dat")
        modelshort.output("Corr_inputmodelshort.dat")
    
      
      ycorr = numpy.correlate(reducedshort - meanshort, reducedlong - meanlong, mode=corr_mode)
      xcorr = numpy.arange(ycorr.size)
      if corr_mode == 'valid':
        lags = xcorr - (modellong.x.size + modelshort.x.size + delta - 1.0)/2.0
        lags = xcorr - right
      elif corr_mode == 'full':
        lags = xcorr - modellong.x.size
      else:
        sys.exit("Sorry! corr_mode = %s not supported yet!" %corr_mode)
      distancePerLag = modellong.x[1] - modellong.x[0]
      offsets = -lags*distancePerLag
      velocity = offsets*3e5 / numpy.median(modelshort.x)   
      corr = DataStructures.xypoint(velocity.size)
      corr.x = velocity[::-1]
      corr.y = ycorr[::-1]/(modelshort_rms*modellong_rms)
        
      #i: Only save part of the correlation
      left = numpy.searchsorted(corr.x, minvel)
      right = numpy.searchsorted(corr.x, maxvel)
      corr.x = corr.x[left:right]
      corr.y = corr.y[left:right]
        
      #j: Adjust correlation by fit, if user wants
      if normalize:
        mean = numpy.mean(corr.y)
        std = numpy.std(corr.y)
        corr.y = (corr.y - mean)/std

      #k: Save correlation
      corrlist.append(corr.copy())
      if debug:
        numpy.savetxt("%s.order%i" %(outfilename, ordernum+1), numpy.transpose((corr.x, corr.y)))
        #corr.output("%s.order%i" %(outfilename, ordernum+1))

    #Add up the individual CCFs
    master_corr = corrlist[0]
    for corr in corrlist[1:]:
      correlation = UnivariateSpline(corr.x, corr.y, s=0)
      master_corr.y += correlation(master_corr.x)

    #Finally, output
    if makefname:
      outfilename = "%s%s.%.0fkps_%sK%+.1f%+.1f" %(outdir, outfilebase, vsini*units.cm.to(units.km), star, gravity, metallicity)
    if save_output:
      print "Outputting to ", outfilename, "\n"
      numpy.savetxt(outfilename, numpy.transpose((master_corr.x, master_corr.y)), fmt="%.10g")
      returnlist.append(outfilename)
    else:
      returnlist.append(master_corr)

  return returnlist


