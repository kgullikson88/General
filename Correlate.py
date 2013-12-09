import subprocess
import matplotlib.pyplot as plt
import sys
import os
import numpy
from collections import defaultdict
from scipy.interpolate import UnivariateSpline
import scipy.signal
import DataStructures
#import Units
from astropy import units, constants
import MakeModel
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

    
"""
#This will do the correlation within python/numpy
#The combine keyword decides whether to combine the chips into a master cross-correlation 
#The normalize keyword decides whether to output as correlation power, or as significance
#The sigmaclip keyword decides whether to perform sigma-clipping on each chip before cross-correlating
#The nsigma keyword tells the program how many sigma to clip. This is ignored if sigmaclip = False
#The clip_order keyword tells what order polynomial to fit the flux to during sigma clipping. Ignored if sigmaclip = False
#The models keyword is a list of models to cross-correlate against (either filenames of two-column ascii files, or
#    each entry should be a list with the first element the x points, and the second element the y points
#The segments keyword controls which orders of the data to use, and which parts of them. Can be used to ignore telluric
#    contamination. Can be a string (default) which will use all of the orders, a list of integers which will
#    use all of the orders given in the list, or a dictionary of lists which gives the segments of each order to use.
#save_output determines whether the cross-correlation is saved to a file, or just returned
def PyCorr(filename, combine=True, normalize=False, sigmaclip=False, nsigma=3, clip_order=3, stars=star_list, temps=temp_list, models=model_list, gravities=gravity_list, metallicities=metallicity_list, corr_mode='valid', process_model=True, vsini=15*units.km.to(units.cm), resolution=100000, segments="all", save_output=True, outdir=outfiledir, outfilename=None, pause=0, debug=False):
  #1: Read in the datafile (if necessary)
  if type(filename) == str:
    chips = DataStructures.ReadGridSearchFile(filename)
  elif type(filename) == list and isinstance(filename[0], DataStructures.GridSearchOut):
    chips = list(filename)
    filename = "Output"
  elif type(filename) == list and isinstance(filename[0], DataStructures.xypoint):
    chips = [chip.ToGridSearchOut() for chip in filename]
    filename = "Output"
  elif isinstance(filename, DataStructures.xypoint):
    chips = [filename.ToGridSearchOut()]
    filename = "Output"
  else:
    sys.exit("Input data of unknown type: %s" %type(filename))

  makefname = False
  if outfilename == None:
    makefname = True
    
  #2: Interpolate data to a single constant wavelength grid in logspace
  maxsize = 0
  for chip in chips:
    if chip.size() > maxsize:
      maxsize = chip.size()
  data = DataStructures.xypoint(len(chips)*chips[min(1, len(chips)-1)].wave.size)
  data.x = numpy.linspace(numpy.log10(chips[0].wave[0]), numpy.log10(chips[-1].wave[-1]), data.x.size)
  data.y = numpy.ones(data.x.size)
  data.err = numpy.ones(data.x.size)
  data.cont = numpy.ones(data.cont.size)
  firstindex = 1e9
  lastindex = -1
  for i, chip in enumerate(chips):
    chip_sections = [[-1, 1e9],]
    #Use this order? Use all of it?
    if type(segments) != str:
      if type(segments) == list:
        for element in segments:
          if element == i+1:
            #Use all of this order
            break
      elif type(segments) == defaultdict or type(segments) == dict:
        try:
          chip_sections = segments[i+1]
        except KeyError:
          chip_sections = [[-1, -1],]
  
    #Sigma-clipping?
    if sigmaclip:
      done = False
      wave = chip.wave.copy()
      flux = chip.opt.copy()
      while not done:
        done = True
        fit = numpy.poly1d(numpy.polyfit(wave, flux, clip_order))
        residuals = flux - fit(wave)
        mean = numpy.mean(residuals)
        std = numpy.std(residuals)
        badindices = numpy.where(numpy.abs(residuals - mean) > nsigma*std)[0]
        flux[badindices] = fit(wave[badindices])
        if badindices.size > 0:
          done = False
      chip.opt = flux.copy()


    #Interpolate to constant wavelength grid (in log-space)
    left = numpy.searchsorted(data.x, numpy.log10(chip.wave[0]))
    right = numpy.searchsorted(data.x, numpy.log10(chip.wave[-1]))
    x = data.x[left:right].copy()
    OPT = UnivariateSpline(numpy.log10(chip.wave), chip.opt, s=0)
    OPTERR = UnivariateSpline(numpy.log10(chip.wave), chip.opterr, s=0)
    CONT = UnivariateSpline(numpy.log10(chip.wave), chip.cont, s=0)
    for section in chip_sections:
      left = numpy.searchsorted(chip.wave, section[0])
      right = numpy.searchsorted(chip.wave, section[1])
      if right == left:
        continue
      if right > 0:
        right -= 1
      
      left = numpy.searchsorted(data.x, numpy.log10(chip.wave[left]))
      right = numpy.searchsorted(data.x, numpy.log10(chip.wave[right]))
      if left < lastindex:
        #Take the average of the two overlapping orders
        data.y[lastindex:left] = (data.y[lastindex:left]/data.cont[lastindex:left] + OPT(data.x[lastindex:left])/CONT(data.x[lastindex:left]))/2.0
        left = lastindex
      data.y[left:right] = OPT(data.x[left:right])/CONT(data.x[left:right])
      data.err[left:right] = OPTERR(data.x[left:right])
      #data.cont[left:right] = CONT(data.x[left:right])
      firstindex = left

  

  #3: Begin loop over model spectra
  returnlist = []
  for i in range(len(models)):
    star = stars[i]
    temp = temps[i]
    gravity = gravities[i]
    metallicity = metallicities[i]
    modelfile = models[i]

    #a: Read in file
    if isinstance(modelfile, str):
      print "******************************\nReading file ", modelfile
      x,y = numpy.loadtxt(modelfile, usecols=(0,1), unpack=True)
      x *= units.angstrom.to(units.nm)
      y = 10**y
      cont = numpy.ones(y.size)*y.max()
    else:
      x = modelfile[0]
      y = modelfile[1]
      cont = numpy.ones(x.size)*y.max()
      if len(modelfile) > 2:
        cont = modelfile[2]
      if not process_model:
        model = DataStructures.xypoint(x=x, y=y, cont=cont)

    if process_model:
      left = numpy.searchsorted(x, 2*10**data.x[0] - 10**data.x[-1])
      right = numpy.searchsorted(x, 2*10**data.x[-1] - 10**data.x[0])
      if left > 0:
        left -= 1
      model = DataStructures.xypoint(right - left + 1)
      x2 = x[left:right].copy()
      y2 = y[left:right].copy()
      cont2 = FittingUtilities.Continuum(x2, y2, fitorder=5)
      MODEL = UnivariateSpline(x2,y2, s=0)
      CONT = UnivariateSpline(x2, cont2, s=0)
      if debug:
        print x2
        print y2
        print cont2

      #b: Make wavelength spacing constant
      model.x = numpy.linspace(x2[0], x2[-1], right - left + 1)
      model.y = MODEL(model.x)
      model.cont = CONT(model.x)

      #d: Rotationally broaden
      if vsini > 1.0*units.km.to(units.cm):
        model = RotBroad.Broaden(model, vsini, linear=True)
      if debug:
        print "After rotational broadening"
        print model.y

      #e: Convolve to detector resolution
      model = MakeModel.ReduceResolution(model.copy(), resolution, extend=False)
      if debug:
        print "After resolution decrease"
        print model.y
      
      #f: Convert to log-space
      MODEL = UnivariateSpline(model.x, model.y, s=0)
      CONT = UnivariateSpline(model.x, model.cont, s=0)
      model.x = numpy.linspace(numpy.log10(model.x[0]), numpy.log10(model.x[-1]), model.x.size)
      model.y = MODEL(10**model.x)
      model.cont = CONT(10**model.x)

      #g: Rebin to the same spacing as the data (but not the same pixels)
      xgrid = numpy.arange(model.x[0], model.x[-1], data.x[1] - data.x[0])
      model = MakeModel.RebinData(model.copy(), xgrid)
      if debug:
        print "After rebinning"
        print model.y

    #h: Cross-correlate
    data_rms = numpy.sqrt(numpy.sum((data.y/data.cont-1)**2))
    model_rms = numpy.sqrt(numpy.sum((model.y/model.cont-1)**2))
    left = numpy.searchsorted(model.x, data.x[0])
    right = model.x.size - numpy.searchsorted(model.x, data.x[-1])
    delta = left - right
    #print "left=%i\tright=%i\tdelta=%i" %(left, right, delta)
    #weights = WEIGHTS(data.x)
    #ycorr = numpy.correlate((data.y/data.cont-1.0), model.y/model.cont-1.0, mode=corr_mode)
    if debug:
      data.output("Corr_inputdata.dat")
      model.output("Corr_inputmodel.dat")

    
    ycorr = scipy.signal.fftconvolve((data.y/data.cont-1.0), (model.y/model.cont-1.0)[::-1], mode=corr_mode)
    xcorr = numpy.arange(ycorr.size)
    #print "data: %i\tmodel: %i\tcorr: %i" %(data.x.size, model.x.size, ycorr.size)
    if corr_mode == 'valid':
      lags = xcorr - (model.x.size + data.x.size + delta -1.0)/2.0
      lags = xcorr - (0 + right)
    elif corr_mode == 'full':
      lags = xcorr - model.x.size
    else:
      sys.exit("Sorry! corr_mode = %s not supported yet!" %corr_mode)
    distancePerLag = model.x[1] - model.x[0]
    offsets = -lags*distancePerLag
    velocity = offsets*3e5*numpy.log(10.0)   
    corr = DataStructures.xypoint(velocity.size)
    corr.x = velocity[::-1]
    corr.y = ycorr[::-1]/(data_rms*model_rms)
    #My version at home has a bug in numpy.correlate, reversing ycorr
    #BUG FIXED IN THE PYTHON VERSION I HAVE FOR LINUX MINT 13
    #if "linux" not in sys.platform:
    #  corr.y = corr.y[::-1]
        
    #i: Fit low-order polynomal to cross-correlation
    left = numpy.searchsorted(corr.x, minvel)
    right = numpy.searchsorted(corr.x, maxvel)
    vel = corr.x[left:right]
    corr = corr.y[left:right]
    #fit = numpy.poly1d(numpy.polyfit(vel, corr, 2))
        
    #j: Adjust correlation by fit
    #corr = corr - fit(vel)
    if normalize:
      mean = numpy.mean(corr)
      std = numpy.std(corr)
      corr = (corr - mean)/std

    #k: Finally, output
    if makefname:
      outfilename = outdir + filename.split("/")[-1] + ".%.0fkps_%sK%+.1f%+.1f" %(vsini*units.cm.to(units.km), star, gravity, metallicity)
    if save_output:
      print "Outputting to ", outfilename, "\n"
      numpy.savetxt(outfilename, numpy.transpose((vel, corr)), fmt="%.10g")
      returnlist.append(outfilename)
    else:
      returnlist.append((vel, corr))
    time.sleep(pause)
  return returnlist
"""





"""
   Function to make a cross-correlation function out of echelle data.
   Expects a list of xypoints as input, and various optional inputs.
"""
def PyCorr2(data, stars=star_list, temps=temp_list, models=model_list, model_fcns = None, gravities=gravity_list, metallicities=metallicity_list, corr_mode='valid', process_model=True, vsini=15*units.km.to(units.cm), resolution=100000, segments="all", save_output=True, outdir=outfiledir, outfilename=None, outfilebase="", debug=False):

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
        if vsini > 1.0*units.km.to(units.cm):
          model2 = RotBroad.Broaden(model2, vsini, linear=True)
          if debug:
            print "After rotational broadening"
      
        #e: Convolve to detector resolution
        model2 = MakeModel.ReduceResolution(model2.copy(), resolution, extend=False)
        if debug:
          print "After resolution decrease"

        #f: Rebin to the same spacing as the data
        logspacing = numpy.log(order.x[1]/order.x[0])
        start = numpy.log(model2.x[0])
        end = numpy.log(model2.x[-1])
        xgrid = numpy.exp(numpy.arange(start, end+logspacing, logspacing))
        #xgrid = numpy.arange(model2.x[0], model2.x[-1], order.x[1] - order.x[0])
        model2 = MakeModel.RebinData(model2.copy(), xgrid)
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
    if makefname:
      outfilename = "%s%s.%.0fkps_%sK%+.1f%+.1f" %(outdir, outfilebase, vsini*units.cm.to(units.km), star, gravity, metallicity)
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
        if vsini > 1.0*units.km.to(units.cm):
          model2 = RotBroad.Broaden(model2, vsini, linear=True)
        if debug:
          print "After rotational broadening"
          print model2.y
      
        #e: Convolve to detector resolution
        model2 = MakeModel.ReduceResolution(model2.copy(), resolution, extend=False)
        if debug:
          print "After resolution decrease"
          print model.y

        #f: Rebin to the same spacing as the data
        xgrid = numpy.arange(model2.x[0], model2.x[-1], order.x[1] - order.x[0])
        modellong = MakeModel.RebinData(model2.copy(), xgrid)
        modelshort = MakeModel.RebinData(model2.copy(), order.x)
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



  


def ReadFile(filename):
  infile = open(filename)
  lines = infile.readlines()
  infile.close()
  wave = []
  rect = []
  opt = []
  recterror = []
  opterror = []
  cont = []
  chips = []
  for line in lines:
    if not (line.startswith("#") or line == "\n"):
      try:
        wave.append(line.split()[0])
        rect.append(line.split()[1])
        opt.append(line.split()[2])
        recterror.append(line.split()[3])
        opterror.append(line.split()[4])
        cont.append(line.split()[5])
      except IndexError:
        print "Format incorrect for file: ", filename, "\nExitting"
        sys.exit(0)
    elif line == "\n" and len(wave) > 0:
      chip = Resid(len(wave))
      chip.wave = numpy.array(wave).astype(float)
      chip.rect = numpy.array(rect).astype(float)
      chip.opt = numpy.array(opt).astype(float)
      chip.recterr = numpy.array(recterror).astype(float)
      chip.opterr = numpy.array(opterror).astype(float)
      chip.cont = numpy.array(cont).astype(float)
      wave = []
      rect = []
      opt = []
      recterror = []
      opterror = []
      cont = []
      chips.append(chip)
  if len(wave) > 0:
    chip = Resid(len(wave))
    chip.wave = numpy.array(wave).astype(float)
    chip.rect = numpy.array(rect).astype(float)
    chip.opt = numpy.array(opt).astype(float)
    chip.recterr = numpy.array(recterror).astype(float)
    chip.opterr = numpy.array(opterror).astype(float)
    chip.cont = numpy.array(cont).astype(float)
    chips.append(chip)
  return chips



#This function takes an array of observations (each of which is an array of chips)
#It determines whether the first chip has much higher standard deviation, indicative of bad continuum
#If it does, we will not use any of the first chip
#If it is fine, we check to see if the average standard deviation in a given nod position
#  is much worse than the other. If so, prompt the user to decide whether we just use the good nod
#Finally, we will cut out all pixels with wavelength < 2285 nm (The CO bandhead starts at ~2290 nm)
#Returns a new array that satisfies the above conditions
def FindSectionsToUse(alldata, allnods, tol=1.15, cutoff=2290):
  #First, get the standard deviations from each chip of each observation
  sigmas = []
  for i in range(len(alldata)):
    data = alldata[i]
    stds = []
    for j in range(len(data)):
      stds.append(numpy.std(data[j].opt/data[j].cont)*numpy.sqrt(numpy.mean(data[j].cont)))
    sigmas.append(stds)

  #Make array for good indices
  all_good_indices = [[] for i in range(len(sigmas[0]))]
  for i in range(len(alldata[0])):
    for j in range(alldata[0][i].wave.size):
      all_good_indices[i].append(True)
  all_good_indices = numpy.array(all_good_indices)
  
  #Check if the average standard deviation for chip 1 is significantly worse than chips 2 and 3
  averages = []
  for j in range(len(sigmas[0])):
    mean = 0.0
    for i in range(len(sigmas)):
      mean = mean + sigmas[i][j]
    averages.append(mean/float(len(sigmas)))
  
  print averages
  if averages[0] > (averages[1] + averages[2])/2.0*tol:
    print "Chip 1 continuum might be bad!"
    print "ratio of standard deviations: ", averages[0]/((averages[1] + averages[2])/2.0)
    plt.plot(alldata[0][0].wave, alldata[0][0].opt)
    plt.plot(alldata[0][0].wave, alldata[0][0].cont)
    plt.show()
    inp = raw_input("Ignore chip 1 (y or n)? ")
    if "y" in inp:
      for j in range(all_good_indices[0].size):
        all_good_indices[0][j] = False

  #Next, see if the nod positions have significantly different 
  useAminusB = True
  useBminusA = True
  A_averages = []
  B_averages = []
  Aindex = 0
  Bindex = 0
  for i in range(len(sigmas)):
    nodpos = allnods[i]
    data = alldata[i]
    total = 0.0
    for j in range(len(data)):
      total = total + sigmas[i][j]
    if nodpos[0] == "A":
      A_averages.append(total/float(len(data)))
      Aindex = i
    elif nodpos[0] == "B":
      B_averages.append(total/float(len(data)))
      Bindex = i

  if numpy.mean(B_averages)/numpy.mean(A_averages) > tol:
    print "Nod position B is much noisier than nod position A. Ratio = ", numpy.mean(B_averages)/numpy.mean(A_averages)
    for i in range(3):
      plt.plot(alldata[Aindex][i].wave, alldata[Aindex][i].opt/alldata[Aindex][i].cont, 'b-', label="Nod A")
      plt.plot(alldata[Bindex][i].wave, alldata[Bindex][i].opt/alldata[Bindex][i].cont - 0.1, 'k-', label="Nod B")
    #plt.legend(loc='best')
    plt.title("Blue: Nod A. Black: Nod B")
    plt.show()
    inp = raw_input("Use only A nod positions? (y or n)? ")
    if "y" in inp:
      useBminusA = False
  elif numpy.mean(A_averages)/numpy.mean(B_averages) > tol:
    print "Nod position A is much noisier than nod position B. Ratio = ", numpy.mean(A_averages)/numpy.mean(B_averages)
    for i in range(3):
      plt.plot(alldata[Aindex][i].wave, alldata[Aindex][i].opt/alldata[Aindex][i].cont, 'b-', label="Nod A")
      plt.plot(alldata[Bindex][i].wave, alldata[Bindex][i].opt/alldata[Bindex][i].cont - 0.1, 'k-', label="Nod B")
    #plt.legend(loc='best')
    plt.title("Blue: Nod A. Black: Nod B")
    plt.show()
    inp = raw_input("Use only B nod positions? (y or n)? ")
    if "y" in inp:
      useAminusB = False

  #Make a new array for all of the data, which satisfies all of the above conditions and has wavelength > cutoff
  new_data = list(alldata)
  for i in range(len(new_data)-1, -1, -1):
    nodpos = allnods[i]
    data = new_data[i]
    if (nodpos[0] == "A" and useAminusB) or (nodpos[0] == "B" and useBminusA):
      for j in range(len(data)):
        for k in range(data[j].wave.size-1, -1, -1):
          if not all_good_indices[j][k] or data[j].wave[k] < cutoff:
            #print "Deleting point at ", data[j].wave[k]
            data[j].wave = numpy.delete(data[j].wave, k)
            data[j].rect = numpy.delete(data[j].rect, k)
            data[j].opt = numpy.delete(data[j].opt, k)
            data[j].recterr = numpy.delete(data[j].recterr, k)
            data[j].opterr = numpy.delete(data[j].opterr, k)
            data[j].cont = numpy.delete(data[j].cont, k)
      for j in range(len(data)-1, -1, -1):
        if data[j].wave.size < 10:
          del data[j]
    else:
      print "Deleting nod position"
      del new_data[i]
    

    
  print "\n", len(new_data), "observations"
  for i in range(len(new_data)):
    print len(new_data[i]), "chips in observation ", i+1
    for j in range(len(new_data[i])):
      print new_data[i][j].wave.size, "pixels in chip ", j+1
    print "\n"

  return new_data



if __name__ == "__main__":
  if len(sys.argv) > 1:
    for fname in sys.argv[1:]:
      for vel in [10.0, 20.0, 30.0, 40.0, 50.0]:
        PyCorr(fname, vsini=vel*units.km.to(units.cm)) #, combine=False, sigmaclip=False)
  else:
    #Make output file basename
    directory = os.getcwd()
    subdirs = directory.split("/")
    star = subdirs[-2]
    date = subdirs[-1]
    outfilebase = star + "_" + date + "_"

    #Find all files that start with "Corrected_"
    allfiles = os.listdir("./")
    corrected = defaultdict(list)
    endstring = "-0.dat"
    for fname in allfiles:
      if fname.startswith("Corrected_") and fname.endswith(endstring) and len(fname.split("_")) == 4:
        #Find wavelength:
        wlen = float(fname.split("_")[-1].split(endstring)[0])
        print fname, wlen
        if wlen in corrected:
          corrected[wlen].append(fname)
        else:
          temp = []
          temp.append(fname)
          corrected[wlen] = temp

    #Combine all the files in a given wavelength setting
    for wlen in sorted(corrected.keys()):
      outfilename = outfilebase + str(wlen) + "-1.dat"
      data = []
      sigma = []
      nods = []
      for fname in corrected[wlen]:
        data.append(ReadFile(fname))
        print fname, "\t", len(data[-1])
        segments = fname.split("_")
        nods.append(segments[1][0] + "-" + segments[2][0])
      data = FindSectionsToUse(data, nods)
      
      master = data[0]
      #sigma.append(numpy.std(data[0].opt/data[0].cont)*numpy.sqrt(numpy.mean(data[0].opt)))
      normalization = [0.0,0.0,0.0,0.0]
      for i in range(len(master)):
        weight = 1.0/numpy.std(master[i].opt/master[i].cont)**2
        normalization[i] = normalization[i] + weight
        print "weight = ", weight 
        master[i].opt = master[i].opt*weight
        master[i].opterr = master[i].opt*weight
        master[i].cont = master[i].cont*weight
      index = 1
    
    
      for nodpos in data[1:]:
        for i in range(len(nodpos)):
          chip = nodpos[i]
          weight = 1.0/numpy.std(chip.opt/chip.cont)**2
          normalization[i] = normalization[i] + weight
          print "weight = ", weight
          OPT = UnivariateSpline(chip.wave, chip.opt, s=0)
          CONT = UnivariateSpline(chip.wave, chip.cont, s=0)
          ERR = UnivariateSpline(chip.wave, chip.opterr, s=0)
          master[i].opt = master[i].opt + OPT(master[i].wave)*weight
          master[i].opterr = master[i].opterr + ERR(master[i].wave)*weight
          master[i].cont = master[i].cont + CONT(master[i].wave)*weight
          #plt.plot(master[i].wave, OPT(master[i].wave))
        index = index + 1
      for i in range(len(master)):
        chip = master[i]
        chip.opt = chip.opt/(float(len(data))*normalization[i])
        chip.err = chip.opterr/(float(len(data))*normalization[i])
        chip.cont = chip.cont/(float(len(data))*normalization[i])
        master[i] = chip
    
      #Output. Do a sigma-clipping algorithm as we go
      print "outputting to ", outfilename
      nsigma = 3
      order = 5
      maxiters = 100
      outfile = open(outfilename, "w")
      for chip in master:
        """
        #1: sigma-clipping algorithm
        done = False
        index = 0
        while not done:
          done = True
          wave_mean = numpy.mean(chip.wave)
          fit = numpy.poly1d(numpy.polyfit(chip.wave-wave_mean, chip.opt, order))
          residuals = chip.opt - fit(chip.wave-wave_mean)
          mean = numpy.mean(residuals)
          std = numpy.std(residuals)
          badindices = numpy.where(numpy.abs(residuals - mean) > nsigma*std)[0]
          if badindices.size > 0:
            chip.opt[badindices] = chip.cont[badindices]
            done = False
          index = index + 1
          #include a maximum number of iterations
          if index > maxiters:
            done = True
        chip.cont = fit(chip.wave-wave_mean)
        """
        fit = numpy.poly1d(numpy.polyfit(chip.wave-chip.wave.mean(), chip.opt, order))
        chip.cont = fit(chip.wave - chip.wave.mean())
        
        #2: output
        for i in range(chip.wave.size):
          outfile.write("%.10g\t" %chip.wave[i] + "%.10g\t" %chip.rect[i] + "%.10g\t" %chip.opt[i] + "%.10g\t" %chip.recterr[i] + "%.10g\t" %chip.opterr[i] + "%.10g\n" %chip.cont[i])
        outfile.write("\n\n\n\n")
      outfile.close()
    
    
      #Finally, cross-correlate the new file with a suite of model atmospheres
      #Corr(outfilename)
      PyCorr(outfilename)
