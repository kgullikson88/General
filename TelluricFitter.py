"""
This module provides the 'TelluricFitter' class, used
to fit the telluric lines in data.

Usage:
  - Initialize fitter: fitter = TelluricFitter()
  - Define variables to fit: must provide a dictionary where
      the key is the name of the variable, and the value is
      the initial guess value for that variable.
      Example: fitter.FitVariable({"ch4": 1.6, "h2o": 45.0})
  - Edit values of constant parameters: similar to FitVariable,
      but the variables given here will not be fit. Useful for 
      settings things like the telescope pointing angle, temperature,
      and pressure, which will be very well-known.
      Example: fitter.AdjustValue({"angle": 50.6})
  - Set bounds on fitted variables (fitter.SetBounds): Give a dictionary
      where the key is the name of the variable, and the value is
      a list of size 2 of the form [lower_bound, upper_bound]
  - Import data (fitter.ImportData): Copy data as a class variable.
      Must be given as a DataStructures.xypoint instance
  - Perform the fit: (fitter.Fit): no arguments given... for now
      Returns a DataStructures.xypoint instance of the model. The 
      x-values in the returned array should be the same as the data.
   - Optional: retrieve a new version of the data, which is 
      wavelength-calibrated using the telluric lines with 
      data2 = fitter.data  


"""


import pylab
import pyfits
import numpy
import sys
import os
import subprocess
import scipy
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import leastsq, brute, fmin
from scipy.linalg import svd, diagsvd
from scipy import mat
import MakeModel
import DataStructures
import FindContinuum
from FittingUtilities import *
import FitBstar


class TelluricFitter:
  def __init__(self):
    #Set up parameters
    self.parnames = ["pressure", "temperature", "angle", "resolution", "wavestart", "waveend",
                     "h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "no",
                     "so2", "no2", "nh3", "hno3"]
    self.const_pars = [795.0, 273.0, 45.0, 50000.0, 2200.0, 2400.0,
                       50.0, 368.5, 3.9e-2, 0.32, 0.14, 1.8, 2.1e5, 1.1e-19,
                       1e-4, 1e-4, 1e-4, 5.6e-4]
    self.bounds = [[] for par in self.parnames]
    self.fitting = [False]*len(self.parnames)
    self.data = None
    self.resolution_bounds = [10000.0, 100000.0]

    homedir = os.environ['HOME']
    self.LineListFile = homedir + "/School/Research/Useful_Datafiles/Linelist2.dat"


  """
    Display the value of each of the parameters, and show whether it is being fit or not
  """
  def DisplayVariables(self, fitonly=False):
    print "%.15s\tValue\t\tFitting?\tBounds" %("Parameter".ljust(15))
    print "-------------\t-----\t\t-----\t\t-----"
    for i in range(len(self.parnames)):
      if (fitonly and self.fitting[i]) or not fitonly:
        if len(self.bounds[i]) == 2:
          print "%.15s\t%.5E\t%s\t\t%g - %g" %(self.parnames[i].ljust(15), self.const_pars[i], self.fitting[i], self.bounds[i][0], self.bounds[i][1])
        else:
          print "%.15s\t%.5g\t\t%s" %(self.parnames[i].ljust(15), self.const_pars[i], self.fitting[i])


  """
    Add one or more variables to the list being fit. vardict must be a dictionary
      where the key is the parameter name and the value is the value of that parameter.
  """
  def FitVariable(self, vardict):
    for par in vardict.keys():
      try:
        idx = self.parnames.index(par)
        self.const_pars[idx] = vardict[par]
        self.fitting[idx] = True
      except ValueError:
        print "Error! Bad parameter name given. Currently available are: "
        self.DisplayVariables()

  """
    Similar to FitVariable, but this just adjust the value of a constant value
  """
  def AdjustValue(self, vardict):
    for par in vardict.keys():
      try:
        idx = self.parnames.index(par)
        self.const_pars[idx] = vardict[par]
      except ValueError:
        print "Error! Bad parameter name given. Currently available are: "
        self.DisplayVariables()

  """
    Similar to FitVariable, but it sets bounds on the variable. This caqn technically
      be done for any variable, but is only useful to set bounds for those variables
      being fit
  """
  def SetBounds(self, bounddict):
    for par in bounddict.keys():
      try:
        idx = self.parnames.index(par)
        self.bounds[idx] = bounddict[par]
        if par == "resolution":
          self.resolution_bounds = bounddict[par]
      except ValueError:
        print "Error! Bad parameter name given. Currently available are: "
        self.DisplayVariables()
    

  """
    Function for the user to give the data. The data should be in the form of
      a DataStructures.xypoint structure.
  """
  def ImportData(self, data):
    self.data = data.copy()


  """
    Edit the location of the telluric line list. These lines are used to improve
      the wavelength calibration of the data, so they should be strong, unblended lines.
      Wavelengths in nanometers
  """
  def SetTelluricLineListFile(self, fname):
    self.LineListFile = fname



  """
    Finally, the main fitting function. Before calling this, the user MUST
      1: call FitVariable at least once, specifying which variables will be fit
      2: import data into the class using the ImportData method.
    resolution_fit_mode controls which function is used to estimate the resolution.
      3: Set resolution bounds (any other bounds are optional)
      "SVD" is for singlular value decomposition, while "gauss" is for convolving with a gaussian
      (and fitting the width of the guassian to give the best fit)

    continuum_fit_mode controls how the continuum is fit in the data. Choices are 'polynomial' and 'smooth'

    fit_primary determines whether an iterative smoothing is applied to the data to approximate the primary star (only works for primary stars with broad lines)

    adjust_wave can be set to either 'data' or 'model'. To wavelength calibrate the data to the telluric lines, set to 'data'. If you think the wavelength calibration is good on the data (such as Th-Ar lines in the optical), then set to 'model'
  """
  def Fit(self, resolution_fit_mode="SVD", fit_primary=False, adjust_wave="data"):
    print "Fitting now!"
    self.resolution_fit_mode=resolution_fit_mode
    self.fit_primary = fit_primary
    self.adjust_wave = adjust_wave

    #Make fitpars array
    fitpars = [self.const_pars[i] for i in range(len(self.parnames)) if self.fitting[i] ]
    if len(fitpars) < 1:
      print "Error! Must fit at least one variable!"
      return

    if self.data == None:
      print "Error! Must supply data to fit!"
      return

    idx = self.parnames.index("resolution")
    if len(self.bounds[idx]) < 2 and self.resolution_fit_mode != "SVD":
      print "Must give resolution bounds!"
      inp = raw_input("Enter the lowest and highest possible resolution, separated by a space: ")
      self.resolution_bounds = [float(inp.split()[0]), float(inp.split()[1])]
    

    #Read in line list:
    linelist = numpy.loadtxt(self.LineListFile)

    #Set up the fitting logfile
    outfile = open("chisq_summary.dat", "a")
    outfile.write("\n\n\n\n")
    for i in range(len(self.parnames)):
      if self.fitting[i]:
        outfile.write("%s\t" %self.parnames[i])
    outfile.write("\n")
    outfile.close()

    #Perform the fit
    self.first_iteration = True
    fitout = leastsq(self.FitErrorFunction, fitpars, args=(linelist, ), full_output=True, epsfcn = 0.0005, maxfev=1000)
    fitpars = fitout[0]
        
    if self.fit_primary:
      return self.GenerateModel(fitpars, linelist, separate_primary=True)
    else:
      return self.GenerateModel(fitpars, linelist)
    


  def FitErrorFunction(self, fitpars, linelist):
    model = self.GenerateModel(fitpars, linelist)
    outfile = open("chisq_summary.dat", 'a')
    weights = 1.0/self.data.err
    weights = weights/weights.sum()
    return_array = (self.data.y  - self.data.cont*model.y)**2*weights
    #Evaluate bound conditions
    fit_idx = 0
    for i in range(len(self.bounds)):
      if self.fitting[i]:
        if len(self.bounds[i]) == 2:
          return_array += bound(self.bounds[i], fitpars[fit_idx])
        outfile.write("%.5g\t" %fitpars[fit_idx])
        fit_idx += 1
      elif len(self.bounds[i]) == 2 and self.parnames[i] != "resolution":
        print "Warning! A constant parameter is bounded!"
        return_array += bound(self.bounds[i], self.const_pars[i])
  
    print "X^2 = ", numpy.sum(return_array)/float(weights.size)
    outfile.write("\n")
    outfile.close()
    return return_array



  def GenerateModel(self, pars, linelist, nofit=False, separate_primary=False):
    data = self.data
    #Update self.const_pars to include the new values in fitpars
    fit_idx = 0
    for i in range(len(self.parnames)):
      if self.fitting[i]:
        self.const_pars[i] = pars[fit_idx]
        fit_idx += 1
    self.DisplayVariables(fitonly=True)
    
    #Extract parameters from pars and const_pars. They will have variable
    #  names set from self.parnames
    fit_idx = 0
    for i in range(len(self.parnames)):
      if self.fitting[i]:
        exec("%s = %g" %(self.parnames[i], pars[fit_idx]))
        fit_idx += 1
      else:
        exec("%s = %g" %(self.parnames[i], self.const_pars[i]))
    wavenum_start = 1e7/waveend
    wavenum_end = 1e7/wavestart
    
    #Make sure certain variables are positive
    if co < 0:
      co = 0
      print "\nWarning! CO was set to be negative. Resetting to zero before generating model!\n\n"
    if ch4 < 0:
      ch4 = 0
      print "\nWarning! CH4 was set to be negative. Resetting to zero before generating model!\n\n"
    if h2o < 0:
      humidity = 0
      print "\nWarning! Humidity was set to be negative. Resetting to zero before generating model!\n\n"
    if angle < 0:
      angle = -angle
      print "\nWarning! Angle was set to be negative. Resetting to a positive value before generating model!\n\n"

    #Generate the model:
    model = MakeModel.Main(pressure, temperature, wavenum_start, wavenum_end, angle, h2o, co2, o3, n2o, co, ch4, o2, no, so2, no2, nh3, hno3, wavegrid=self.data.x, resolution=resolution)

    model_original = model.copy()
  
    #Reduce to initial guess resolution
    if "SVD" in self.resolution_fit_mode and not self.first_iteration:
      broadening_fcn = self.broadstuff[0]
      xarr = self.broadstuff[1]
      Model = UnivariateSpline(model_original.x, model_original.y, s=0)
      model_new = Model(xarr)
      model = DataStructures.xypoint(x=xarr)
      Broadened = UnivariateSpline(xarr, numpy.convolve(model_new, broadening_fcn, mode="same"),s=0)
      model.y = Broadened(model.x)
      model = MakeModel.RebinData(model, data.x)
      
    elif "gauss" in self.resolution_fit_mode or self.first_iteration:
      Continuum = UnivariateSpline(data.x.copy(), data.cont.copy(), s=0)
      if (resolution - 10 < self.resolution_bounds[0] or resolution+10 > self.resolution_bounds[1]):
        resolution = numpy.mean(self.resolution_bounds)
      model = MakeModel.ReduceResolution(model.copy(), resolution, Continuum)
      model = MakeModel.RebinData(model.copy(), data.x.copy())

    if nofit:
      return model
     

    shift = self.CCImprove(data, model)
    if self.adjust_wave == "data":
      data.x += shift
    elif self.adjust_wave == "model":
      model_original.x -= shift
    else:
      sys.exit("Error! adjust_wave parameter set to invalid value: %s" %self.adjust_wave)
      
    if "SVD" in self.resolution_fit_mode and not self.first_iteration:
      Model = UnivariateSpline(model_original.x, model_original.y, s=0)
      model_new = Model(xarr)
      model = DataStructures.xypoint(x=xarr)
      Broadened = UnivariateSpline(xarr, numpy.convolve(model_new, broadening_fcn, mode="same"),s=0)
      model.y = Broadened(model.x)
      model = MakeModel.RebinData(model, data.x)
      
    elif "gauss" in self.resolution_fit_mode or self.first_iteration:
      model = MakeModel.ReduceResolution(model_original.copy(), resolution, Continuum)
      model = MakeModel.RebinData(model.copy(), data.x.copy())

    model.y[model.y < 0.05] = (data.y/data.cont)[model.y < 0.05]
    resid = data.y/model.y
    if self.fit_primary:
      data2 = data.copy()
      data2.y /= model.y
      primary_star = FitBstar.GetApproximateSpectrum(data2, bcwidth=100)
      primary_star.y /= primary_star.y.mean()
      PRIMARY_STAR = UnivariateSpline(primary_star.x, primary_star.y, s=0)

      model2 = model.copy()
      model2.y *= primary_star.y
      resid /= primary_star.y
    
    #data.cont = FindContinuum.Continuum(data.x, resid, fitorder=3, lowreject=3, highreject=3)
    data.cont = FindContinuum.Continuum(data.x, resid, fitorder=9, lowreject=2, highreject=2)
    
    if self.fit_primary:
      modelfcn, mean = self.FitWavelength(data, model2.copy(), linelist)
    else:
      modelfcn, mean = self.FitWavelength(data, model.copy(), linelist)
    if self.adjust_wave == "data":
      test = modelfcn(data.x - mean)
      xdiff = [test[j] - test[j-1] for j in range(1, len(test)-1)]
      if min(xdiff) > 0 and numpy.mean(test - data.x) < 0.5:
        data.x = test.copy()
      else:
        print "Warning! Wavelength calibration did not succeed!"
    elif self.adjust_wave == "model":
      test = modelfcn(model.x - mean)
      xdiff = [test[j] - test[j-1] for j in range(1, len(test)-1)]
      if min(xdiff) > 0 and numpy.mean(test - model.x) < 0.5:
        model.x = test.copy()
        model_original.x = modelfcn(model_original.x - mean)
      else:
        print "Warning! Wavelength calibration did not succeed!"
    else:
      sys.exit("Error! adjust_wave set to an invalid value: %s" %self.adjust_wave)

    #Fit resolution
    done = False
    while not done:
      done = True
      if "SVD" in self.resolution_fit_mode:
        if self.fit_primary:
          model2 = model_original.copy()
          prim = PRIMARY_STAR(model2.x)
          prim[prim < 0.0] = 0.0
          prim[prim > 10.0] = 10.0
          model2.y *= prim
          model, self.broadstuff = self.Broaden(data.copy(), model2, full_output=True)
        else:
          model, self.broadstuff = self.Broaden(data.copy(), model_original.copy(), full_output=True)
      elif "gauss" in self.resolution_fit_mode:
        if self.fit_primary:
          model2 = model_original.copy()
          prim = PRIMARY_STAR(model2.x)
          prim[prim < 0.0] = 0.0
          prim[prim > 10.0] = 10.0
          model2.y *= prim
          model, resolution = self.FitResolution(data.copy(), model_original.copy(), resolution)
        else:
          model, resolution = self.FitResolution(data.copy(), model_original.copy(), resolution)
        #Save resolution
        idx = self.parnames.index("resolution")
        self.const_pars[idx] = resolution
      else:
        done = False
        print "Resolution fit mode set to an invalid value: %s" %self.resolution_fit_mode
        self.resolution_fit_mode = raw_input("Enter a valid mode (SVD or guass): ")
    
    self.data = data
    self.first_iteration = False
    if separate_primary:
      primary = model.copy()
      primary.y = PRIMARY_STAR(primary.x)
      model.y /= primary.y
      return primary, model
    else:
      return model


  
  #Wavelength-fitting function that just shifts lines, instead of fitting them to gaussians
  def WavelengthErrorFunction(self, shift, data, model):
    modelfcn = UnivariateSpline(model.x, model.y, s=0)
    weight = 1e9 * numpy.ones(data.x.size)
    weight[data.y > 0] = 1.0/numpy.sqrt(data.y[data.y > 0])
    weight[weight < 0.01] = 0.0
    newmodel = modelfcn(model.x + float(shift))
    if shift < 0:
      newmodel[model.x - float(shift) < model.x[0]] = 0
    else:
      newmodel[model.x - float(shift) > model.x[-1]] = 0
    returnvec = (data.y - newmodel)**2*weight
    return returnvec

  #Gaussian absorption line
  def GaussianFitFunction(self, x,params):
    cont = params[0]
    depth = params[1]
    mu = params[2]
    sig = params[3]
    return cont - depth*numpy.exp(-(x-mu)**2/(2*sig**2))

  #Returns the residuals between the fit from above and the actual values
  def GaussianErrorFunction(self, params, x, y):
    return self.GaussianFitFunction(x,params) - y
  

  def FitWavelength(self, data_original, telluric, linelist, tol=0.05, oversampling=4, debug=False):
    old = []
    new = []

    #Interpolate to finer spacing
    DATA_FCN = UnivariateSpline(data_original.x, data_original.y, s=0)
    CONT_FCN = UnivariateSpline(data_original.x, data_original.cont, s=0)
    MODEL_FCN = UnivariateSpline(telluric.x, telluric.y, s=0)
    data = DataStructures.xypoint(data_original.x.size*oversampling)
    data.x = numpy.linspace(data_original.x[0], data_original.x[-1], data_original.x.size*oversampling)
    data.y = DATA_FCN(data.x)
    data.cont = CONT_FCN(data.x)
    model = DataStructures.xypoint(data.x.size)
    model.x = numpy.copy(data.x)
    model.y = MODEL_FCN(model.x)

    if debug:
      print linelist
      print data.x
  
    #Begin loop over the lines
    numlines = 0
    for line in linelist:
      if line-tol > data.x[0] and line+tol < data.x[-1]:
        numlines += 1
        #Find line in the model
        left = numpy.searchsorted(model.x, line - tol)
        right = numpy.searchsorted(model.x, line + tol)
        minindex = model.y[left:right].argmin() + left

        mean = model.x[minindex]
        left2 = numpy.searchsorted(model.x, mean - tol*2)
        right2 = numpy.searchsorted(model.x, mean + tol*2)

        argmodel = DataStructures.xypoint(right2 - left2)
        argmodel.x = numpy.copy(model.x[left2:right2])
        argmodel.y = numpy.copy(model.y[left2:right2])

        #Do the same for the data
        left = numpy.searchsorted(data.x, line - tol)
        right = numpy.searchsorted(data.x, line + tol)
        minindex = data.y[left:right].argmin() + left

        mean = data.x[minindex]

        argdata = DataStructures.xypoint(right2 - left2)
        argdata.x = numpy.copy(data.x[left2:right2])
        argdata.y = numpy.copy(data.y[left2:right2]/data.cont[left2:right2])

        #Fit argdata to gaussian:
        cont = 1.0
        depth = cont - argdata.y[argdata.y.size/2]
        mu = argdata.x[argdata.x.size/2]
        sig = 0.025
        params = [cont, depth, mu, sig]
        params,success = leastsq(self.GaussianErrorFunction, params, args=(argdata.x, argdata.y))
      
        mean = params[2]
        #Do a cross-correlation first, to get the wavelength solution close
        ycorr = scipy.correlate(argdata.y-1.0, argmodel.y-1.0, mode="full")
        xcorr = numpy.arange(ycorr.size)
        maxindex = ycorr.argmax()
        lags = xcorr - (argdata.x.size-1)
        distancePerLag = (argdata.x[-1] - argdata.x[0])/float(argdata.x.size)
        offsets = -lags*distancePerLag
        shift = offsets[maxindex]
        shift, success = leastsq(self.WavelengthErrorFunction, shift, args=(argdata, argmodel))
        if (debug):
          print argdata.x[0], argdata.x[-1], argdata.x.size
          print "wave: ", mean, "\tshift: ", shift, "\tsuccess = ", success
          pylab.plot(model.x[left:right]-shift, model.y[left:right])
          pylab.plot(argmodel.x, argmodel.y)
          pylab.plot(argdata.x, argdata.y)
        if (success < 5):
          old.append(mean)
          if self.adjust_wave == "data":
            new.append(mean + float(shift))
          elif self.adjust_wave == "model":
            new.append(mean - float(shift))
          else:
            sys.exit("Error! adjust_wave set to an invalid value: %s" %self.adjust_wave)
    if debug:
      pylab.show()
      pylab.plot(old, new, 'ro')
      pylab.show()
    #Iteratively fit to a cubic with sigma-clipping
    order = 3
    if numlines < order:
      fit = lambda x: x
      mean = 0.0
      return fit, mean
    done = False
    while not done:
      done = True
      mean = numpy.mean(old)
      fit = numpy.poly1d(numpy.polyfit(old - mean, new, order))
      residuals = fit(old - mean) - new
      std = numpy.std(residuals)
      badindices = numpy.where(numpy.logical_or(residuals > 2*std, residuals < -2*std))[0]
      for badindex in badindices[::-1]:
        del old[badindex]
        del new[badindex]
        done = False
    if debug:
      pylab.plot(old, fit(old - mean) - new, 'ro')
      pylab.show()
    return fit, mean


  """
    Improve the wavelength solution by a constant shift
  """
  def CCImprove(self, data, model, be_safe=True, tol=0.5):
    ycorr = scipy.correlate(data.y/data.cont-1.0, model.y-1.0, mode="full")
    xcorr = numpy.arange(ycorr.size)
    maxindex = ycorr.argmax()
    lags = xcorr - (data.y.size-1)
    distancePerLag = (data.x[-1] - data.x[0])/float(data.x.size)
    offsets = -lags*distancePerLag
    print "maximum offset: ", offsets[maxindex], " nm"

    if numpy.abs(offsets[maxindex]) < tol or not be_safe:
      #Apply offset
      print "Applying offset"
      return offsets[maxindex]
    else:
      return 0.0


  #Function to fit the resolution
  def FitResolution(self, data, model, resolution=75000, debug = False):
    ####resolution is the initial guess####
    #Interpolate to a constant wavelength grid
    print "Fitting resolution"
    if debug:
      pylab.plot(data.x, data.y/data.cont)
      pylab.plot(model.x, model.y)
      pylab.show()
    ModelFcn = UnivariateSpline(model.x, model.y, s=0)
    newmodel = DataStructures.xypoint(model.x.size)
    newmodel.x = numpy.linspace(model.x[0], model.x[-1], model.x.size)
    newmodel.y = ModelFcn(newmodel.x)

    if debug:
      pylab.plot(data.x, data.y/data.cont)
      pylab.plot(newmodel.x, newmodel.y)
      pylab.show()

    Continuum = UnivariateSpline(data.x, data.cont, s=0)

    #Do a brute force grid search first, then refine with Levenberg-Marquardt
    searchgrid = (self.resolution_bounds[0], self.resolution_bounds[1], 5000)
    ResolutionFitErrorBrute = lambda resolution, data, model, cont_fcn: numpy.sum(self.ResolutionFitError(resolution, data, model, cont_fcn))
    resolution = brute(ResolutionFitErrorBrute,(searchgrid,), args=(data,newmodel,Continuum))
    resolution, success = leastsq(self.ResolutionFitError, resolution, args=(data, newmodel, Continuum), epsfcn=10)
    print "Optimal resolution found at R = ", float(resolution)
    newmodel = MakeModel.ReduceResolution(newmodel, float(resolution), Continuum)
    return MakeModel.RebinData(newmodel, data.x), float(resolution)
  
  
  def ResolutionFitError(self, resolution, data, model, cont_fcn):
    newmodel = MakeModel.ReduceResolution(model, resolution, cont_fcn)
    newmodel = MakeModel.RebinData(newmodel, data.x)
    weights = 1.0/data.err
    weights = weights/weights.sum()
    returnvec = (data.y - data.cont*newmodel.y)**2*weights + bound(self.resolution_bounds, resolution)
    print "Resolution-fitting X^2 = ", numpy.sum(returnvec)/float(weights.size), "at R = ", resolution
    if numpy.isnan(numpy.sum(returnvec**2)):
      print "Error! NaN found in ResolutionFitError!"
      outfile=open("ResolutionFitError.log", "a")
      for i in range(data.y.size):
        outfile.write("%.10g\t" %data.x[i] + "%.10g\t" %data.y[i] + "%.10g\t" %data.cont[i] + "%.10g\t" %newmodel.x[i] + "%.10g\n" %newmodel.y[i])
      outfile.write("\n\n\n\n")
      outfile.close()
    return returnvec

  
#  def ResolutionFitErrorBrute(resolution, data, model, cont_fcn):
#    return numpy.sum(ResolutionFitError(resolution, data, model, cont_fcn))


  #Fits the broadening profile using singular value decomposition
  #oversampling is the oversampling factor to use before doing the SVD
  #m is the size of the broadening function, in oversampled units
  #dimension is the number of eigenvalues to keep in the broadening function. (Keeping too many starts fitting noise)
  def Broaden(self, data, model, oversampling = 5, m = 101, dimension = 15, full_output=False):
    n = data.x.size*oversampling
  
    #resample data
    Spectrum = UnivariateSpline(data.x, data.y/data.cont, s=0)
    Model = UnivariateSpline(model.x, model.y, s=0)
    xnew = numpy.linspace(data.x[0], data.x[-1], n)
    ynew = Spectrum(xnew)
    model_new = MakeModel.RebinData(model, xnew).y

    #Make 'design matrix'
    design = numpy.zeros((n-m,m))
    for j in range(m):
      for i in range(m/2,n-m/2-1):
        design[i-m/2,j] = model_new[i-j+m/2]
    design = mat(design)
    
    #Do Singular Value Decomposition
    try:
      U,W,V_t = svd(design, full_matrices=False)
      outfilename = "SVD_Error2.log"
      outfile = open(outfilename, "a")
      numpy.savetxt(outfile, numpy.transpose((data.x, data.y, data.cont)))
      outfile.write("\n\n\n\n\n")
      numpy.savetxt(outfile, numpy.transpose((model.x, model.y, model.cont)))
      outfile.write("\n\n\n\n\n")
      outfile.close()
    except numpy.linalg.linalg.LinAlgError:
      outfilename = "SVD_Error.log"
      outfile = open(outfilename, "a")
      numpy.savetxt(outfile, numpy.transpose((data.x, data.y, data.cont)))
      outfile.write("\n\n\n\n\n")
      numpy.savetxt(outfile, numpy.transpose((model.x, model.y, model.cont)))
      outfile.write("\n\n\n\n\n")
      outfile.close()
      sys.exit("SVD did not converge! Outputting data to %s" %outfilename)
      
      
    
    #Invert matrices:
    #   U, V are orthonormal, so inversion is just their transposes
    #   W is a diagonal matrix, so its inverse is 1/W
    W1 = 1.0/W
    U_t = numpy.transpose(U)
    V = numpy.transpose(V_t)
  
    #Remove the smaller values of W
    W1[dimension:] = 0
    W2 = diagsvd(W1,m,m)
    
    #Solve for the broadening function
    spec = numpy.transpose(mat(ynew[m/2:n-m/2-1]))
    temp = numpy.dot(U_t, spec)
    temp = numpy.dot(W2,temp)
    Broadening = numpy.dot(V,temp)
    
    #Convolve model with this function
    spacing = xnew[2] - xnew[1]
    xnew = numpy.arange(model.x[0], model.x[-1], spacing)
    model_new = Model(xnew)
    Broadening = numpy.array(Broadening)[...,0]
    model = DataStructures.xypoint(x=xnew)
    Broadened = UnivariateSpline(xnew, numpy.convolve(model_new,Broadening, mode="same"),s=0)
    model.y = Broadened(model.x)

    if full_output:
      return MakeModel.RebinData(model, data.x), [Broadening, xnew]
    else:
      return MakeModel.RebinData(model, data.x)

