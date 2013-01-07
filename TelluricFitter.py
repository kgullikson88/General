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

homedir = os.environ['HOME']
TelluricModelDir = homedir + "/School/Research/lblrtm/run_examples/MyModel/"
LineListFile = "LineList2.dat"
ContinuumFile = homedir + "/School/Research/Models/PlanetFinder/src/CRIRES/ContinuumRegions.dat"

#Define some bounds. Bound functions are defined below
ch4_bounds = [1.0, 2.5]
humidity_bounds = [0,100]
co_bounds = [0,1]
resolution_bounds = [10000, 60000]
breakpoint_bounds = [4,100]
decay_bounds = [2,100]
angle_bounds = [0,90]
searchgrid = ((1.2,2.0,0.025),(0.25,15, 0.25),(0,1,.25),(35,36,1))


class ContinuumSegments:
  def __init__(self,size):
    self.low = numpy.zeros(size)
    self.high = numpy.zeros(size)


class TelluricFitter:
  def __init__(self):
    #Set up parameters
    self.parnames = ["pressure", "temperature", "angle", "resolution", "wavestart", "waveend",
                     "h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "no",
                     "so2", "no2", "nh3", "hno3"]
    self.const_pars = [795.0, 273.0, 45.0, 50000.0, 2200.0, 2400.0,
                       50.0, 268.5, 3.9e-2, 0.32, 0.14, 1.8, 2.1e5, 1.1e-19,
                       1e-4, 1e-4, 1e-4, 5.6e-4]
    self.bounds = [[] for par in self.parnames]
    self.fitting = [False]*len(self.parnames)
    self.data = None

    homedir = os.environ['HOME']
    self.LineListFile = "LineList2.dat"
    self.ContinuumFile = homedir + "/School/Research/Models/PlanetFinder/src/CRIRES/ContinuumRegions.dat"


  """
    Display the value of each of the parameters, and show whether it is being fit or not
  """
  def DisplayVariables(self, fitonly=False):
    print "%.15s\tValue\t\tFitting?\tBounds" %("Parameter".ljust(15))
    print "-------------\t-----\t\t-----\t\t-----"
    for i in range(len(self.parnames)):
      if (fitonly and self.fitting[i]) or not fitonly:
        if len(self.bounds[i]) == 2:
          print "%.15s\t%.5g\t\t%s\t\t%g - %g" %(self.parnames[i].ljust(15), self.const_pars[i], self.fitting[i], self.bounds[i][0], self.bounds[i][1])
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
    Edit the locations of continuum in the spectrum (i.e. where there are no telluric lines)
      Wavelengths in nanometers.
  """
  def SetContinuumSectionsFile(self, fname):
    self.ContinuumFile = fname


  """
    Finally, the main fitting function. Before calling this, the user MUST
      1: call FitVariable at least once, specifying which variables will be fit
      2: import data into the class using the ImportData method.
    resolution_fit_mode controls which function is used to estimate the resolution.
      "SVD" is for singlular value decomposition, while "gauss" is for convolving with a gaussian
      (and fitting the width of the guassian to give the best fit)
  """
  def Fit(self, resolution_fit_mode="SVD"):
    print "Fitting now!"
    self.resolution_fit_mode=resolution_fit_mode

    #Make fitpars array
    fitpars = [self.const_pars[i] for i in range(len(self.parnames)) if self.fitting[i] ]
    if len(fitpars) < 1:
      print "Error! Must fit at least one variable!"
      return

    if self.data == None:
      print "Error! Must supply data to fit!"
      return

    #Read in line list:
    linelist = numpy.loadtxt(self.LineListFile)
  
    #Read in continuum database:
    contlist = ReadContinuumDatabase(self.ContinuumFile)

    #Set up the fitting logfile
    outfile = open("chisq_summary.dat", "a")
    outfile.write("\n\n\n\n")
    for i in range(len(self.parnames)):
      if self.fitting[i]:
        outfile.write("%s\t" %self.parnames[i])
    outfile.write("\n")
    outfile.close()

    #Perform the fit
    fitout = leastsq(self.FitErrorFunction, fitpars, args=(linelist, contlist), full_output=True, epsfcn = 0.0005, maxfev=1000)
    fitpars = fitout[0]
        
    return self.GenerateModel(fitpars, linelist, contlist)
    


  def FitErrorFunction(self, fitpars, linelist, segments):
    model = self.GenerateModel(fitpars, linelist, segments)
    
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
      elif len(self.bounds[i]) == 2:
        print "Warning! A constant parameter is bounded!"
        return_array += bound(self.bounds[i], self.const_pars[i])
  
    print "X^2 = ", numpy.sum(return_array)/float(weights.size)
    outfile.write("\n")
    outfile.close()
    return return_array



  def GenerateModel(self, pars, linelist, contlist, nofit=False):
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
    if "FullSpectrum.freq" in os.listdir(TelluricModelDir):
      cmd = "rm "+ TelluricModelDir + "FullSpectrum.freq"
      command = subprocess.check_call(cmd, shell=True)

    model_original = model.copy()
  
    #Reduce to initial guess resolution
    data = self.data
    Continuum = UnivariateSpline(data.x.copy(), data.cont.copy(), s=0)
    if (resolution - 10 < resolution_bounds[0] or resolution+10 > resolution_bounds[1]):
      resolution = numpy.mean(resolution_bounds)
  
    model = MakeModel.ReduceResolution(model.copy(), resolution, Continuum)
    model = MakeModel.RebinData(model.copy(), data.x.copy())

    if nofit:
      return model

    data = self.CCImprove(data, model)

    resid = data.y/model.y
    data.cont = FindContinuum.Continuum(data.x, resid, fitorder=3, lowreject=3, highreject=3)

    modelfcn, mean = self.FitWavelength(data, model.copy(), linelist)
    #model.x = modelfcn(model.x - mean)
    #model_original.x = modelfcn(model_original.x - mean)
    data.x = modelfcn(data.x - mean)

    #Fit resolution
    done = False
    while not done:
      done = True
      if "SVD" in self.resolution_fit_mode:
        model = self.Broaden(data.copy(), model_original.copy())
      elif "gauss" in self.resolution_fit_mode:
        model, resolution = self.FitResolution(data.copy(), model_original.copy(), resolution)
      else:
        done = False
        print "Resolution fit mode set to an invalid value: %s" %self.resolution_fit_mode
        self.resolution_fit_mode = raw_input("Enter a mode (SVD or guass): ")
    
    self.data = data
    return model


  
  #Wavelength-fitting function that just shifts lines, instead of fitting them to gaussians
  def WavelengthErrorFunction(self, shift, data, model):
    modelfcn = UnivariateSpline(model.x, model.y, s=0)
    weight = 1.0/numpy.sqrt(data.y)
    weight[weight < 0.01] = 0.0
    newmodel = modelfcn(model.x + float(shift))
    if shift < 0:
      newmodel[model.x - float(shift) < model.x[0]] = 0
    else:
      newmodel[model.x - float(shift) > model.x[-1]] = 0
    returnvec = (data.y - newmodel)**2*weight
    return returnvec

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
    for line in linelist:
      if line-tol > data.x[0] and line+tol < data.x[-1]:
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
        params,success = leastsq(GaussianErrorFunction, params, args=(argdata.x, argdata.y))
      
        mean = params[2]
        #Do a cross-correlation first, to get the wavelength solution close
        ycorr = scipy.correlate(argdata.y-1.0, argmodel.y-1.0, mode="full")
        xcorr = numpy.arange(ycorr.size)
        maxindex = ycorr.argmax()
        lags = xcorr - (argdata.x.size-1)
        distancePerLag = (argdata.x[-1] - argdata.x[0])/float(argdata.x.size)
        offsets = -lags*distancePerLag
        shift = offsets[maxindex]
        shift, success = leastsq(WavelengthErrorFunction, shift, args=(argdata, argmodel))
        if (debug):
          print argdata.x[0], argdata.x[-1], argdata.x.size
          print "wave: ", mean, "\tshift: ", shift, "\tsuccess = ", success
          pylab.plot(model.x[left:right]-shift, model.y[left:right])
          pylab.plot(argmodel.x, argmodel.y)
          pylab.plot(argdata.x, argdata.y)
        if (success < 5):
          old.append(mean)
          new.append(mean + float(shift))
    if debug:
      pylab.show()
      pylab.plot(old, new, 'ro')
      pylab.show()
    #fit = UnivariateSpline(old, new, k=1, s=0)
    #Iteratively fit to a cubic with sigma-clipping
    order = 3
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
  def CCImprove(self, data, model):
    ycorr = scipy.correlate(data.y-1.0, model.y-1.0, mode="full")
    xcorr = numpy.arange(ycorr.size)
    maxindex = ycorr.argmax()
    lags = xcorr - (data.y.size-1)
    distancePerLag = (data.x[-1] - data.x[0])/float(data.x.size)
    offsets = -lags*distancePerLag
    print "maximum offset: ", offsets[maxindex], " nm"

    if numpy.abs(offsets[maxindex]) < 0.2:
      #Apply offset
      data.x = data.x + offsets[maxindex]
    return data


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
    searchgrid = (resolution_bounds[0], resolution_bounds[1], 5000)
    resolution = brute(ResolutionFitErrorBrute,(searchgrid,), args=(data,newmodel,Continuum))
    resolution, success = leastsq(ResolutionFitError, resolution, args=(data, newmodel, Continuum), epsfcn=10)
    print "Optimal resolution found at R = ", float(resolution)
    newmodel = MakeModel.ReduceResolution(newmodel, float(resolution), Continuum)
    return MakeModel.RebinData(newmodel, data.x), float(resolution)
  
  
  def ResolutionFitError(self, resolution, data, model, cont_fcn):
    newmodel = MakeModel.ReduceResolution(model, resolution, cont_fcn)
    newmodel = MakeModel.RebinData(newmodel, data.x)
    weights = 1.0/data.err
    weights = weights/weights.sum()
    returnvec = (data.y - data.cont*newmodel.y)**2*weights + bound(resolution_bounds, resolution)
    print "Resolution-fitting X^2 = ", numpy.sum(returnvec)/float(weights.size), "at R = ", resolution
    if numpy.isnan(numpy.sum(returnvec**2)):
      print "Error! NaN found in ResolutionFitError!"
      outfile=open("ResolutionFitError.log", "a")
      for i in range(data.y.size):
        outfile.write("%.10g\t" %data.x[i] + "%.10g\t" %data.y[i] + "%.10g\t" %data.cont[i] + "%.10g\t" %newmodel.x[i] + "%.10g\n" %newmodel.y[i])
      outfile.write("\n\n\n\n")
      outfile.close()
    return returnvec

  
  def ResolutionFitErrorBrute(resolution, data, model, cont_fcn):
    return numpy.sum(ResolutionFitError(resolution, data, model, cont_fcn))


  #Fits the broadening profile using singular value decomposition
  #oversampling is the oversampling factor to use before doing the SVD
  #m is the size of the broadening function, in oversampled units
  #dimension is the number of eigenvalues to keep in the broadening function. (Keeping too many starts fitting noise)
  def Broaden(self, data, model, oversampling = 5, m = 101, dimension = 15):
    n = data.x.size*oversampling
  
    #resample data
    Spectrum = UnivariateSpline(data.x, data.y/data.cont, s=0)
    Model = UnivariateSpline(model.x, model.y, s=0)
    xnew = numpy.linspace(data.x[0], data.x[-1], n)
    ynew = Spectrum(xnew)
    #model_new = Model(xnew)
    model_new = MakeModel.RebinData(model, xnew).y

    #Make 'design matrix'
    design = numpy.zeros((n-m,m))
    for j in range(m):
      for i in range(m/2,n-m/2-1):
        design[i-m/2,j] = model_new[i-j+m/2]
    design = mat(design)
    #Do Singular Value Decomposition
    U,W,V_t = svd(design, full_matrices=False)
  
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

    return MakeModel.RebinData(model, data.x)
    


def Main(filename, humidity=None, resolution=None, angle=None, ch4=None, co=None):
  
  #Get some information from the fits header
  hdulist = pyfits.open(filename)
  header = hdulist[0].header
  
  #Resolution:
  if resolution == None:
    slitwidth = header["hierarch eso ins slit1 wid"]
    resolution = -50000/0.2*(slitwidth-0.2)+100000
    
  #Angle:
  if angle == None:
    altitude = header["hierarch eso tel alt"]
    angle = 90 - altitude
  angle_bounds = [angle-2, angle+2] #Do not allow zenith angle to change much
  
  #Humidity:
  if humidity == None:
    humidity = header["hierarch eso tel ambi rhum"]/2.0  #My humidity profile always over-predicts humidity
  
  #Always use fits header for the temperature and pressure
  temperature = header["hierarch eso tel ambi temp"] + 273.15
  pressure_start = header["hierarch eso tel ambi pres start"]
  pressure_end = header["hierarch eso tel ambi pres end"]
  pressure = (pressure_start + pressure_end)/2.0
  
  wave_start = 2250
  wave_end = 2450
  wavenum_start = int(1.0e7/wave_end)
  wavenum_end = int(1.0e7/wave_start+1)
  
  chips = []
  xspacing = 1e-4    #Change this to change the interpolation spacing
  #Generate interpolated chip data
  for i in range(1,5):
    data = hdulist[i].data
    chip = DataStructures.xypoint(data.field(7).size)
    chip.x = data.field(7)
    chip.y = data.field(1)
    chip.cont = data.field(9)
    chip.err = data.field(3)
    #make sure there are no zeros (fancy indexing)
    chip.err[chip.err <= 0] = 1e10
    chips.append(chip)
  hdulist.close()
  
  #Read in line list:
  linelist = numpy.loadtxt(LineListFile)
  
  #Read in continuum database:
  segments = ReadContinuumDatabase(ContinuumFile)
  
  #Set up parameter lists
  if ch4 == None:
    ch4 = 1.6   #CH4 abundance in ppm
  co2 = 368.5 
  if co == None:
    co = 0.14
  o3 = 4e-2
  #CH4 profile parameters:
  breakpoint = 15   #height in km where profile switches from flat to gaussian
  decay = 18.7   #standard deviation of gaussian part
  continuum_fit_order = [5,3,3,3]
  const_pars = [temperature, pressure, co2, o3, wavenum_start, wavenum_end, resolution, angle, 5]
  pars = [ch4, humidity, co]
  
  #Make outfilename from the input fits file
  outfilename = "Corrected_" + filename[3:].split("-")[0] + ".dat"
  print "outputting to ", outfilename
  outfile = open(outfilename, "w")
  debug = False
  if (debug):
    ErrorFunctionBrute = lambda pars, chip, const_pars, linelist, contlist: numpy.sum(ErrorFunction(pars, chip, const_pars, linelist, contlist)**2)
  for i in range(3):
    #ErrorFunction(pars, chips[i], const_pars, linelist, segments)
    
    print "Fitting chip ", i+1, "with size ", chips[i].x.size
    if not debug:
      #const_pars[8] = continuum_fit_order[i]
      fitout = leastsq(ErrorFunction, pars, args=(chips[i], const_pars, linelist, segments), full_output=True, epsfcn = 0.0005, maxfev=1000)
      fitpars = fitout[0]
      pars = fitpars
    else:
      fitout = brute(ErrorFunctionBrute, searchgrid, args=(chips[i], const_pars, linelist, segments))
      fitpars = pars
      
    outfile2 = open("chisq_summary.dat", 'a')
    outfile2.write("\n\n\n\n")
    outfile2.close()
    
    print fitpars
    print "Done fitting chip ", i+1, "\n\n\n"
    model = FitFunction(chips[i], fitpars, const_pars)
    model_original = model.copy()
  
    #convolve to approximate resolution (will fit later)
    resolution = const_pars[6]
    Continuum = UnivariateSpline(chips[i].x, chips[i].cont, s=0)
    model = MakeModel.ReduceResolution(model, resolution, Continuum)
    model = MakeModel.RebinData(model, chips[i].x)
    
    #Fit Continuum of the chip, using the model
    #chips[i] = FitContinuum2(chips[i],model,segments)
    chips[i] = FitContinuum3(chips[i], model,4)
    #if i == 0:
    #  chips[i] = FitContinuum(chips[i], model, condition=0.99)
    #else:
    #  chips[i] = FitContinuum(chips[i], model, order=3, condition=0.99)
  
    #Fit model wavelength to the chip
    #model = FitWavelength(chips[i], model,linelist)
    modelfcn, mean = FitWavelength2(chips[i], model, linelist)
    model_original.x = modelfcn(model_original.x - mean)    

    #Fit resolution:
    model, resolution = FitResolution(chips[i], model_original, resolution)
    Model = UnivariateSpline(model.x, model.y,s=0)
    
    #Estimate errors:
    model2 = Model(chips[i].x)
    residuals = model2 - chips[i].y
    std = numpy.std(residuals)
    covariance = numpy.sqrt(fitout[1]*std)

    #For very deep lines, just set the line cores equal to the model (will destroy an info in the line, but will prevent huge residuals)
    indices = numpy.where(model2 < 0.05)[0]
    chips[i].y[indices] = model2[indices]*chips[i].cont[indices]

    #Output
    if not debug:
      outfile.write("#Temperature: " + str(const_pars[0]) + "\n")
      outfile.write("#Pressure: " + str(const_pars[1]) + "\n")
      outfile.write("#CH4: " + str(fitpars[0]) + " +/- " + str(covariance[0][0]) + "\n")
      outfile.write("#Humidity: " + str(fitpars[1]) + " +/- " + str(covariance[1][1]) + "\n")
      outfile.write("#CO: " + str(fitpars[2]) + " +/- " + str(covariance[2][2]) + "\n")
      outfile.write("#Angle: " + str(const_pars[7]) + "\n")# + " +/- " + str(covariance[2][2]) + "\n")
      outfile.write("#Resolution: " + str(resolution) + "\n")
      outfile.write("#Convergence message: " + fitout[3] + "\n")
      outfile.write("#Convergence code: " + str(fitout[4]) + "\n")
      for j in range(chips[i].x.size):
        outfile.write("%.15f" %chips[i].x[j] + "\t1.0\t%.15f" %(chips[i].y[j]/model2[j]) + "\t1.0\t%.15f" %chips[i].err[j] + "\t%.15f" %chips[i].cont[j] + "\n")
      outfile.write("\n\n\n")
    
    #pylab.plot(chips[i].x, chips[i].y/chips[i].cont, label="data")
    #pylab.plot(chips[i].x, model2, label="model")
    
  outfile.close()
  #pylab.legend()
  #pylab.show()
  
#This function reads the continuum database, and returns the segments
def ReadContinuumDatabase(filename):
  low,high = numpy.loadtxt(filename,usecols=(0,1),unpack=True)
  segments = ContinuumSegments(low.size)
  segments.low = low
  segments.high = high
  return segments  
  
########################################################################
#This function will Generate a telluric model with the given parameters#
########################################################################
def FitFunction(chip, pars, const_pars):
  temperature = const_pars[0]
  pressure = const_pars[1]
  co2 = const_pars[2]
  o3 = const_pars[3]
  wavenum_start = const_pars[4]
  wavenum_end = const_pars[5]
  angle = const_pars[7]
  resolution = const_pars[6]
  ch4 = pars[0]
  humidity = pars[1]
  #angle = pars[2]
  co = pars[2]
  plotflg = False
  
  #Make sure certain variables are positive
  if co < 0:
    co = 0
    print "\nWarning! CO was set to be negative. Resetting to zero before generating model!\n\n"
    #plotflg = True
  if ch4 < 0:
    ch4 = 0
    print "\nWarning! CH4 was set to be negative. Resetting to zero before generating model!\n\n"
    #plotflg = True
  if humidity < 0:
    humidity = 0
    print "\nWarning! Humidity was set to be negative. Resetting to zero before generating model!\n\n"
    #plotflg = True
  if angle < 0:
    angle = -angle
    print "\nWarning! Angle was set to be negative. Resetting to a positive value before generating model!\n\n"
    #plotflg = True

  #Generate the model:
  model = MakeModel.Main(pressure, temperature, humidity, wavenum_start, wavenum_end, angle, co2, o3, ch4, co, chip.x, resolution)
  if "FullSpectrum.freq" in os.listdir(TelluricModelDir):
    cmd = "rm "+ TelluricModelDir + "FullSpectrum.freq"
    command = subprocess.check_call(cmd, shell=True)

  if plotflg:
    pylab.plot(chip.x, chip.y/chip.cont)
    pylab.plot(model.x, model.y)
    pylab.show()
  return model
  
  
def ErrorFunction(pars, chip, const_pars, linelist, contlist):
  #all_pars = const_pars
  #all_pars.extend(pars)
  #print "Pars: ", all_pars
  #ch4_bounds = [1.1, 2.5]
  #humidity_bounds = [0,100]
  #co_bounds = [0,1]
  #resolution_bounds = [35000,125000]
  #breakpoint_bounds = [4,100]
  #decay_bounds = [2,100]
  model = FitFunction(chip.copy(), pars, const_pars)
   
  model_original = model.copy()
  plotflg = False
  #if (pars[1] <= 0):
  #  plotflg = True
  
  #Reduce to initial guess resolution
  Continuum = UnivariateSpline(chip.x.copy(), chip.cont.copy(), s=0)
  resolution = const_pars[6]
  if (resolution - 10 < resolution_bounds[0] or resolution+10 > resolution_bounds[1]):
    resolution = 80000
  
  model = MakeModel.ReduceResolution(model.copy(), resolution, Continuum)
  model = MakeModel.RebinData(model.copy(), chip.x.copy())

  #Fit Continuum of the chip, using the model
  #chip = FitContinuum2(chip,model,contlist)
  chip = FitContinuum3(chip, model, 4)
  #fit_order = const_pars[8]
  #chip = FitContinuum(chip, model, condition=0.99, tol=3, order=fit_order)
  
  #Fit model wavelength to the chip
  #model = FitWavelength(chip,model,linelist)

  modelfcn, mean = FitWavelength2(chip.copy(), model.copy(), linelist)
  model.x = modelfcn(model.x - mean)
  model_original.x = modelfcn(model_original.x - mean)

  #Fit resolution
  model, resolution = FitResolution(chip.copy(), model_original.copy(), resolution, plotflg)
  const_pars[6] = resolution
  #pylab.plot(chip.x, chip.y/chip.cont, label="data")
  #pylab.plot(model.x, model.y, label="model")
  #pylab.legend()
  #pylab.show()
  
  #pylab.plot(chip.x, chip.y, label="data")
  #pylab.plot(model.x, model.y, label="model")
  #pylab.legend(loc=3)
  #pylab.show()
  
  weights = 1.0/chip.err
  weights = weights/weights.sum()
  return_array = ((chip.y  - chip.cont*model.y)**2*weights + bound(ch4_bounds,pars[0]) + 
                                          bound(humidity_bounds,pars[1]) +
                                  	  bound(co_bounds, pars[2]))
  					  #bound(angle_bounds, pars[2]))
  print "X^2 = ", numpy.sum(return_array)/float(weights.size)
  outfile = open("chisq_summary.dat", 'a')
  outfile.write(str(pars[0])+"\t"+str(pars[1])+"\t"+str(pars[2])+"\t"+str(resolution)+"\t"+str(numpy.sum(return_array)/float(weights.size))+"\n")
  outfile.close()
  return return_array
  
  
#Define bounding functions:
# lower bound:            lbound(boundary_value, parameter)
# upper bound:            ubound(boundary_value, parameter)
# lower and upper bounds: bound([low, high], parameter)
# fixed parameter:        fixed(fixed_value, parameter)
lbound = lambda p, x: 1e4*numpy.sqrt(p-x) + 1e-3*(p-x) if (x<p) else 0
ubound = lambda p, x: 1e4*numpy.sqrt(x-p) + 1e-3*(x-p) if (x>p) else 0
bound  = lambda p, x: lbound(p[0],x) + ubound(p[1],x)
fixed  = lambda p, x: bound((p,p), x)


#Function to fit the resolution
def FitResolution(data, model, resolution=75000, plotflg = False):
  ####resolution is the initial guess####
  #Interpolate to a constant wavelength grid
  print "Fitting resolution"
  if plotflg:
    #pylab.plot(data.x, data.y)
    pylab.plot(model.x, model.y)
    pylab.show()
  ModelFcn = UnivariateSpline(model.x, model.y, s=0)
  newmodel = DataStructures.xypoint(model.x.size*2)
  newmodel.x = numpy.linspace(model.x[0], model.x[-1], model.x.size*2)
  newmodel.y = ModelFcn(newmodel.x)

  Continuum = UnivariateSpline(data.x, data.cont, s=0)

  #errfunc = lambda R, data, model: (data.y - MakeModel.ReduceResolutionAndRebinData(model,R,data.x).y - bound(resolution_bounds, R))
  #Do a brute force grid search first, then refine with Levenberg-Marquardt
  searchgrid = (resolution_bounds[0], resolution_bounds[1], 5000)
  resolution = brute(ResolutionFitErrorBrute,(searchgrid,), args=(data,newmodel,Continuum)) #, finish=fmin)
  resolution, success = leastsq(ResolutionFitError, resolution, args=(data, newmodel, Continuum), epsfcn=10)
  #resolution, success = leastsq(ResolutionFitError, resolution, args=(data,newmodel))
  print "Optimal resolution found at R = ", float(resolution)
  newmodel = MakeModel.ReduceResolution(newmodel, float(resolution), Continuum)
  return MakeModel.RebinData(newmodel, data.x), float(resolution)
  
  
def ResolutionFitError(resolution, data, model, cont_fcn):
  newmodel = MakeModel.ReduceResolution(model, resolution, cont_fcn)
  newmodel = MakeModel.RebinData(newmodel, data.x)
  weights = 1.0/data.err
  weights = weights/weights.sum()
  returnvec = (data.y - data.cont*newmodel.y)**2*weights + bound(resolution_bounds, resolution)
  print "Resolution-fitting X^2 = ", numpy.sum(returnvec)/float(weights.size), "at R = ", resolution
  if numpy.isnan(numpy.sum(returnvec**2)):
    #sys.exit("NaN found in ResolutionFitError! Exiting...")
    print "Error! NaN found in ResolutionFitError!"
    outfile=open("ResolutionFitError.log", "a")
    for i in range(data.y.size):
      outfile.write("%.10g\t" %data.x[i] + "%.10g\t" %data.y[i] + "%.10g\t" %data.cont[i] + "%.10g\t" %newmodel.x[i] + "%.10g\n" %newmodel.y[i])
    outfile.write("\n\n\n\n")
    outfile.close()
  return returnvec
 
def ResolutionFitErrorBrute(resolution, data, model, cont_fcn):
  return numpy.sum(ResolutionFitError(resolution, data, model, cont_fcn))


#Fits the broadening profile using singular value decomposition
#oversampling is the oversampling factor to use before doing the SVD
#m is the size of the broadening function, in oversampled units
#dimension is the number of eigenvalues to keep in the broadening function. (Keeping too many starts fitting noise)
def Broaden(data, model, oversampling = 5, m = 101, dimension = 15):
  n = data.x.size*oversampling
  
  #resample data
  Spectrum = UnivariateSpline(data.x, data.y/data.cont, s=0)
  Model = UnivariateSpline(model.x, model.y, s=0)
  xnew = numpy.linspace(data.x[0], data.x[-1], n)
  ynew = Spectrum(xnew)
  #model_new = Model(xnew)
  model_new = MakeModel.RebinData(model, xnew).y

  #Make 'design matrix'
  design = numpy.zeros((n-m,m))
  for j in range(m):
    for i in range(m/2,n-m/2-1):
      design[i-m/2,j] = model_new[i-j+m/2]
  design = mat(design)
  #Do Singular Value Decomposition
  U,W,V_t = svd(design, full_matrices=False)
  
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
  model.x = xnew
  Broadened = UnivariateSpline(xnew, numpy.convolve(model_new,Broadening, mode="same"),s=0)
  model.y = Broadened(model.x)

  return model


#Function to Fit the wavelength solution, using a bunch of telluric lines
#This assumes that we are already quite close to the correct solution
def FitWavelength(data, model, linelist, tol=2e-2, FWHM=0.05, debug=False):
  sigma = FWHM/2.65	#approximate relation, but good enough for initial guess
  
  #Fitting function: gaussian on a linear background:
  fitfunc = lambda p,x: p[0] + p[1]*x + p[2]*numpy.exp(-(x-p[3])**2/(2*p[4]**2))
  errfunc = lambda p,x,y: y - fitfunc(p,x)
  fitfunc2 = lambda p,x: p[0] + p[1]*numpy.exp(-(x-p[2])**2/(2*p[3]**2))
  errfunc2 = lambda p,x,y,err: (y - fitfunc2(p,x))/(err)

  data_points = []
  model_points = []
  data_rms = []
  model_rms = []
  
  if debug:
    print "Plotting in debug mode"
    pylab.plot(data.x, data.y)
    pylab.plot(model.x, model.y)
  
  #Begin loop over the lines
  for line in linelist:
    if line-tol-FWHM/2.0 > data.x[0] and line+tol+FWHM/2.0 < data.x[-1]:
      #This line falls on the chip. Use it
      #First, find the lowest point within tol of the given line position
      left = numpy.searchsorted(data.x, line - tol)
      right = numpy.searchsorted(data.x, line + tol)
      minindex = data.y[left:right].argmin() + left
      
      #Set up guess parameters for line fit:
      mean = data.x[minindex]
      left2 = numpy.searchsorted(data.x, mean - FWHM/2.0)
      right2 = numpy.searchsorted(data.x, mean + FWHM/2.0)
      slope = (data.y[right2] - data.y[left2])/(data.x[right2] - data.x[left2])
      a = data.y[right2] - slope*data.x[right2]
      b = slope
      amplitude = (a + b*mean) - data.y[minindex]
      #pars = [a,b,-amplitude,mean,sigma]
      pars=[1,-amplitude, mean, sigma]
      print "Number of points in fit: ", right2 - left2 + 1

      #Do the fit
      #pars, success = leastsq(errfunc, pars, args=(data.x[left2:right2], data.y[left2:right2]))
      pars, success = leastsq(errfunc2, pars, args=(data.x[left2:right2], data.y[left2:right2]/data.cont[left2:right2], data.err[left2:right2]))      
      data_fit_failed = True
      if success < 5:
        #Determine rms error
        resid = data.y[left2:right2] - fitfunc2(pars, data.x[left2:right2])
        data_rms.append(numpy.sqrt(numpy.sum(resid**2)))
        
        #Save the mean value:
        mean = pars[2]
        data_points.append(mean)
        data_fit_failed = False
        if debug:
          print "Plotting in debug mode"
          pylab.plot(data.x[left2:right2], fitfunc2(pars, data.x[left2:right2]))
          pylab.plot(mean, fitfunc2(pars, mean), 'ro')
      elif debug:
        print "Data fit bad"
      ################################################################
      #  Now, do the same thing for the model
      ################################################################
      minindex = model.y[left:right].argmin() + left
      
      #Set up guess parameters for line fit:
      mean = model.x[minindex]
      left2 = numpy.searchsorted(model.x, mean - FWHM/2.0)
      right2 = numpy.searchsorted(model.x, mean + FWHM/2.0)
      slope = (model.y[right2] - model.y[left2])/(model.x[right2] - model.x[left2])
      a = model.y[right2] - slope*model.x[right2]
      b = slope
      amplitude = (a + b*mean) - model.y[minindex]
      #pars = [a,b,-amplitude,mean,sigma]
      pars = [1,-amplitude, mean, sigma]
      
      #Do the fit
      #pars, success = leastsq(errfunc, pars, args=(model.x[left2:right2], model.y[left2:right2]))
      pars, success = leastsq(errfunc2, pars, args=(model.x[left2:right2], model.y[left2:right2]/model.y[left2:right2], numpy.ones(model.y[left2:right2].size)))
      
      if success < 5 and not data_fit_failed:
        #Determine rms error
        resid = model.y[left2:right2] - fitfunc2(pars, model.x[left2:right2])
        model_rms.append(numpy.sqrt(numpy.sum(resid**2)))

        #Save the mean value:
        mean=pars[2]
        model_points.append(mean)
        if debug:
          print "Plotting in debug mode"
          pylab.plot(model.x[left2:right2], fitfunc2(pars, model.x[left2:right2]))
          pylab.plot(mean, fitfunc2(pars,mean), 'bo')
      elif success >=5 and not data_fit_failed:
        data_points.pop()
      elif debug:
        print "Model fit bad"
        
  #Output figure:
  figs = os.listdir("FitPictures")
  j = 0
  for fname in figs:
   if "lines-" in fname:
     i = fname.split("lines-")[-1].split(".")[0]
     if int(i) > j:
       j = int(i)
  linefig = "FitPictures/lines-" + str(j+1) + ".png"
  fitfig = "FitPictures/fit-" + str(j+1) + ".png"
  if (debug):
    pylab.savefig(linefig, dpi=600)
    pylab.show()
    pylab.cla()
  #Remove points with too large an rms
  done = False
  order = 3
  while not done:
    done = True
    pars = numpy.polyfit(model_points, data_points, order)
    fit = numpy.poly1d(pars)
    residuals = list(data_points - fit(model_points))
    mean = numpy.mean(residuals)
    std = numpy.std(residuals)
    for i in range(len(data_points)-1,-1,-1):
      if numpy.abs(residuals[i]) > mean + 2.5*std:
        if (debug):
          print "Removing point near ", data_points[i]
          pylab.plot(model_points[i], data_points[i], 'bx')
        data_points.pop(i)
        model_points.pop(i)
        data_rms.pop(i)
        model_rms.pop(i)
        residuals.pop(i)
        done = False
  if debug:  
    for i in range(len(data_points)):
      pylab.plot(model_points[i], data_points[i], 'ro')
  
  #Fit datapoints and modelpoints to a quadratic
  #pars = numpy.polyfit(model_points, data_points, order)
  #fit = numpy.poly1d(pars)
  fit = UnivariateSpline(model_points, data_points,k=2,s=0)
  if (debug):
    pylab.plot(model.x, fit(model.x))
    pylab.savefig(fitfig, dpi=600)
    pylab.show()
    residuals = data_points - fit(model_points)
    pylab.plot(data_points, residuals, 'ro')
    pylab.show()
    pylab.cla()
  
  model.x = fit(model.x)
  return model


#Second wavelength-fitting function that just shifts lines, instead of fitting them to gaussians
def WavelengthErrorFunction(shift, data, model):
  modelfcn = UnivariateSpline(model.x, model.y, s=0)
  weight = 1.0/numpy.sqrt(data.y)
  weight[weight < 0.01] = 0.0
  newmodel = modelfcn(model.x + float(shift))
  if shift < 0:
    newmodel[model.x - float(shift) < model.x[0]] = 0
  else:
    newmodel[model.x - float(shift) > model.x[-1]] = 0
  returnvec = (data.y - newmodel)**2*weight
  return returnvec

def FitWavelength2(chip, telluric, linelist, tol=0.05, oversampling = 4, debug=False):
  old = []
  new = []

  #Interpolate to finer spacing
  DATA_FCN = UnivariateSpline(chip.x, chip.y, s=0)
  CONT_FCN = UnivariateSpline(chip.x, chip.cont, s=0)
  MODEL_FCN = UnivariateSpline(telluric.x, telluric.y, s=0)
  data = DataStructures.xypoint(chip.x.size*oversampling)
  data.x = numpy.linspace(chip.x[0], chip.x[-1], chip.x.size*oversampling)
  data.y = DATA_FCN(data.x)
  data.cont = CONT_FCN(data.x)
  model = DataStructures.xypoint(data.x.size)
  model.x = numpy.copy(data.x)
  model.y = MODEL_FCN(model.x)
  
  #Begin loop over the lines
  for line in linelist:
    if line-tol > data.x[0] and line+tol < data.x[-1]:
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
      params,success = leastsq(GaussianErrorFunction, params, args=(argdata.x, argdata.y))
      
      mean = params[2]
      #Do a cross-correlation first, to get the wavelength solution close
      ycorr = scipy.correlate(argdata.y-1.0, argmodel.y-1.0, mode="full")
      xcorr = numpy.arange(ycorr.size)
      maxindex = ycorr.argmax()
      lags = xcorr - (argdata.x.size-1)
      distancePerLag = (argdata.x[-1] - argdata.x[0])/float(argdata.x.size)
      offsets = -lags*distancePerLag
      shift = offsets[maxindex]
      shift, success = leastsq(WavelengthErrorFunction, shift, args=(argdata, argmodel))
      if (debug):
        print argdata.x[0], argdata.x[-1], argdata.x.size
        print "wave: ", mean, "\tshift: ", shift, "\tsuccess = ", success
        pylab.plot(model.x[left:right]-shift, model.y[left:right])
        pylab.plot(argmodel.x, argmodel.y)
        pylab.plot(argdata.x, argdata.y)
      if (success < 5):
        old.append(mean)
        new.append(mean - float(shift))
  if debug:
    pylab.show()
    pylab.plot(old, new, 'ro')
    pylab.show()
  #fit = UnivariateSpline(old, new, k=1, s=0)
  #Iteratively fit to a cubic with sigma-clipping
  order = 3
  done = False
  while not done:
    done = True
    mean = numpy.mean(old)
    fit = numpy.poly1d(numpy.polyfit(old - mean, new, order))
    residuals = fit(old - mean) - new
    std = numpy.std(residuals)
    #if debug:
    #  pylab.plot(old, residuals, 'ro')
    #  pylab.plot(old, std*numpy.ones(len(old)))
    #  pylab.show()
    badindices = numpy.where(numpy.logical_or(residuals > 2*std, residuals < -2*std))[0]
    for badindex in badindices[::-1]:
      del old[badindex]
      del new[badindex]
      done = False
  if debug:
    pylab.plot(old, fit(old - mean) - new, 'ro')
    pylab.show()
  #if len(old) > 0:
    #This is just to ensure that we didn't remove all of the points in the sigma-clipping
  #  telluric.x = fit(telluric.x - mean)
  return fit, mean
  

#Gaussian absorption line
def GaussianFitFunction(x,params):
  cont = params[0]
  depth = params[1]
  mu = params[2]
  sig = params[3]
  return cont - depth*numpy.exp(-(x-mu)**2/(2*sig**2))

#Returns the residuals between the fit from above and the actual values
def GaussianErrorFunction(params, x, y):
  return GaussianFitFunction(x,params) - y


#This function will fit the continuum in the regions given
def FitContinuum(chip,model,contlist):
  wave = []
  cont = []
  weight = []
  minimum = chip.x[0]
  maximum = chip.x[-1]
    
  #loop over the segments
  fig1 = pylab.figure(1)
  pylab.plot(chip.x, chip.y, 'r-')
  fig2 = pylab.figure(2)
  pylab.plot(model.x, model.y, 'r-')
  for i in range(contlist.low.size):
    if contlist.high[i] > minimum and contlist.low[i] < maximum:
      left = numpy.searchsorted(chip.x, contlist.low[i])
      right = numpy.searchsorted(chip.x, contlist.high[i])
      data = (numpy.mean(chip.y[left:right]))
      pylab.figure(1)
      pylab.plot(chip.x[left:right], chip.y[left:right], 'b-')
        
      wave.append(chip.x[(left+right)/2])
        
      left = numpy.searchsorted(model.x, contlist.low[i])
      right = numpy.searchsorted(model.x, contlist.high[i])
      telluric_model = (numpy.mean(model.y[left:right]))
      pylab.figure(2)
      pylab.plot(model.x[left:right], model.y[left:right], 'b-')
      
      cont.append(data/telluric_model)
      weight.append(float(right-left))
  
  pylab.show()
  
  #Interpolate/Extrapolate the continuum:
  wave = numpy.array(wave)
  cont = numpy.array(cont)
  weight = numpy.array(weight)
  weight = weight/numpy.max(weight)
  chisq = numpy.ones(5)
  #fit = extrap1d(interp1d(wave,cont,kind='linear'))
  """
  for i in range(chisq.size):
    fit = UnivariateSpline(wave,cont,w=weight,k=i+1)
    chisq[i] = numpy.sum((model.y - chip.y/fit(chip.x))**2)
    #chisq[i] = fit.get_residual()
    print "k = ",i+1,": ",chisq[i]
 
  #Use the best-fitting spline order
  fit = UnivariateSpline(wave,cont,w=weight,k=chisq.argmin()+1)  
  print "Using spline of order k=", chisq.argmin()+1, " to fit continuum"
  """
  pars = numpy.polyfit(wave,cont,3)
  pars, success = leastsq(ContFitFcn, pars, args=(wave,cont,weight))
  fit = numpy.poly1d(pars)
  
  chip.cont = fit(chip.x)
  
  pylab.plot(wave, cont, 'ro')
  pylab.plot(chip.x, chip.cont)
  pylab.plot(chip.x, chip.y)
  pylab.show()
  pylab.plot(chip.x, chip.y/chip.cont)
  pylab.plot(model.x, model.y)
  pylab.show()
  
  #pylab.plot(chip.x, chip.y, label="data")
  #pylab.plot(wave, cont/fit(wave), 'ro')
  #pylab.plot(model.x, model.y, label="model")
  #pylab.legend(loc=3)
  #pylab.show()

  return chip
  
#This function will fit the continuum in the regions given
def FitContinuum2(chip,model,contlist):
  wave = []
  cont = []
  data = []
  weight = []
  minimum = chip.x[0]
  maximum = chip.x[-1]
  model_high = 0.0  
  data_high = 0.0
  
  Model = UnivariateSpline(model.x, model.y, s=0)

  #loop over the segments
  for i in range(contlist.low.size):
    if contlist.high[i] > minimum and contlist.low[i] < maximum:
      left = numpy.searchsorted(chip.x, contlist.low[i])
      right = numpy.searchsorted(chip.x, contlist.high[i])
      for i in range(left,right):
        #data.append(float(chip.y[i]/Model(chip.x[i])))
        data.append(chip.y[i]/model.y[i])
        wave.append(chip.x[i])
        weight.append(numpy.sqrt(model.y[i]))
 
  #Interpolate/Extrapolate the continuum:
  wave = numpy.array(wave)
  #cont = numpy.array(cont)/model_high
  data = numpy.array(data)
  weight = numpy.array(weight)
  weight = weight/numpy.sum(weight)
  #weight = numpy.ones(data.size)
  
  pars = numpy.polyfit(wave,data,4)
  pars, success = leastsq(ContFitFcn, pars, args=(wave,data,weight))
  fit = numpy.poly1d(pars)
  chip.cont = fit(chip.x)
  return chip
 

def ContFitFcn(pars, x, y, w):
  #retval = pars[0]
  retval = 0.0
  for i in range(len(pars)):
    retval = retval + pars[i]*x**float(len(pars) - 1 - i)
  return (retval - y)*w
  
def FitContinuum3(chip,model,order=2):
  wave = chip.x.copy()
  flux = chip.y.copy()
  model2 = model.y.copy()
  weight = (model2/model2.max())**10
  done = False
  while not done:
    done = True
    wave_mean = numpy.mean(wave)
    pars = numpy.polyfit(wave - wave_mean, flux/model2, order)
    pars, success = leastsq(ContFitFcn, pars, args=(wave - wave_mean, flux/model2, weight))
    fit = numpy.poly1d(pars)
    residuals = flux/(model2*fit(wave - wave_mean))
    std = numpy.std(residuals)
    mean = numpy.mean(residuals)
    badindices = numpy.where(numpy.abs(residuals-mean) > std*3)[0]
    wave = numpy.delete(wave, badindices)
    flux = numpy.delete(flux, badindices)
    model2 = numpy.delete(model2, badindices)
    weight = numpy.delete(weight, badindices)
    if badindices.size > 0:
      done = False
  chip.cont = fit(chip.x - wave_mean)
  return chip

 
def FitContinuum(chip, model, condition=0.95, tol=3, order=5):
  #Fit continuum, using all points with model transmission > condition
  done = False
  while not done:
    done = True
    wave = model.x[model.y > condition]
    cont = chip.y[model.y > condition]/model.y[model.y > condition]
  
    #make sure there are some points on the edges of the chip
    if wave[0] > chip.x[0] + 1.0 or wave[-1] < chip.x[-1] - 1.0:
      condition = condition - 0.005
      done = False
  
  done = False
  while not done:
    done = True
    fit = numpy.poly1d(numpy.polyfit(wave - wave.mean(), cont, order))
    resid = cont - fit(wave - wave.mean())
    mean = numpy.mean(resid)
    std = numpy.std(resid)
    badvals = numpy.abs(resid - mean) > std*tol
    if numpy.sum(badvals) > 0:
      done = False
    deleteindices = []
    for i in range(badvals.size):
      if badvals[i]:
        deleteindices.append(i)
        
    cont = numpy.delete(cont, deleteindices)
    wave = numpy.delete(wave, deleteindices)
      
  fit = numpy.poly1d(numpy.polyfit(wave - wave.mean(), cont, order))
  chip.cont = fit(chip.x - wave.mean())
  
  return chip
  
  
#allows interp1d to extrapolate
def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return numpy.array(map(pointwise, numpy.array(xs)))

    return ufunclike

if __name__=="__main__":
  #Parse command-line arguments for filename as well as optional abundances
  try:
    filename = sys.argv[1]
  except IndexError:
    filename = raw_input("Enter filename to fit: ") 
  outfile=open("chisq_summary.dat", "w")
  outfile.write("#CH4\tH20\tCO\tR\tX^2\n")
  outfile.close()
  j=2
  water = None
  ch4 = None
  resolution=None
  angle=None
  co=None
  for i in range(j,len(sys.argv)):
    if j >=len(sys.argv):
      break
    elif "h2o" in sys.argv[j]:
      water = float(sys.argv[j+1])
      j = j+1
      print "water = ", water
    elif "ch4" in sys.argv[j]:
      ch4 = float(sys.argv[j+1])
      j = j+1
      print "CH4 = ", ch4
    elif "resolution" in sys.argv[j]:
      resolution = float(sys.argv[j+1])
      j=j+1
      print "Resolution = ", resolution
    elif "angle" in sys.argv[j]:
      angle = float(sys.argv[j+1])
      j = j+1
      print "angle = ", angle
    elif "co" in sys.argv[j]:
      co = float(sys.argv[j+1])
      j = j+1
      print "CO = ", co
    j = j+1 
  Main(filename, humidity=water, resolution=resolution, angle=angle, ch4=ch4, co=co)
  
  
