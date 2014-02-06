"""
  This class performs the analysis of B stars to determine the following:
  - vsini
  -rv
  -Teff
  -logg
  -vmacro
  -vmicro
  -[Fe/H]


  Example usage:
  --------------
  Put usage example here
"""

#import matplotlib
#matplotlib.use("GTKAgg")
import matplotlib.pyplot as plt
import HelperFunctions
import numpy
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import interp1d
import DataStructures
from collections import defaultdict
import os
import warnings
import subprocess
import time
import FittingUtilities
import Correlate
import SpectralTypeRelations
import RotBroad
from astropy import units, constants


class Analyse():
  def __init__(self, gridlocation="/Volumes/DATADRIVE/Stellar_Models/BSTAR06", SpT=None, debug=False, fname=None, Teff=None, logg=None):
    # Just define some class variables
    self.gridlocation = gridlocation
    self.debug = debug
    self.windval2str = {-13.8: "A",
                   -14.0: "a",
                   -13.4: "B",
                   -13.6: "b",
                   -13.15: "C",
                   -12.7: "D",
                   -14.30: "O"}
    self.grid = defaultdict(lambda : defaultdict(
                            lambda : defaultdict(
                            lambda : defaultdict(
                            lambda : defaultdict(
                            lambda : defaultdict(
                            lambda : defaultdict(
                            lambda : defaultdict(
                            lambda : defaultdict(DataStructures.xypoint)))))))))


    self.Teffs =  range(10000, 20000, 500) + range(20000, 31000, 1000)
    self.Heliums = (0.1, 0.15, 0.2)
    self.Silicons = (-4.19, -4.49, -4.79)
    self.logQ = (-14.3, -14.0, -13.8, -13.6, -13.4, -13.15, -12.7)
    self.betas = (0.9, 1.2, 1.5, 2.0, 3.0)
    self.species = ['BR10', 'BR11', 'BRALPHA', 'BRBETA', 'BRGAMMA', 
                    'HALPHA', 'HBETA', 'HDELTA', 'HEI170', 'HEI205', 
                    'HEI211', 'HEI4010', 'HEI4026', 'HEI4120', 'HEI4140', 
                    'HEI4387', 'HEI4471', 'HEI4713', 'HEI4922', 'HEI6678', 
                    'HEII218', 'HEII4200', 'HEII4541', 'HEII4686', 'HEII57', 
                    'HEII611', 'HEII6406', 'HEII6527', 'HEII6683', 'HEII67', 
                    'HEII712', 'HEII713', 'HEPS', 'HGAMMA', 'PALPHA', 
                    'PBETA', 'PF10', 'PF9', 'PFGAMMA', 'PGAMMA', 
                    'SiII4128', 'SiII4130', 'SiII5041', 'SiII5056', 'SiIII4552', 
                    'SiIII4567', 'SiIII4574', 'SiIII4716', 'SiIII4813', 'SiIII4819', 
                    'SiIII4829', 'SiIII5739', 'SiIV4089', 'SiIV4116', 'SiIV4212',
                    'SiIV4950', 'SiIV6667', 'SiIV6701']

    # Get spectral type if the user didn't enter it
    if SpT == None:
      SpT = raw_input("Enter Spectral Type: ")
    self.SpT = SpT

    #Use the spectral type to get effective temperature and log(g)
    #   (if those keywords were not given when calling this)
    MS = SpectralTypeRelations.MainSequence()
    if Teff == None:
      Teff = MS.Interpolate(MS.Temperature, SpT)
    if logg == None:
      M = MS.Interpolate(MS.Mass, SpT)
      R = MS.Interpolate(MS.Radius, SpT)
      G = constants.G.cgs.value
      Msun = constants.M_sun.cgs.value
      Rsun = constants.R_sun.cgs.value
      logg = numpy.log10(G*M*Msun/(R*Rsun**2))
    self.Teff_guess = Teff
    self.logg_guess = logg
    self.vsini = 300
    

    # Read the filename if it is given
    self.data = None
    if fname != None:
      self.InputData(fname)

    # Initialize a figure for drawing
    #plt.figure(1)
    #plt.plot([1], [1], 'ro')
    #plt.show(block=False)
    #plt.cla()
    
                    
   


                
  def GetModel(self, Teff, logg, Q, beta, helium, silicon, species, vmacro):
    """
      This method takes the following values, and finds the closest match
        in the grid. It will warn the user if the values are not actual
        grid points, which they should be!

      Parameters:
      -----------
      Teff:           Effective temperature of the star (K)
                      Options: 10000-20000K in 500K steps, 20000-30000 in 1000K steps
                      
      logg:           log of the surface gravity of the star (cgs)
                      Options: 4.5 to 4log(Teff) - 15.02 in 0.1 dex steps
                      
      Q:              log of the wind strength logQ = log(Mdot (R * v_inf)^-1.5
                      Options: -14.3, -14.0, -13.8, -13.6, -13.4, -13.15, -12.7
                      
      beta:           Wind velocity law for the outer expanding atmosphere
                      Options: 0.9, 1.2, 1.5, 2.0, 3.0
                      
      helium:         Helium fraction of the star's atmsophere
                      Options: 0.10, 0.15, 0.20
                      
      silicon:        Relative silicon abundance as log(nSi/nH)
                      Options: -4.19, -4.49, -4.79

      vmacro:         Macroturbulent velocity (km/s)
                      Options: 3,6,10,12,15 for Teff<20000
                               6,10,12,15,20 for Teff>20000
                      
      species:        The name of the spectral line you want
                      Options: Many. Just check the model grid.
    """

    # Check to see if the input is in the grid
    if Teff not in self.Teffs:
      warnings.warn("Teff value (%g) not in model grid!" %Teff)
      Teff = HelperFunctions.GetSurrounding(self.Teffs, Teff)[0]
      print "\tClosest value is %g\n\n" %Teff
    # logg and vmacro depend on Teff, so make those lists now
    loggs = [round(g, 2) for g in numpy.arange(4.5, 4*numpy.log10(Teff) - 15.02, -0.1)]
    if Teff < 20000:
      self.vmacros = (3,6,10,12,15)
    else:
      self.vmacros = (6,10,12,15,20)

    # Continue checking if the inputs are on a grid point
    if logg not in loggs:
      warnings.warn("log(g) value (%g) not in model grid!" %logg)
      logg = HelperFunctions.GetSurrounding(loggs, logg)[0]
      print "\tClosest value is %g\n\n" %logg
    if Q not in self.logQ:
      warnings.warn("log(Q) wind value (%g) not in model grid!" %Q)
      Q = HelperFunctions.GetSurrounding(self.logQ, Q)[0]
      print "\tClosest value is %g\n\n" %Q
    if beta not in self.betas:
      warnings.warn("Beta value (%g) not in model grid!" %beta)
      beta = HelperFunctions.GetSurrounding(self.betas, beta)[0]
      print "\tClosest value is %g\n\n" %beta
    if helium not in self.Heliums:
      warnings.warn("Helium fraction (%g) not in model grid!" %helium)
      helium = HelperFunctions.GetSurrounding(self.Heliums, helium)[0]
      print "\tClosest value is %g\n\n" %helium
    if silicon not in self.Silicons:
      warnings.warn("Silicon relative abundance (%g) not in model grid!" %silicon)
      silicon = HelperFunctions.GetSurrounding(self.Silicons, silicon)[0]
      print "\tClosest value is %g\n\n" %silicon
    if species not in self.species:
      raise ValueError("Desired species ( %s ) not in grid!" %species)
    if vmacro not in self.vmacros:
      warnings.warn("Macroturbulent velocity (%g) not in model grid!" %vmacro)
      vmacro = HelperFunctions.GetSurrounding(self.vmacros, vmacro)[0]
      print "\tClosest value is %g\n\n" %vmacro


    # Now, make the filename that this model (or the closest one) is at
    windstr = self.windval2str[Q]
    abstr = "He%iSi%i" %(helium*100, -silicon*100)
    fname = "%s/T%i/g%i/%s%.2i/%s/OUT.%s_VT%.3i.gz" %(self.gridlocation, Teff, logg*10, windstr, beta*10, abstr, species, vmacro)

    # Gunzip that file to a temporary one.
    tmpfile = "tmp%f" %time.time()
    lines = subprocess.check_output(['gunzip', '-c', fname])
    output = open(tmpfile, "w")
    output.writelines(lines)
    output.close()
    x, y = numpy.genfromtxt(tmpfile, invalid_raise=False, usecols=(2,4), unpack=True)

    #Removes NaNs from random extra lines in the files...
    while numpy.any(numpy.isnan(y)):
      x = x[:-1]
      y = y[:-1]

    # Check for duplicate x points (why does this grid suck so much?!)
    xdiff = numpy.array([x[i+1] - x[i] for i in range(x.size-1)])
    goodindices = numpy.where(xdiff > 1e-7)[0]
    x = x[goodindices]
    y = y[goodindices]

    #Convert from angrstrom to nm, and switch to air wavelengths
    x = x*units.angstrom.to(units.nm) / 1.00026

    # delete the temporary file
    subprocess.check_call(['rm', tmpfile])
    return DataStructures.xypoint(x=x, y=y)


  
  def InputData(self, fname):
    """
      This takes a fits file, and reads it as a bunch of echelle orders
    """
    orders = HelperFunctions.ReadFits(fname, extensions=True, x="wavelength", y="flux", cont="continuum", errors="error")
    self.data = orders
    self.fname = fname.split("/")[-1]
    return


  
  def FindVsini(self, vsini_lines="%s/School/Research/Useful_Datafiles/vsini.list" %os.environ["HOME"]):
    """
      This function will read in the linelist useful for determining vsini.
    For each one, it will ask the user if the line is usable by bringing up 
    a plot of the appropriate order in the data. If the user says it is, then
    the vsini is determined as in Simon-Diaz (2007).
    """

    # First, check to make sure the user entered a datafile
    if self.data == None:
      fname = raw_input("Enter filename for the data: ")
      self.InputData(fname)

    # Read in the vsini linelist file
    center, left, right = numpy.loadtxt(vsini_lines, usecols=(1,2,3), unpack=True)
    center /= 10.0
    left /= 10.0
    right /= 10.0

    # Find each line in the data
    plt.figure(1)
    vsini_values = []
    for c,l,r in zip(center, left, right):
      found = False
      for order in self.data:
        if c > order.x[0] and c < order.x[-1]:
          found = True
          break
      if not found:
        continue
      first = numpy.searchsorted(order.x, l)
      last = numpy.searchsorted(order.x, r)
      segment = order[first:last]
      segment.cont = FittingUtilities.Continuum(segment.x, segment.y, fitorder=1, lowreject=1, highreject=5)
      segment.y /= segment.cont
      
      plt.plot(segment.x, segment.y)
      yrange = plt.gca().get_ylim()
      plt.plot((c,c), yrange, 'r--', lw=2)
      plt.xlabel("Wavelength (nm)")
      plt.ylabel("Relative flux")
      plt.draw()
      valid = raw_input("Is this line usable for vsini determination (y/n)? ")
      if "n" in valid.lower():
        plt.cla()
        continue

      # Ask if the continuum needs to be renormalized
      done = False
      while not done:
        renorm = raw_input("Renormalize continuum (y/n)? ")
        if "y" in renorm.lower():
          plt.cla()
          segment = self.UserContinuum(segment)
          plt.plot(segment.x, segment.y)
          plt.plot(segment.x, segment.cont)
          plt.draw()
        else:
          segment.y /= segment.cont
          done = True

      # Fourier transform the line, and let the user click on the first minimum
      plt.cla()
      vsini = self.UserVsini(segment)
      vsini_values.append(vsini)

      plt.cla()

    # Save the mean and standard deviation in the file 'vsini.dat'
    outfile = open("vsini.dat", "a")
    outfile.write("%s%.2f\t%.3f\n" %(self.fname.ljust(20), numpy.mean(vsini_values), numpy.std(vsini_values)))
    self.vsini = numpy.mean(vsini_values)
    return



  def GetRadialVelocity(self):
    """
      This function will get the radial velocity by cross-correlating a model
    of the star against all orders of the data. The maximum of the CCF will 
    likely be broad due to rotational broadening, but will still encode the
    rv of the star (plus Earth, if the observations are not barycentric-corrected)
    """
    # Get all of the models with the appropriate temperature and log(g)
    # We will assume solar abundances of everything, and no wind for this

    xgrid = numpy.arange(self.data[0].x[0]-20, self.data[-1].x[-1]+20, 0.01)
    full_model = DataStructures.xypoint(x=xgrid, y=numpy.ones(xgrid.size))
    Teff = HelperFunctions.GetSurrounding(self.Teffs, self.Teff_guess)[0]
    loggs = [round(g, 2) for g in numpy.arange(4.5, 4*numpy.log10(Teff) - 15.02, -0.1)]
    logg = HelperFunctions.GetSurrounding(loggs, self.logg_guess)[0]
    corrlist = []
    normalization = 0.0
    species = ['BRALPHA', 'BRBETA', 'BRGAMMA', 
                'HALPHA', 'HBETA', 'HDELTA', 'HGAMMA']
    for spec in species:
      print "\nGetting model for %s" %spec
      model = self.GetModel(Teff,
                            logg,
                            -14.3,
                            0.9,
                            0.1,
                            -4.49,
                            spec,
                            10)
      # Find the appropriate order
      w0 = (model.x[0] + model.x[-1])/2.0
      idx = -1
      diff = 9e9
      for i, order in enumerate(self.data):
        x0 = (order.x[0] + order.x[-1])/2.0
        if abs(x0-w0) < diff and w0 > order.x[0] and w0 < order.x[-1]:
          diff = abs(x0-w0)
          idx = i
      if idx < 0 or (idx == i and diff > 10.0):
        continue
      order = self.data[idx]
      
      #model.x *= (1+19.6/3e5)

      # Make sure the model is bigger than this order
      if model.x[0] > order.x[0]-5.0:
        model.x = numpy.r_[(order.x[0]-5.0,), model.x]
        model.y = numpy.r_[(1.0,), model.y]
      if model.x[-1] < order.x[-1]+5.0:
        model.x = numpy.r_[model.x, (order.x[-1]+5.0,)]
        model.y = numpy.r_[model.y, (1.0,)]
      model.cont = numpy.ones(model.x.size)

      # Rotationally broaden model
      xgrid = numpy.arange(model.x[0], model.x[-1], 0.001)
      model = FittingUtilities.RebinData(model, xgrid)
      model = RotBroad.Broaden(model, self.vsini*units.km.to(units.cm))

      # Find low point:
      idx = numpy.argmin(model.y)
      w0 = model.x[idx]
      idx = numpy.argmin(order.y/order.cont)
      x0 = order.x[idx]
      print "Model wavelength = %.5f" %w0
      print "Data wavelength = %.5f" %x0
      print "Velocity shift = %g km/s" %(3e5*(x0-w0)/w0)
      

      # Rebin data to constant (log) spacing
      start = numpy.log(order.x[0])
      end = numpy.log(order.x[-1])
      neworder = order.copy()
      neworder.x = numpy.logspace(start, end, order.size(), base=numpy.e)
      neworder = FittingUtilities.RebinData(order, neworder.x)

      # Rebin the model to the same spacing
      logspacing = numpy.log(neworder.x[1]/neworder.x[0])
      left = numpy.searchsorted(model.x, order.x[0] - 10)
      right = numpy.searchsorted(model.x, order.x[-1] + 10)
      right = min(right, model.size()-2)
      left, right = 0, -1
      start = numpy.log(model.x[left])
      end = numpy.log(model.x[right])
      xgrid = numpy.exp(numpy.arange(start, end+logspacing*1.1, logspacing))
      
      segment = FittingUtilities.RebinData(model, xgrid) 
      plt.figure(3)
      plt.plot(neworder.x, neworder.y/neworder.cont)
      plt.plot(segment.x, segment.y/segment.cont)

      corr = Correlate.Correlate([neworder,], [segment,], debug=True)
      plt.figure(2)
      plt.plot(corr.x, corr.y)


      if not numpy.any(numpy.isnan(corr.y)):
        corrlist.append(corr)
        normalization += float(order.size())
      
      
      #fcn = interp1d(model.x, model.y, kind='linear', bounds_error=False, fill_value=1.0)
      
      #full_model.y *= fcn(full_model.x)
      #plt.plot(model.x, model.y)
      
    #plt.plot(full_model.x, full_model.y)
    #plt.show()

    #output = Correlate.GetCCF(self.data, full_model, vsini=0.0, resolution=60000, process_model=True, rebin_data=True, debug=True)
    #ccf = output["CCF"]
    #plt.plot(ccf.x, ccf.y)
    #idx = numpy.argmax(ccf.y)
    #print "Maximum CCF at %g km/s" %(ccf.x[idx])
    #plt.show()

    # Add up the individual CCFs (use the Maximum Likelihood method from Zucker 2003, MNRAS, 342, 1291)
    total = corrlist[0].copy()
    total.y = numpy.ones(total.size())
    for i, corr in enumerate(corrlist):
      correlation = spline(corr.x, corr.y, k=1)
      N = self.data[i].size()
      total.y *= numpy.power(1.0 - correlation(total.x)**2, float(N)/normalization)
    master_corr = total.copy()
    master_corr.y = 1.0 - numpy.power(total.y, 1.0/float(len(corrlist)))

    idx = numpy.argmax(master_corr.y)
    rv = master_corr.x[idx]
    print "Radial velocity = %g km/s" %rv

    plt.figure(1)
    plt.plot(master_corr.x, master_corr.y, 'k-')
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("CCF")
    plt.show()

    return rv
    
    



  def UserContinuum(self, spectrum):
    """
      This will let the user click twice to define continuum points, and
    will then fit a straight line through the points as the continuum vector.
    Expects a short segment of spectrum, such that the continuum is quite linear.
    """
    self.interactive_mode = "continuum"
    fig = plt.figure(1)
    cid = fig.canvas.mpl_connect('button_press_event', self.mouseclick)
    self.clicks = []
    plt.plot(spectrum.x, spectrum.y)
    plt.draw()
    plt.waitforbuttonpress()
    plt.waitforbuttonpress()
    fig.canvas.mpl_disconnect(cid)
    plt.cla()
    # Once we get here, the user has clicked twice
    for click in self.clicks:
      print click.xdata, "\t", click.ydata
    slope = (self.clicks[1].ydata - self.clicks[0].ydata) / (self.clicks[1].xdata - self.clicks[0].xdata)
    spectrum.cont = self.clicks[0].ydata + slope*(spectrum.x - self.clicks[0].xdata)
    return spectrum



  def UserVsini(self, spectrum):
    """
      This does a Fourier transform on the spectrum, and then lets
    the user click on the first minimum, which indicates the vsini of the star.
    """
    # Set up plotting
    self.interactive_mode = "vsini"
    fig = plt.figure(1)
    cid = fig.canvas.mpl_connect('button_press_event', self.mouseclick)

    # Make wavelength spacing uniform
    xgrid = numpy.linspace(spectrum.x[0], spectrum.x[-1], spectrum.size())
    spectrum = FittingUtilities.RebinData(spectrum, xgrid)
    extend = numpy.array(40*spectrum.size()*[1,])
    spectrum.y = numpy.r_[extend, spectrum.y, extend]

    # Do the fourier transorm and keep the positive frequencies
    fft = numpy.fft.fft(spectrum.y - 1.0)
    freq = numpy.fft.fftfreq(spectrum.y.size, d=spectrum.x[1]-spectrum.x[0])
    good = numpy.where(freq > 0)[0]
    fft = fft[good].real**2 + fft[good].imag**2
    freq = freq[good]

    # Plot inside a do loop, to let user try a few times
    done = False
    trials = []
    plt.loglog(freq, fft)
    plt.xlim((1e-2, 10))
    plt.draw()
    for i in range(10):
      plt.waitforbuttonpress()
      sigma_1 = self.click.xdata
      if self.click.button == 1:
        c = constants.c.cgs.value * units.cm.to(units.km)
        vsini = 0.66*c/(spectrum.x.mean()*sigma_1)
        print "vsini = ", vsini, " km/s"
        trials.append(vsini)
        plt.cla()
      else:
        done = True
        break
        
    fig.canvas.mpl_disconnect(cid)
    if len(trials) == 1:
      return trials[0]
    
    print "\n"
    for i, vsini in enumerate(trials):
      print "\t[%i]: vsini = %.1f km/s" %(i+1, vsini)
    inp = raw_input("\nWhich vsini do you want to use (choose from the options above)? ")
    return trials[int(inp)-1]
    



  
  def mouseclick(self, event):
    """
      This is a generic mouseclick method. It will act differently
    based on what the value of self.interactive_mode is.
    """
    if self.interactive_mode == "continuum":
      if len(self.clicks) < 2:
        plt.plot(event.xdata, event.ydata, 'rx', markersize=12)
        plt.draw()
        self.clicks.append(event)

        
    elif self.interactive_mode == "vsini":
      self.click = event
    return

    
    
    
