"""
  This is a class to interpolate TLUSTY spectra
  WARNING! I HAVE DISABLED THE INTERPOLATION OVER MACROTURBULENT VELOCITY!
    UN-COMMENT THE RELEVANT LINES TO RE-ENABLE THE (UNTESTED) VERSION!
"""


import numpy
from scipy.interpolate import UnivariateSpline, LinearNDInterpolator
import os
from collections import defaultdict
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import DataStructures
from astropy import units, constants
from MakeModel import RebinData

homedir = os.environ["HOME"] + "/"
model_directory = "/Volumes/Time Machine Backups/Stellar_Models/TLUSTY/"

class Models:
  def __init__(self, modeldir=model_directory, xgrid=numpy.arange(300, 1000, 0.001), debug=False): 
    if not modeldir.endswith("/"):
      modeldir = modeldir + "/"

    #Set up dictionaries to hold the filenames and data for each model
    #Both dicts are keyed on temperature, then metallicity, then log(g), 
    #  and finally the microturbulent velocity
    self.model_dict = defaultdict(lambda : defaultdict( lambda: defaultdict( lambda: defaultdict(str) )))  
    self.read_dict = defaultdict(lambda : defaultdict( lambda: defaultdict( lambda: defaultdict(bool) )))  
    self.models = defaultdict(lambda : defaultdict( lambda: defaultdict( lambda: defaultdict(DataStructures.xypoint) ))) 

    self.modeldir = modeldir
    self.xgrid = xgrid
    self.debug = debug
    self.Tstep = 1000              #Hard-coded steps for now. Might be bad...
    self.Zstep = numpy.log10(2.0)
    self.gstep = 0.25
    self.vstep = 8.0

    
    self.grid = []
    self.Tgrid = []
    self.Zgrid = []
    self.ggrid = []
    self.vgrid = []
    allfiles = os.listdir(modeldir)
    for fname in allfiles:
      if fname.startswith("B") and not fname.endswith(".tar") and "vis.7" in fname and not "CN" in fname:
        mchar = fname[1]
        if mchar == "G":
          metal = 0.0
        elif mchar == "C":
          metal = numpy.log10(2.0)
        elif mchar == "L":
          metal = numpy.log10(0.5)
        else:
          print "metallicity code for file %s not recognized!" %fname
          continue

        try:
          temperature = float(fname[2:].split("g")[0])
          logg = float(fname[2:].split("g")[1].split("v")[0]) / 100.0
          vmacro = float(fname[2:].split("v")[1].split(".")[0])
          self.grid.append((temperature, metal, logg, vmacro))
          self.model_dict[temperature][metal][logg][vmacro] = modeldir + fname
          if temperature not in self.Tgrid:
            self.Tgrid.append(temperature)
          if metal not in self.Zgrid:
            self.Zgrid.append(metal)
          if logg not in self.ggrid:
            self.ggrid.append(logg)
          if vmacro not in self.vgrid:
            self.vgrid.append(vmacro)
        except ValueError:
          print "Un-parsable filename: %s" %fname
          

  #Function to read in a model with temperature T
  def GetSpectrum(self, T, metal, logg, vmacro):
    #Make sure it is not already read in
    if self.read_dict[T][metal][logg][vmacro]:
      if self.debug:
        print "Model with T = %g and log(g) = %g already read. Skipping..." %(T, logg)
      return self.models[T][metal][logg][vmacro]
    elif self.model_dict[T][metal][logg][vmacro] == '':
      if self.debug:
        print "No model found with: \n\tT=%f\n\t[Fe/H]=%f\n\tlog(g)=%f\n\tvmacro=%f" %(T, metal, logg, vmacro)
      raise ValueError
    else:
      #If we get here, the model does exist and has not yet been read in.
      fluxfile = self.model_dict[T][metal][logg][vmacro]
      contfile = fluxfile.replace("vis.7", "vis.17")
      x,y = numpy.loadtxt(fluxfile, usecols=(0,1), unpack=True)
      x2,c = numpy.loadtxt(contfile, usecols=(0,1), unpack=True)
      flux = DataStructures.xypoint(x=x*units.angstrom.to(units.nm), y=y)
      cont = DataStructures.xypoint(x=x2*units.angstrom.to(units.nm), y=c)
      flux = RebinData(flux, self.xgrid)
      cont = RebinData(cont, self.xgrid)
      model = DataStructures.xypoint(x=self.xgrid, y=flux.y, cont=cont.y)
      self.read_dict[T][metal][logg][vmacro] = True
      self.models[T][metal][logg][vmacro] = model
      return model
      
      
  

  #Does a linear Taylor expansion to get the approximate spectrum for any values
  def GetArbitrarySpectrum(self, T, metal, logg, vmacro):
    #First, find the closest values to those requested
    bestindex = 0
    distance = 9e9
    #Weight so that temperature and metallicity are most important
    weights = (10, 10, 1, 0.1)
    vmacro = 2.0  #UN-COMMENT THIS TO ENABLE MACROTURBULENT VELOCITY GRID!
    vmacro0 = 2.0  #This too!!
    
    for idx, gridpoint in enumerate(self.grid):
      d = ((T-gridpoint[0])**2 * weights[0] + 
          (metal-gridpoint[1])**2 * weights[1] + 
          (logg-gridpoint[2])**2 * weights[2] + 
          (vmacro-gridpoint[3])**2 * weights[3])
      if d < distance:
        distance = d
        bestindex = idx

    #Here are the closest grid points to those requested
    T0, metal0, logg0, vmacro0 = self.grid[bestindex][0], self.grid[bestindex][1], self.grid[bestindex][2], self.grid[bestindex][3]

    #Now, we need to get grid points that will give us appropriate slopes for the Taylor series
    T1 = T0 + self.Tstep if T0 < T else T0 - self.Tstep
    metal1 = metal0 + self.Zstep if metal0 < metal else metal0 - self.Zstep
    logg1 = logg0 + self.gstep if logg0 < logg else logg0 - self.gstep
    #vmacro1 = vmacro0 + self.vstep if vmacro0 < vmacro else vmacro0 - self.vstep

    #Make sure the grid points exist. Also make sure T0 et al isn't on the edge of the grid
    if self.model_dict[T1][metal0][logg0][vmacro0] == '':
      #Temperature point does not exist!
      if T0 == self.Tgrid[0]:
        T1 = self.Tgrid[1]
      elif T0 == self.Tgrid[-1]:
        T1 = self.Tgrid[-2]
      else:
        print "Error! Suitable grid points not found for T = %g" %T
        print "  when looking for T = %g" %T1
        print "Returning the closest grid point instead, which has parameters:"
        print "\tT=%f\n\t[Fe/H]=%f\n\tlog(g)=%f\n\tvmacro=%f" %(T0, metal0, logg0, vmacro0)
        return self.GetSpectrum(T0, metal0, logg0, vmacro0)
      
    if self.model_dict[T0][metal1][logg0][vmacro0] == '':
      #Temperature point does not exist!
      if metal0 == self.Zgrid[0]:
        metal1 = self.Zgrid[1]
      elif metal0 == self.Zgrid[-1]:
        metal1 = self.Zgrid[-2]
      else:
        print "Error! Suitable grid points not found for [Fe/H] = %g" %metal
        print "  when looking for [Fe/H] = %g" %metal1
        print "Returning the closest grid point instead, which has parameters:"
        print "\tT=%f\n\t[Fe/H]=%f\n\tlog(g)=%f\n\tvmacro=%f" %(T0, metal0, logg0, vmacro0)
        return self.GetSpectrum(T0, metal0, logg0, vmacro0)
      
    if self.model_dict[T0][metal0][logg1][vmacro0] == '':
      #Temperature point does not exist!
      if logg0 == self.ggrid[0]:
        logg1 = self.ggrid[1]
      elif logg0 == self.ggrid[-1]:
        logg1 = self.ggrid[-2]
      else:
        print "Error! Suitable grid points not found for log(g) = %g" %logg
        print "  when looking for log(g) = %g" %logg1
        print "Returning the closest grid point instead, which has parameters:"
        print "\tT=%f\n\t[Fe/H]=%f\n\tlog(g)=%f\n\tvmacro=%f" %(T0, metal0, logg0, vmacro0)
        return self.GetSpectrum(T0, metal0, logg0, vmacro0)

    """
    if self.model_dict[T0][metal0][logg0][vmacro1] == '':
      #Temperature point does not exist!
      if vmacro0 == self.vgrid[0]:
        vmacro1 = self.vgrid[1]
      elif vmacro0 == self.vgrid[-1]:
        vmacro1 = self.vgrid[-2]
      else:
        print "Error! Suitable grid points not found for v_macro = %g" %vmacro
        print "  when looking for v_macro = %g" %vmacro1
        print "Returning the closest grid point instead, which has parameters:"
        print "\tT=%f\n\t[Fe/H]=%f\n\tlog(g)=%f\n\tvmacro=%f" %(T0, metal0, logg0, vmacro0)
        return self.GetSpectrum(T0, metal0, logg0, vmacro0)
    """

    #Not, get all of the spectra we need.
    models = defaultdict(lambda : defaultdict( lambda: defaultdict( lambda: defaultdict(DataStructures.xypoint) )))
    models[T0][metal0][logg0][vmacro0] = self.GetSpectrum(T0, metal0, logg0, vmacro0)
    models[T1][metal0][logg0][vmacro0] = self.GetSpectrum(T1, metal0, logg0, vmacro0)
    models[T0][metal1][logg0][vmacro0] = self.GetSpectrum(T0, metal1, logg0, vmacro0)
    models[T0][metal0][logg1][vmacro0] = self.GetSpectrum(T0, metal0, logg1, vmacro0)
    #models[T0][metal0][logg0][vmacro1] = self.GetSpectrum(T0, metal0, logg0, vmacro1)

    #Make the derivatives for the Taylor series
    d_dT = (models[T0][metal0][logg0][vmacro0].y - models[T1][metal0][logg0][vmacro0].y) / (T0-T1)
    d_dZ = (models[T0][metal0][logg0][vmacro0].y - models[T0][metal1][logg0][vmacro0].y) / (metal0-metal1)
    d_dg = (models[T0][metal0][logg0][vmacro0].y - models[T0][metal0][logg1][vmacro0].y) / (logg0-logg1)
    #d_dv = (models[T0][metal0][logg0][vmacro0].y - models[T0][metal0][logg0][vmacro1].y) / (vmacro0-vmacro1)

    newspectrum = models[T0][metal0][logg0][vmacro0].copy()
    newspectrum.y += (d_dT * (T - T0) +
                      d_dZ * (metal - metal0) +
                      d_dg * (logg - logg0) )#   Need to remove this parenthesis to enable macroturbulent velocity too!+
    #d_dv * (vmacro0 - vmacro1))

    #Do the same thing for the continuum vectors
    d_dT = (models[T0][metal0][logg0][vmacro0].cont - models[T1][metal0][logg0][vmacro0].cont) / (T0-T1)
    d_dZ = (models[T0][metal0][logg0][vmacro0].cont - models[T0][metal1][logg0][vmacro0].cont) / (metal0-metal1)
    d_dg = (models[T0][metal0][logg0][vmacro0].cont - models[T0][metal0][logg1][vmacro0].cont) / (logg0-logg1)
    #d_dv = (models[T0][metal0][logg0][vmacro0].cont - models[T0][metal0][logg0][vmacro1].cont) / (vmacro0-vmacro1)

    newspectrum.cont += (d_dT * (T - T0) +
                      d_dZ * (metal - metal0) +
                      d_dg * (logg - logg0) )#   Need to remove this parenthesis to enable macroturbulent velocity too!+
    #d_dv * (vmacro0 - vmacro1))

    return newspectrum
      


  """
  def GetInterpolatedSpectrum(self, T, metal, logg, vmacro):
    #First, read in all the spectra within a smaller grid surrounding the requested point
    gridpoints = []
    spectra = []
    flux = []
    Temps = sorted(self.model_dict.keys())
    left = max(0, numpy.searchsorted(Temps, T-1000) - 1)
    right = min(len(Temps), numpy.searchsorted(Temps, T+1000) + 1)
    for Ti in Temps[left:right]:
      #print Ti
      metallicities = sorted(self.model_dict[Ti].keys())
      left = max(0, numpy.searchsorted(metallicities, metal-1) - 1)
      right = min(len(metallicities), numpy.searchsorted(metallicities, metal+1) + 1)
      for Zi in metallicities[left:right]:
        #print "\t", Zi
        grav = sorted(self.model_dict[Ti][Zi].keys())
        left = max(0, numpy.searchsorted(grav, logg-0.5) - 1)
        right = min(len(grav), numpy.searchsorted(grav, logg+0.5) + 1)
        for g in grav[left:right]:
          #print "\t\t", g
          vm = sorted(self.model_dict[Ti][Zi][g].keys())
          left = max(0, numpy.searchsorted(grav, vmacro-1) - 1)
          right = min(len(vm), numpy.searchsorted(grav, vmacro+1) + 1)
          #print "\t\t", left, right, vm
          for v in vm[left:right]:
            #print "\t\t\t", v
            print "Reading model with parameters: "
            print "\tT=%f\n\t[Fe/H]=%f\n\tlog(g)=%f\n\tvmacro=%f" %(Ti, Zi, g, v)
            spectra.append(self.GetSpectrum(Ti, Zi, g, v))
            gridpoints.append(((Ti, Zi, g, v)))
    


    #Now, interpolate each point
    print "Grid read in. Interpolating now..."
    output = DataStructures.xypoint(x=self.xgrid)
    for i in range(self.xgrid.size):
      flux = []
      cont = []
      for s in spectra:
        flux.append(s.y[i])
        cont.append(s.cont[i])
      outfile = open("Initial_grid.dat", "w")
      for i in range(len(flux)):
        outfile.write("%g\t%g\t%g\t%g\t%g\t%g\n" %(gridpoints[i][0], gridpoints[i][1], gridpoints[i][2], gridpoints[i][3], flux[i], cont[i]))
      outfile.close()
      sys.exit()
      f = LinearNDInterpolator(gridpoints, flux)
      c = LinearNDInterpolator(gridpoints, cont)
      output.y[i] = f(output.x)
      output.cont[i] = c(output.x)
    return output

  """



  #Function to linearly interpolate to a given Temperature and log(g)
  def GetClosestSpectrum(self, T, metal, logg, vmacro, weights=None):
    bestindex = 0
    distance = 9e9
    if weights == None or not (type(weights) == list or type(weights) == tuple or isinstance(weights, numpy.ndarray))  or len(weights) < 4:
      weights = numpy.ones(4)
    
    for idx, gridpoint in enumerate(self.grid):
      d = ((T-gridpoint[0])**2 * weights[0] + 
          (metal-gridpoint[1])**2 * weights[1] + 
          (logg-gridpoint[2])**2 * weights[2] + 
          (vmacro-gridpoint[3])**2 * weights[3])
      if d < distance:
        distance = d
        bestindex = idx
    
    T, metal, logg, vmacro = self.grid[bestindex][0], self.grid[bestindex][1], self.grid[bestindex][2], self.grid[bestindex][3]
    
    print "Closest spectrum has the following parameters:"
    print "\tT=%f\n\t[Fe/H]=%f\n\tlog(g)=%f\n\tvmacro=%f" %(T, metal, logg, vmacro)

    return self.GetSpectrum(T, metal, logg, vmacro)

    
   
