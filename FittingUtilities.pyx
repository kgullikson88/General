"""
Just a set of useful functions that I use often while fitting
"""
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline as smoother
from scipy.linalg import solve_banded
import scipy.stats
from scipy.signal import argrelmin
import DataStructures
from astropy import units, constants
import scipy.signal as sig
import pywt
import mlpy
import matplotlib.pyplot as plt
import functools
import MakeModel
import emcee


#Define bounding functions:
# lower bound:            lbound(boundary_value, parameter)
# upper bound:            ubound(boundary_value, parameter)
# lower and upper bounds: bound([low, high], parameter)
# fixed parameter:        fixed(fixed_value, parameter)
lbound = lambda p, x: 1e8*numpy.sqrt(p-x) + 1e-3*(p-x) if (x<p) else 0
ubound = lambda p, x: 1e8*numpy.sqrt(x-p) + 1e-3*(x-p) if (x>p) else 0
bound  = lambda p, x: lbound(p[0],x) + ubound(p[1],x)
fixed  = lambda p, x: bound((p,p), x)


#CCImprove
"""
  Improve the wavelength solution by a constant shift
"""
def CCImprove(data, model, be_safe=True, tol=0.2, debug=False):
  correction = data.y.size + float(numpy.searchsorted(model.x, data.x[0]))/2.0 - 1
  ycorr = numpy.correlate(data.y/data.cont-1.0, model.y/model.cont-1.0, mode="full")
  xcorr = numpy.arange(ycorr.size)
  lags = xcorr - correction
  distancePerLag = (data.x[-1] - data.x[0])/float(data.x.size)
  offsets = -lags*distancePerLag
  offsets = offsets[::-1]
  ycorr = ycorr[::-1]

  if be_safe:
    left = numpy.searchsorted(offsets, -tol)
    right = numpy.searchsorted(offsets, tol)
  else:
    left, right = 0, ycorr.size
    
  maxindex = ycorr[left:right].argmax() + left

  if debug:
    return offsets[maxindex], DataStructures.xypoint(x=offsets, y=ycorr)
  else:
    return offsets[maxindex]
 

"""
  This function fits the continuum spectrum by iteratively removing
points over one standard deviation below the mean, which are assumed
to be absorption lines.
"""
def Continuum(x, y, fitorder=3, lowreject=2, highreject=4, numiter=10000, function="poly"):
  done = False
  x2 = numpy.copy(x)
  y2 = numpy.copy(y)
  iteration = 0
  while not done and iteration < numiter:
    numiter += 1
    done = True
    if function == "poly":
      fit = numpy.poly1d(numpy.polyfit(x2 - x2.mean(), y2, fitorder))
    elif function == "spline":
      fit = smoother(x2, y2, s=fitorder)
    residuals = y2 - fit(x2 - x2.mean())
    mean = numpy.mean(residuals)
    std = numpy.std(residuals)
    badpoints = numpy.where(numpy.logical_or((residuals - mean) < -lowreject*std, residuals - mean > highreject*std))[0]
    if badpoints.size > 0 and x2.size - badpoints.size > 5*fitorder:
      done = False
      x2 = numpy.delete(x2, badpoints)
      y2 = numpy.delete(y2, badpoints)
  return fit(x - x2.mean())



"""
  General error function for leastsq
"""
def GeneralLSErrorFunction(data, model, normalize=True):
  if normalize:
    return (data.y/data.cont - model.y/model.cont)**2 / data.err**2
  else:
    return (data.y - model.y)**2 / data.err**2 



#Smoothing function
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1]/m.sum(), y, mode='valid')


#Iterative version of the savitzky-golay smoothing function
def Iterative_SV(y, window_size, order, lowreject=3, highreject=3, numiters=100, deriv=0, rate=1):
  done = False
  iteration = 0
  while not done and iteration < numiters:
    iteration += 1
    done = True
    smoothed = savitzky_golay(y, window_size, order, deriv, rate)
      
    reduced = y/smoothed
    sigma = numpy.std(reduced)
    mean = numpy.mean(reduced)
    badindices = numpy.where(numpy.logical_or((reduced - mean)/sigma < -lowreject, (reduced - mean)/sigma > highreject))[0]
    if badindices.size > 0:
      done = False
      y[badindices] = smoothed[badindices]
  return smoothed



"""
  Function to apply a low-pass filter to data.
    Data must be in an xypoint container, and have linear wavelength spacing
    vel is the width of the features you want to remove, in velocity space (in cm/s)
    width is how long it takes the filter to cut off, in units of wavenumber
"""
def LowPassFilter(data, vel, width=5, linearize=False):
  if linearize:
    data = data.copy()
    datafcn = spline(data.x, data.y, k=1)
    errorfcn = spline(data.x, data.err, k=1)
    contfcn = spline(data.x, data.cont, k=1)
    linear = DataStructures.xypoint(data.x.size)
    linear.x = numpy.linspace(data.x[0], data.x[-1], linear.size())
    linear.y = datafcn(linear.x)
    linear.err = errorfcn(linear.x)
    linear.cont = contfcn(linear.x)
    data = linear
    
  #Figure out cutoff frequency from the velocity.
  featuresize = data.x.mean()*vel/constants.c.cgs.value    #vel MUST be given in units of cm
  dlam = data.x[1] - data.x[0]   #data.x MUST have constant x-spacing
  Npix = featuresize / dlam
  cutoff_hz = 1.0/Npix   #Cutoff frequency of the filter
  cutoff_hz = 1.0/featuresize

  nsamples = data.size()
  sample_rate = 1.0/dlam
  nyq_rate = sample_rate / 2.0    # The Nyquist rate of the signal.
  width /= nyq_rate

  # The desired attenuation in the stop band, in dB.
  ripple_db = 60.0

  # Compute the order and Kaiser parameter for the FIR filter.
  N, beta = sig.kaiserord(ripple_db, width)

  # Use firwin with a Kaiser window to create a lowpass FIR filter.
  taps = sig.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

  #Extend data to prevent edge effects
  y = numpy.r_[data.y[::-1], data.y, data.y[::-1]]

  # Use lfilter to filter data with the FIR filter.
  smoothed_y = sig.lfilter(taps, 1.0, y)

  # The phase delay of the filtered signal.
  delay = 0.5 * (N-1) / sample_rate
  delay_idx = numpy.searchsorted(data.x, data.x[0] + delay) - 1
  smoothed_y = smoothed_y[data.size()+delay_idx:-data.size()+delay_idx]
  if linearize:
    return linear.x, smoothed_y
  else:
    return smoothed_y



def IterativeLowPass(data, vel, numiter=100, lowreject=3, highreject=3, width=5, linearize=False):
  datacopy = data.copy()
  if linearize:
    datafcn = spline(datacopy.x, datacopy.y, k=3)
    errorfcn = spline(datacopy.x, datacopy.err, k=1)
    contfcn = spline(datacopy.x, datacopy.cont, k=1)
    linear = DataStructures.xypoint(datacopy.x.size)
    linear.x = numpy.linspace(datacopy.x[0], datacopy.x[-1], linear.size())
    linear.y = datafcn(linear.x)
    linear.err = errorfcn(linear.x)
    linear.cont = contfcn(linear.x)
    #import matplotlib.pyplot as plt
    #plt.figure(10)
    #plt.plot(datacopy.x, datacopy.y)
    #plt.plot(linear.x, linear.y)
    #plt.show()
    datacopy = linear.copy()
    
  done = False
  iter = 0
  datacopy.cont = Continuum(datacopy.x, datacopy.y, fitorder=9, lowreject=2.5, highreject=5)
  while not done and iter<numiter:
    done = True
    iter += 1
    smoothed = LowPassFilter(datacopy, vel, width=width)
    residuals = datacopy.y / smoothed
    mean = numpy.mean(residuals)
    std = numpy.std(residuals)
    badpoints = numpy.where(numpy.logical_or((residuals - mean) < -lowreject*std, residuals - mean > highreject*std))[0]
    if badpoints.size > 0:
      done = False
      datacopy.y[badpoints] = smoothed[badpoints]
  if linearize:
    return linear.x, smoothed
  else:
    return smoothed




"""
  Function to apply a high-pass filter to data.
    Data must be in an xypoint container, and have linear wavelength spacing
    vel is the width of the features you want to remove, in velocity space (in cm/s)
    width is how long it takes the filter to cut off, in units of wavenumber
"""
def HighPassFilter(data, vel, width=5, linearize=False):
  if linearize:
    data = data.copy()
    datafcn = spline(data.x, data.y, k=3)
    errorfcn = spline(data.x, data.err, k=3)
    contfcn = spline(data.x, data.cont, k=3)
    linear = DataStructures.xypoint(data.x.size)
    linear.x = numpy.linspace(data.x[0], data.x[-1], linear.size())
    linear.y = datafcn(linear.x)
    linear.err = errorfcn(linear.x)
    linear.cont = contfcn(linear.x)
    data = linear
  
  #Figure out cutoff frequency from the velocity.
  featuresize = 2*data.x.mean()*vel/constants.c.cgs.value    #vel MUST be given in units of cm
  dlam = data.x[1] - data.x[0]   #data.x MUST have constant x-spacing
  Npix = featuresize / dlam

  nsamples = data.size()
  sample_rate = 1.0/dlam
  nyq_rate = sample_rate / 2.0    # The Nyquist rate of the signal.
  width /= nyq_rate
  cutoff_hz = min(1.0/featuresize, nyq_rate-width*nyq_rate/2.0)   #Cutoff frequency of the filter

  # The desired attenuation in the stop band, in dB.
  ripple_db = 60.0

  # Compute the order and Kaiser parameter for the FIR filter.
  N, beta = sig.kaiserord(ripple_db, width)
  if N%2 == 0:
    N += 1

  # Use firwin with a Kaiser window to create a lowpass FIR filter.
  taps = sig.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)

  #Extend data to prevent edge effects
  y = numpy.r_[data.y[::-1], data.y, data.y[::-1]]

  # Use lfilter to filter data with the FIR filter.
  smoothed_y = sig.lfilter(taps, 1.0, y)

  # The phase delay of the filtered signal.
  delay = 0.5 * (N-1) / sample_rate
  delay_idx = numpy.searchsorted(data.x, data.x[0] + delay) - 1
  smoothed_y = smoothed_y[data.size()+delay_idx:-data.size()+delay_idx]
  if linearize:
    return linear.x, smoothed_y
  else:
    return smoothed_y





"""
  This function implements the denoising given in the url below:
  http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4607982&tag=1

  with title "Astronomical Spectra Denoising Based on Simplifed SURE-LET Wavelet Thresholding"
"""

def Denoise(data):
  y, boolarr = mlpy.wavelet.pad(data.y)
  WC = mlpy.wavelet.dwt(y, 'd', 10, 0)
  #Figure out the unknown parameter 'a'
  sum1 = 0.0
  sum2 = 0.0
  numlevels = int(numpy.log2(WC.size))
  start = 2**(numlevels-1)
  median = numpy.median(WC[start:])
  sigma = numpy.median(numpy.abs(WC[start:] - median)) / 0.6745
  for w in WC:
    phi = w*numpy.exp(-w**2 / (12.0*sigma**2) )
    dphi = numpy.exp(-w**2 / (12.0*sigma**2) ) * (1 - 2*w**2 / (12*sigma**2) )
    sum1 += sigma**2 * dphi
    sum2 += phi**2
  a = -sum1 / sum2

  #Adjust all wavelet coefficients
  WC = WC + a*WC*numpy.exp( -WC**2 / (12*sigma**2) )

  #Now, do a soft threshold
  threshold = scipy.stats.scoreatpercentile(WC, 80.0)
  WC[numpy.abs(WC) <= threshold] = 0.0
  WC[numpy.abs(WC) > threshold] -= threshold*numpy.sign(WC[numpy.abs(WC) > threshold])

  #Transform back
  y2 = mlpy.wavelet.idwt(WC, 'd', 10)
  data.y = y2[boolarr]
  return data



#Kept for legacy support, since I was using Denoise3 in several codes in the past.
def Denoise3(data):
  return Denoise(data)
  



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
  lines = list(argrelmin(spectrum.y, order=N)[0])
  for i in range(len(lines)-1, -1, -1):
    idx = lines[i]
    xval = spectrum.x[idx]
    yval = spectrum.y[idx]
    if yval > tol:
      lines.pop(i)
    elif debug:
      plt.plot([xval, xval], [yval-0.01, yval-0.03], 'r-')
      
  if debug:
    plt.plot(spectrum.x, spectrum.y, 'k-')
    plt.title("Lines found in FittingUtilities.FindLine")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux")
    plt.show()
  return numpy.array(lines)




"""
  This function will do a Bayesian fit to the model.

  Parameter description:
    data:         A DataStructures.xypoint instance containing the data
    model_fcn:    A function that takes an x-array and parameters,
                     and returns a y-array. The number of parameters
                     should be the same as the length of the 'priors'
                     parameter
    priors:       Either a 2d numpy array or a list of lists. Each index
                     should contain the expected value and the uncertainty
                     in that value (assumes all Gaussian priors!).
    limits:       If given, it should be a list of the same shape as
                     'priors', giving the limits of each parameter
    burn_in:      The burn-in period for the MCMC before you start counting
    nwalkers:     The number of emcee 'walkers' to use.
    nsamples:     The number of samples to use in the MCMC sampling. Note that
                      the actual number of samples is nsamples * nwalkers
    nthreads:     The number of processing threads to use (parallelization)
                      This probably needs MPI installed to work. Not sure though...
    full_ouput:   Return the full sample chain instead of just the mean and
                      standard deviation of each parameter.
    a:            See emcee.EnsembleSampler. Basically, it controls the step size
"""
def BayesFit(data, model_fcn, priors, limits=None, burn_in=100, nwalkers=100, nsamples=100, nthreads=1, full_output=False, a=2):

  # Priors needs to be a numpy array later, so convert to that first
  priors = numpy.array(priors)
  
  # Define the likelihood, prior, and posterior probability functions
  likelihood = lambda pars, data, model_fcn: numpy.sum( -(data.y - model_fcn(data.x, pars))**2 / (2.0*data.err**2))
  if limits == None:
    prior = lambda pars, priors: numpy.sum( -(pars-priors[:,0])**2 / (2.0*priors[:,1]**2))
    posterior = lambda pars, data, model_fcn, priors: likelihood(pars, data, model_fcn) + prior(pars, priors)
  else:
    limits = numpy.array(limits)
    prior = lambda pars, priors, limits: -9e19 if any(numpy.logical_or(pars<limits[:,0], pars>limits[:,1])) else numpy.sum( -(pars-priors[:,0])**2 / (2.0*priors[:,1]**2))
    posterior = lambda pars, data, model_fcn, priors, limits: likelihood(pars, data, model_fcn) + prior(pars, priors, limits)

    
  # Set up the MCMC sampler
  ndim = priors.shape[0]
  if limits == None:
    p0 = [numpy.random.normal(loc=priors[:,0], scale=priors[:,1]) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, threads=nthreads, args=(data, model_fcn, priors), a=4)
  else:
    ranges = numpy.array([l[1] - l[0] for l in limits])
    p0 = [numpy.random.rand(ndim)*ranges+limits[:,0] for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, threads=nthreads, args=(data, model_fcn, priors, limits), a=a)

  # Burn-in the sampler
  pos, prob, state = sampler.run_mcmc(p0, burn_in)

  # Reset the chain to remove the burn-in samples.
  sampler.reset()

  # Run the sampler
  pos, prob, state = sampler.run_mcmc(pos, nsamples, rstate0=state)


  print "Acceptance fraction = %f" %numpy.mean(sampler.acceptance_fraction)
  maxprob_indice = numpy.argmax(prob)
  priors[:,0] = pos[maxprob_indice]
  #Get the parameter estimates
  chain = sampler.flatchain
  for i in range(ndim):
    priors[i][1] = numpy.std(chain[:,i])

  if full_output:
    return priors, chain
  return priors


