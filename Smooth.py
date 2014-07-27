import FittingUtilities
import HelperFunctions
from collections import Counter
from sklearn.gaussian_process import GaussianProcess


def SmoothData(order, windowsize=91, smoothorder=5, lowreject=3, highreject=3, numiters=10):
  denoised = HelperFunctions.Denoise(order.copy())
  denoised.y = FittingUtilities.Iterative_SV(denoised.y, windowsize, smoothorder, lowreject=lowreject, highreject=highreject, numiters=numiters)
  denoised.y /= denoised.y.max()
  return denoised
  

def GPSmooth(data, low=0.1, high=10, debug=False):
  """
  This will smooth the data using Gaussian processes. It will find the best
  smoothing parameter via cross-validation to be between the low and high.

  The low and high keywords are reasonable bounds for  A and B stars with 
  vsini > 100 km/s.
  """

  smoothed = data.copy()

  # First, find outliers by doing a guess smooth
  smoothed = SmoothData(data, normalize=False)
  temp = smoothed.copy()
  temp.y = data.y/smoothed.y
  temp.cont = FittingUtilities.Continuum(temp.x, temp.y, lowreject=2, highreject=2, fitorder=3)
  outliers = HelperFunctions.FindOutliers(temp, numsiglow=3, expand=5)
  data.y[outliers] = smoothed.y[outliers]
    
  gp = GaussianProcess(corr='squared_exponential',
                       theta0 = np.sqrt(low*high),
                       thetaL = low,
                       thetaU = high,
                       normalize = False,
                       nugget = (data.err / data.y)**2,
                       random_start=1)
  try:
    gp.fit(data.x[:,None], data.y)
  except ValueError:
    #On some orders with large telluric residuals, this will fail.
    # Just fall back to the old smoothing method in that case.
    return SmoothData(data), 91
  if debug:
    print "\tSmoothing parameter theta = ", gp.theta_
  smoothed.y, smoothed.err = gp.predict(data.x[:,None], eval_MSE=True)
  return smoothed, gp.theta_[0][0]
