import FittingUtilities


def SmoothData(order, windowsize=91, smoothorder=5, lowreject=3, highreject=3, numiters=10):
  denoised = FittingUtilities.Denoise3(order.copy())
  denoised.y = FittingUtilities.Iterative_SV(denoised.y, windowsize, smoothorder, lowreject=lowreject, highreject=highreject, numiters=numiters)
  denoised.y /= denoised.y.max()
  return denoised
  
