import sys
import matplotlib.pyplot as plt



def MakePlot(x, y, title="", fig=None, labelsize=20, titlesize=25, plotcolor='black', linewidth=2):
  if fig == None:
    fig = plt.figure()
  elif isinstance(fig, int):
    fig = plt.figure(fig)

  plt.plot(x, y, color=plotcolor, lw=linewidth)
  plt.xlabel("Velocity (km/s)", fontsize=labelsize)
  plt.ylabel("CCF", fontsize=labelsize)
  plt.title(title, fontsize=titlesize)

  ax = fig.gca()
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(labelsize)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(labelsize)
  plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95)
  return fig
