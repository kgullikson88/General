import sys
import pySIMBAD as sim

class stardata:
  def __init__(self):
    self.spectype = ""
    self.Vmag = 0.0
    self.ra = ""
    self.dec = ""
    self.par = 0.0   #parallax in arcseconds


def GetData(starname):
  link = sim.buildLink(starname, cfa_mirror=True)
  star = sim.simbad(link)
  data = stardata()
  data.spectype = star.SpectralType()
  data.Vmag = star.flux()["V"]
  coord = star.fk5()
  data.ra = star.ra(coord)
  data.dec = star.dec(coord)
  try:
    data.par = star.Parallax()
  except: data.par = 10.0  #Just guess. This is roughly the median for my sample
  return data


if __name__ == "__main__":
  for starname in sys.argv[1:]:
    data = GetData(starname)
    print starname, data.spectype
    
    
