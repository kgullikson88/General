import sys
import pySIMBAD as sim

class stardata:
  def __init__(self):
    self.spectype = ""
    self.Vmag = 0.0
    self.ra = ""
    self.dec = ""


def GetData(starname):
  link = sim.buildLink(starname)
  star = sim.simbad(link)
  data = stardata()
  data.spectype = star.SpectralType()
  data.Vmag = star.flux()["V"]
  coord = star.fk5()
  data.ra = star.ra(coord)
  data.dec = star.dec(coord)
  return data


if __name__ == "__main__":
  for starname in sys.argv[1:]:
    data = GetData(starname)
    print starname, data.spectype
    
    
