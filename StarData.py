import sys

from astroquery.simbad import Simbad

Simbad.add_votable_fields('sp', 'flux(V)', 'plx')

class stardata:
    def __init__(self):
        self.spectype = ""
        self.Vmag = 0.0
        self.ra = ""
        self.dec = ""
        self.par = 0.0  # parallax in arcseconds


def GetData(starname):
    star = Simbad.query_object(starname)
    data = stardata()
    data.spectype = star['SP_TYPE'].item()
    data.Vmag = star['FLUX_V'].item()
    data.ra = star['RA'].item().strip().replace(' ', ':')
    data.dec = star['DEC'].item().strip().replace(' ', ':')
    data.par = star['PLX_VALUE'].item()
    """
    try:
        link = sim.buildLink(starname, cfa_mirror=True)
        star = sim.simbad(link)
    except sim.ConnectionError:
        link = sim.buildLink(starname, cfa_mirror=False)
	star = sim.simbad(link)
    data = stardata()
    data.spectype = star.SpectralType()
    data.Vmag = star.flux()["V"]
    coord = star.fk5()
    data.ra = star.ra(coord)
    data.dec = star.dec(coord)
    try:
        data.par = star.Parallax()
    except:
        data.par = 10.0  # Just guess. This is roughly the median for my sample
    """
    return data


if __name__ == "__main__":
    for starname in sys.argv[1:]:
        data = GetData(starname)
        print starname, data.spectype
    
    
