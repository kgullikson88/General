import sys

from astroquery.simbad import Simbad

Simbad.add_votable_fields('sp', 'flux(V)', 'plx')

class stardata:
    def __init__(self):
        self.main_id = ''
        self.spectype = ""
        self.Vmag = 0.0
        self.ra = ""
        self.dec = ""
        self.par = 0.0  # parallax in arcseconds

def GetData(starname):
    star = Simbad.query_object(starname)
    data = stardata()
    data.main_id = star['MAIN_ID'].item()
    data.spectype = star['SP_TYPE'].item()
    data.Vmag = star['FLUX_V'].item()
    data.ra = star['RA'].item().strip().replace(' ', ':')
    data.dec = star['DEC'].item().strip().replace(' ', ':')
    data.par = star['PLX_VALUE'].item()
    return data


if __name__ == "__main__":
    for starname in sys.argv[1:]:
        data = GetData(starname)
        print starname, data.spectype
    
    
