import sys
import os
import logging

from astroquery.simbad import Simbad
import pandas as pd
from astropy.io import fits

Simbad.SIMBAD_URL = 'http://simbak.cfa.harvard.edu/simbad/sim-script'
Simbad.TIMEOUT = 120
Simbad.add_votable_fields('sp', 'flux(V)', 'flux(K)', 'plx')

data_cache = {}

class stardata:
    def __init__(self):
        self.main_id = ''
        self.spectype = ""
        self.Vmag = 0.0
        self.Kmag = 0.0
        self.ra = ""
        self.dec = ""
        self.par = 0.0  # parallax in milli-arcseconds

def GetData(starname, safe_spt=False):
    """
    Search simbad for information about the given star. 
    :param starname: A simbad-queryable name for the star
    :param safe_spt: If True, convert spectral types with 'm' in them to '5': eg. 'Am' --> 'A5'
    """
    if starname in data_cache:
        return data_cache[starname]

    star = Simbad.query_object(starname)
    data = stardata()
    data.main_id = star['MAIN_ID'].item()
    data.spectype = star['SP_TYPE'].item()
    if safe_spt:
        data.spectype = data.spectype.replace('m', '5')
    data.Vmag = star['FLUX_V'].item()
    data.Kmag = star['FLUX_K'].item()
    data.ra = star['RA'].item().strip().replace(' ', ':')
    data.dec = star['DEC'].item().strip().replace(' ', ':')
    data.par = star['PLX_VALUE'].item()
    data_cache[starname] = data
    return data


def get_vsini(file_list):
    """
    Get the vsini for every fits file in file_list. Uses the OBJECT keyword and a pre-tabulated vsini table
    :param file_list:
    :return:
    """
    homedir = os.environ['HOME']
    vsini = pd.read_csv("{}/School/Research/Useful_Datafiles/Vsini.csv".format(homedir), sep='|', skiprows=8)[1:]
    vsini_dict = {}
    prim_vsini = []
    for fname in file_list:
        root = fname.split('/')[-1][:9]
        if root in vsini_dict:
            prim_vsini.append(vsini_dict[root])
        else:
            header = fits.getheader(fname)
            star = header['OBJECT']
            try:
                v = vsini.loc[vsini.Identifier.str.strip() == star]['vsini(km/s)'].values[0]
                prim_vsini.append(float(v) * 0.8)
                vsini_dict[root] = float(v) * 0.8
            except IndexError:
                logging.warn('No vsini found for star {}! No primary star removal will be attempted!'.format(star))
                prim_vsini.append(None)
    for fname, vsini in zip(file_list, prim_vsini):
        print fname, vsini
    return prim_vsini


if __name__ == "__main__":
    for starname in sys.argv[1:]:
        data = GetData(starname)
        print starname, data.spectype
    
    
