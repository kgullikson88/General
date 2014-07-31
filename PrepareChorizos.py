"""
This script reads in BVJHK photometry from Kharchenko et al 2009,
and measured E(B-V) from my spectra. It then prepares the 
Chorizos input file
"""

import os


class StarInfo:
    def __init__(self):
        self.name = ""  # Star designation
        self.EBV = 0.0  # E(B-V)
        self.B = 0.0  # B mag
        self.B_e = 0.0  # B mag error
        self.V = 0.0  # V mag
        self.V_e = 0.0  # V mag error
        self.J = 0.0  # J mag
        self.J_e = 0.0  # J mag error
        self.H = 0.0  # H mag
        self.H_e = 0.0  # H mag error
        self.K = 0.0  # K mag
        self.K_e = 0.0  # K mag error


if __name__ == "__main__":
    phot_filename = "%s/Dropbox/School/Research/AstarStuff/TargetLists/Kharchenko2009.csv" % (os.environ["HOME"])
    red_filename = "%s/Dropbox/School/Research/AstarStuff/TargetLists/Na_Params.csv" % (os.environ["HOME"])

    all_stars = []
    full_info = []

    # Read in the photometry
    infile = open(phot_filename)
    lines = infile.readlines()
    infile.close()
    for line in lines[69:]:
        if line.strip() == "" or line.startswith("#"):
            continue
        #print line
        segments = line.split("|")
        star = StarInfo()
        star.name = segments[0].strip().replace(" ", "_")
        star.B = float(segments[7])
        star.B_e = float(segments[8])
        star.V = float(segments[9])
        star.V_e = float(segments[10])
        star.J = float(segments[11])
        star.J_e = float(segments[12])
        star.H = float(segments[13])
        star.H_e = float(segments[14])
        star.K = float(segments[15])
        star.K_e = float(segments[16])

        #Check if it is a repeat
        for s in all_stars:
            if star.name == s.name:
                print "Repeat: ", star.name
        all_stars.append(star)


    #Now, read in the reddening data
    infile = open(red_filename)
    lines = infile.readlines()
    infile.close()
    for line in lines[1:]:
        segments = line.split("|")
        name = segments[0].strip().replace(" ", "_")
        # Look for this star in the stars list
        for star in all_stars:
            if name == star.name:
                star.EBV = float(segments[3])
                full_info.append(star)


    #Finally, generate the Chorizos setup file
    for star in full_info:
        print "%s " % star.name + "%.3f " * 18 % (star.B,
                                                  star.B_e,
                                                  star.V,
                                                  star.V_e,
                                                  star.J,
                                                  star.J_e,
                                                  star.H,
                                                  star.H_e,
                                                  star.K,
                                                  star.K_e,
                                                  8000.0,
                                                  30000.0,
                                                  3.0,
                                                  5.0,
                                                  -1.0,
                                                  0.0,
                                                  star.EBV,
                                                  star.EBV)

