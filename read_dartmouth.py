import os
import pandas
import numpy as np
from collections import defaultdict

DEFAULT_GRID_DIR = "{}/Dropbox/School/Research/Stellar_Evolution/Dartmouth/".format(os.environ["HOME"])
DEFAULT_TRACK_DIR = "{}tracks/".format(DEFAULT_GRID_DIR)
DEFAULT_ISO_DIR = "{}isochrones/UBVRIJHKsKp/".format(DEFAULT_GRID_DIR)


def read_tracks(feh, afe, y=0, grid_dir=DEFAULT_TRACK_DIR):
    """
    Reads in evolutionary tracks for all the masses with the given parameters
    :params feh: [Fe/H] to use
    :params afe: [alpha/Fe] to use
    :params y: helium abundance (default = 0)
    :grid_dir: directory of the grid
    """
    if not grid_dir.endswith("/"):
        grid_dir += "/"

    # First, read in the grid to see what we have
    full_grid = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    grid_directories = [d for d in os.listdir(grid_dir) if "feh" in d]
    for directory in grid_directories:
        feh_i, afe_i, y_i = classify_file(directory)
        full_grid[feh_i][afe_i][y_i] = "{}{}/".format(grid_dir, directory)

    #Check to make sure the requested parameters exist
    if not check_exists(full_grid.keys(), feh, '[Fe/H]'):
        raise KeyError("Bad [Fe/H] value")
    if not check_exists(full_grid[feh].keys(), afe, '[alpha/Fe]'):
        raise KeyError("Bad [alpha/Fe] value")
    if not check_exists(full_grid[feh][afe].keys(), feh, 'Y'):
        raise KeyError("Bad Y value")

    #Finally, read in the grid
    age, mass, logT, logg, logL = read_tracks_files(full_grid[feh][afe][y])

    df = pandas.DataFrame(data={"age": age,
                                "mass": mass,
                                "logT": logT,
                                "logg": logg,
                                "logL": logL})
    return df


def read_iso(feh, afe, y=0, grid_dir=DEFAULT_ISO_DIR):
    """
    Reads in dartmouth isochrones of the appropriate parameters
    :params feh: [Fe/H] to use
    :params afe: [alpha/Fe] to use
    :params y: helium abundance (default = 0)
    :grid_dir: directory of the grid

    returns: a pandas dataframe containing the grid
    """
    if not grid_dir.endswith("/"):
        grid_dir += "/"

    # First, read in the grid to see what we have
    full_grid = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gridfiles = [f for f in os.listdir(grid_dir) if "feh" in f]
    for fname in gridfiles:
        feh_i, afe_i, y_i = classify_file(fname)
        full_grid[feh_i][afe_i][y_i].append("{}{}".format(grid_dir, fname))

    #Check to make sure the requested parameters exist
    if not check_exists(full_grid.keys(), feh, '[Fe/H]'):
        raise KeyError("Bad [Fe/H] value")
    if not check_exists(full_grid[feh].keys(), afe, '[alpha/Fe]'):
        raise KeyError("Bad [alpha/Fe] value")
    if not check_exists(full_grid[feh][afe].keys(), feh, 'Y'):
        raise KeyError("Bad Y value")

    #Read in the grid. It might have multiple files (different ages)
    age = []
    mass = []
    logT = []
    logg = []
    logL = []
    for fname in full_grid[feh][afe][y]:
        age, mass, logT, logg, logL = read_isofile(fname, age=age, mass=mass, logT=logT, logg=logg, logL=logL)

    df = pandas.DataFrame(data={"age": age,
                                "mass": mass,
                                "logT": logT,
                                "logg": logg,
                                "logL": logL})

    return df


def classify_file(filename):
    """
    Classifies the [Fe/H], [alpha/Fe], and helium abundance, given the filename
    """
    feh_str = filename[3:6]
    sign = 1.0 if feh_str[0] == "p" else -1.0
    feh = sign * float(feh_str[1:]) / 10.0

    afe_str = filename[9:11]
    sign = 1.0 if afe_str[0] == "p" else -1.0
    afe = sign * float(afe_str[1]) / 10.0

    if len(filename) > 14 and filename[11] == "y":
        y = float(filename[12:14]) / 100.0
    else:
        y = 0.0

    return feh, afe, y


def check_exists(l, key, name):
    if key not in l:
        print "The requested {} ({}) does not exist.\nValid keys are:".format(name, key)
        for val in sorted(l):
            print "\t{}".format(val)
        return False
    return True


def read_isofile(filename, age=[], mass=[], logT=[], logg=[], logL=[]):
    """
    Read an isochrone file
    """
    infile = open(filename)
    lines = infile.readlines()
    infile.close()

    current_age = 0
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            if "AGE" in line:
                age_str = line.split("=")[1].split()[0]
                current_age = float(age_str)
        else:
            segments = line.split()
            age.append(current_age)
            mass.append(float(segments[1]))
            logT.append(float(segments[2]))
            logg.append(float(segments[3]))
            logL.append(float(segments[4]))

    return age, mass, logT, logg, logL


def read_tracks_files(directory):
    """
    Reads in all the files in a given directory. They are assumed to be evolutionary tracks
    """
    allfiles = os.listdir(directory)
    age = []
    mass = []
    logT = []
    logg = []
    logL = []
    for fname in allfiles:
        star_mass = float(fname[1:].split("feh")[0]) / 100.0
        a, T, g, L = np.loadtxt("{}{}".format(directory, fname), usecols=(0, 1, 2, 3), unpack=True)
        age.extend(a)
        mass.extend([star_mass] * a.size)
        logT.extend(T)
        logg.extend(g)
        logL.extend(L)

    return age, mass, logT, logg, logL
