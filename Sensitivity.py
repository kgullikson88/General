from __future__ import division, print_function

import os
import sys
from re import search
from collections import defaultdict
import itertools
import logging
import FittingUtilities
from astropy import units, constants

from scipy.interpolate import InterpolatedUnivariateSpline as interp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.io import ascii
from astropy.analytic_functions import blackbody_lambda
import h5py
import seaborn as sns

import DataStructures
import GenericSearch
import StellarModel
import StarData
import SpectralTypeRelations
from PlotBlackbodies import Planck
import GenericSmooth
import HelperFunctions
import Broaden
import Correlate
import EstimateDetection
import Mamajek_Table


sns.set_context('paper', font_scale=1.5)

MS = SpectralTypeRelations.MainSequence()

# Define some constants to use
lightspeed = constants.c.cgs.value * units.cm.to(units.km)


def GetFluxRatio(sptlist, Tsec, xgrid):
    """
      Returns the flux ratio between the secondary star of temperature Tsec
      and the (possibly multiple) primary star(s) given in the
      'sptlist' list (given as spectral types)
      xgrid is a np.ndarray containing the x-coordinates to find the
        flux ratio at (in nm)

    """
    prim_flux = np.zeros(xgrid.size)

    # Determine the flux from the primary star(s)
    for spt in sptlist:
        end = search("[0-9]", spt).end()
        T = MS.Interpolate(MS.Temperature, spt[:end])
        R = MS.Interpolate(MS.Radius, spt[:end])
        prim_flux += Planck(xgrid * units.nm.to(units.cm), T) * R ** 2

    # Determine the secondary star flux
    s_spt = MS.GetSpectralType(MS.Temperature, Tsec)
    R = MS.Interpolate(MS.Radius, s_spt)
    sec_flux = Planck(xgrid * units.nm.to(units.cm), Tsec) * R ** 2

    return sec_flux / prim_flux


def GetMass(spt):
    """
    Returns the mass of the system in solar masses
    spt: Spectral type of the star
    """

    # Get temperature
    end = search("[0-9]", spt).end()
    return MS.Interpolate(MS.Mass, spt[:end])


def Analyze_Old(fileList,
                vsini_secondary=10 * units.km.to(units.cm),
                resolution=None,
                smooth_factor=0.8,
                vel_list=range(-400, 450, 50),
                companion_file="%s/Dropbox/School/Research/AstarStuff/TargetLists/Multiplicity.csv" % (
                os.environ["HOME"]),
                vsini_file="%s/School/Research/Useful_Datafiles/Vsini.csv" % (os.environ["HOME"]),
                tolerance=5.0,
                vsini_skip=10,
                vsini_idx=1,
                badregions=[],
                trimsize=1,
                object_keyword="object",
                modeldir="./",
                addmode="ml",
                debug=False):
    """
      Perform the sensitivity analysis.
    :param fileList:  The files to analyze
    :param vsini_secondary:   The vsini of the secondary star
    :param resolution: The detector resolution
    :param smooth_factor: The factor to multiply the vsini of the primary by when smoothing
    :param vel_list: The list of velocities to try
    :param companion_file: A csv file giving the known companions to each primary star
    :param vsini_file: A csv file giving the vsini of each primary star
    :param vsini_skip: How many lines to skip in vsini_file
    :param vsini_idx: What index the vsini is in in the vsini_file
    :param tolerance: How far off the peak can be and still be counted as a detection (default: 5 km/s)
    :param badregions: Regions to ignore. These should be the same as in Search!
    :param trimsize: The number of pixels to cut out of each echelle order (from both sides)
    :param object_keyword: The keyword of the object name in the fits header
    :param modeldir: The directory storing the models
    :param debug:
    """

    # Define some constants to use
    lightspeed = constants.c.cgs.value * units.cm.to(units.km)

    # Make sure each output file exists:
    logfilenames = {}
    output_directories = {}
    for fname in fileList:
        output_dir = "Sensitivity/"
        outfilebase = fname.split(".fits")[0]
        if "/" in fname:
            dirs = fname.split("/")
            output_dir = ""
            outfilebase = dirs[-1].split(".fits")[0]
            for directory in dirs[:-1]:
                output_dir = output_dir + directory + "/"
            output_dir = output_dir + "Sensitivity/"
        HelperFunctions.ensure_dir(output_dir)
        output_directories[fname] = output_dir

        # Make the summary file
        logfile = open(output_dir + "logfile.dat", "w")
        logfile.write("Sensitivity Analysis:\n*****************************\n\n")
        logfile.write(
            "Filename\t\t\tPrimary Temperature\tSecondary Temperature\tMass (Msun)\tMass Ratio\tVelocity\tPeak Correct?\tSignificance\n")
        logfile.close()
        logfilenames[fname] = output_dir + "logfile.dat"


    # Read in the companion file
    companions = ascii.read(companion_file)[20:]

    # Read in the vsini file
    vsini_data = ascii.read(vsini_file)[vsini_skip:]

    # Now, start loop over the models:
    model_list = StellarModel.GetModelList(metal=[0, ], temperature=range(3000, 6100, 100), model_directory=modeldir)
    for modelnum, modelfile in enumerate(model_list):
        temp, gravity, metallicity = StellarModel.ClassifyModel(modelfile)
        print("Reading in file %s" % modelfile)
        x, y, c = np.loadtxt(modelfile, usecols=(0, 1, 2), unpack=True)
        print("Processing file...")
        # c = FittingUtilities.Continuum(x, y, fitorder=2, lowreject=1.5, highreject=5)
        n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4  # Index of refraction of air
        model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm) / n, y=10 ** y, cont=10 ** c)
        model = FittingUtilities.RebinData(model, np.linspace(model.x[0], model.x[-1], model.size()))
        model = Broaden.RotBroad(model, vsini_secondary)
        model = Broaden.ReduceResolution2(model, resolution)
        modelfcn = interp(model.x, model.y / model.cont)


        # Now that we have a spline function for the broadened data,
        # begin looping over the files
        for fname in fileList:
            print(fname)
            output_dir = output_directories[fname]
            outfile = open(logfilenames[fname], "a")

            # Read in and process the data like I am about to look for a companion
            orders_original = HelperFunctions.ReadExtensionFits(fname)
            # orders_original = GenericSearch.Process_Data(fname, badregions=badregions, trimsize=trimsize)

            # Find the vsini of the primary star with my spreadsheet
            starname = fits.getheader(fname)[object_keyword]
            found = False
            for data in vsini_data:
                if data[0] == starname:
                    vsini = abs(float(data[vsini_idx]))
                    found = True
            if not found:
                sys.exit("Cannot find %s in the vsini data: %s" % (starname, vsini_file))

            if debug:
                print(starname, vsini)

            # Check for companions in my master spreadsheet
            known_stars = []
            if starname in companions.field(0):
                row = companions[companions.field(0) == starname]
                known_stars.append(row['col1'].item())
                ncompanions = int(row['col4'].item())
                for comp in range(ncompanions):
                    spt = row["col%i" % (6 + 4 * comp)].item()
                    if not "?" in spt and (spt[0] == "O" or spt[0] == "B" or spt[0] == "A" or spt[0] == "F"):
                        sep = row["col%i" % (7 + 4 * comp)].item()
                        if (not "?" in sep) and float(sep) < 4.0:
                            known_stars.append(spt)
            else:
                sys.exit("Star ({:s}) not found in multiplicity library ({:s}!".format(starname, companion_file))

            # Determine the age of the system and properties of the primary and secondary star
            primary_spt = known_stars[0]
            end = search("[0-9]", primary_spt).end()
            primary_temp = MS.Interpolate(MS.Temperature, primary_spt[:end])
            primary_mass = GetMass(primary_spt)
            secondary_spt = MS.GetSpectralType(MS.Temperature, temp)
            secondary_mass = GetMass(secondary_spt)
            massratio = secondary_mass / primary_mass

            for rv in vel_list:
                print("Testing model with rv = ", rv)
                orders = [order.copy() for order in orders_original]  # Make a copy of orders
                model_orders = []
                for ordernum, order in enumerate(orders):
                    # Get the flux ratio
                    scale = GetFluxRatio(known_stars, temp, order.x)
                    if debug:
                        print("Scale factor for order %i is %.3g" % (ordernum, scale.mean()))

                    # Add the model to the data
                    model = (modelfcn(order.x * (1.0 + rv / lightspeed)) - 1.0) * scale
                    order.y += model * order.cont


                    # Smooth data using the vsini of the primary star
                    dx = order.x[1] - order.x[0]
                    npixels = max(21, GenericSmooth.roundodd(vsini / lightspeed * order.x.mean() / dx * smooth_factor))
                    smoothed = GenericSmooth.SmoothData(order,
                                                        windowsize=npixels,
                                                        smoothorder=3,
                                                        lowreject=3,
                                                        highreject=3,
                                                        expand=10,
                                                        numiters=10,
                                                        normalize=False)
                    order.y /= smoothed.y
                    orders[ordernum] = order


                # Trim the data now
                orders = GenericSearch.Process_Data(orders, badregions=badregions, trimsize=trimsize)
                orderweights = []
                for ordernum, order in enumerate(orders):
                    # log-space the data
                    start = np.log(order.x[0])
                    end = np.log(order.x[-1])
                    xgrid = np.logspace(start, end, order.size(), base=np.e)
                    logspacing = np.log(xgrid[1] / xgrid[0])
                    order = FittingUtilities.RebinData(order, xgrid)

                    # Generate a model with the same log-spacing (and no rv shift)
                    dlambda = order.x[order.size() / 2] * 1000 * 1.5 / lightspeed
                    start = np.log(order.x[0] - dlambda)
                    end = np.log(order.x[-1] + dlambda)
                    xgrid = np.exp(np.arange(start, end + logspacing, logspacing))
                    model = DataStructures.xypoint(x=xgrid, cont=np.ones(xgrid.size))
                    model.y = modelfcn(xgrid)

                    # Save model order
                    model_orders.append(model)
                    orders[ordernum] = order
                    scale = GetFluxRatio(known_stars, temp, order.x)
                    orderweights.append(scale.mean())

                # Do the actual cross-correlation
                print("Cross-correlating...")
                corr = Correlate.Correlate(orders,
                                           model_orders,
                                           debug=debug,
                                           outputdir="Sensitivity_Testing/",
                                           addmode=addmode,
                                           prim_teff=primary_temp,
                                           get_weights=True)

                # Check if we found the companion
                idx = np.argmax(corr.y)
                vmax = corr.x[idx]
                fit = FittingUtilities.Continuum(corr.x, corr.y, fitorder=2, lowreject=3, highreject=2.5)
                corr.y -= fit
                goodindices = np.where(np.abs(corr.x - rv) > 100)[0]
                mean = corr.y[goodindices].mean()
                std = corr.y[goodindices].std()
                significance = (corr.y[idx] - mean) / std
                if debug:
                    corrfile = "%s%s_t%i_v%i" % (output_dir, fname.split("/")[-1].split(".fits")[0], temp, rv)
                    print("Outputting CCF to %s" % corrfile)
                    np.savetxt(corrfile, np.transpose((corr.x, corr.y - mean, np.ones(corr.size()) * std)),
                               fmt="%.10g")
                if abs(vmax - rv) <= tolerance:
                    #Found
                    outfile.write("%s\t%i\t\t\t%i\t\t\t\t%.2f\t\t%.4f\t\t%i\t\tyes\t\t%.2f\n" % (
                        fname, primary_temp, temp, secondary_mass, massratio, rv, significance))
                else:
                    #Not found
                    outfile.write("%s\t%i\t\t\t%i\t\t\t\t%.2f\t\t%.4f\t\t%i\t\tno\t\tN/A\n" % (
                        fname, primary_temp, temp, secondary_mass, massratio, rv))
                print("Done with rv ", rv)
            outfile.close()


def get_sec_spt(row):
    """
    Get the secondary spectral type from the information we have
    """
    if pd.notnull(row['Sp2']):
        return row['Sp2']
    elif pd.notnull(row['Sp1']) and pd.notnull(row['mag1']) and pd.notnull(row['mag2']):
        # TODO: Do better than assuming V band!
        band = 'V'
        absmag_prim = MS.GetAbsoluteMagnitude(row['Sp1'], color=band)
        dm = float(row['mag1']) - absmag_prim
        absmag_sec = float(row['mag2']) - dm
        return MS.GetSpectralType_FromAbsMag(absmag_sec, color=band)[0]
    elif pd.notnull(row['Sp1']) and pd.notnull(row['K1']) and pd.notnull(row['K2']):
        mass = MS.Interpolate('mass', row['Sp1'])
        q = float(row['K1']) / float(row['K2'])
        sec_mass = q * mass
        return MS.GetSpectralType('mass', sec_mass)[0]
    else:
        print(row)
        raise ValueError('Must give enough information to figure out the spectral type!')


def split_by_component(df):
    df['prim_comp'] = df.Comp.map(lambda s: s[0])
    df['sec_comp'] = df.Comp.map(lambda s: s[-1])
    comps = pd.concat((df[['prim_comp', 'Sp1']], df[['sec_comp', 'Sp2']]))
    prim = comps.loc[comps.prim_comp.notnull()].rename(columns={'Sp1': 'SpT', 'prim_comp': 'comp'})
    sec = comps.loc[comps.sec_comp.notnull()].rename(columns={'Sp2': 'SpT', 'sec_comp': 'comp'})
    return pd.concat((prim, sec))[['comp', 'SpT']].drop_duplicates(subset='comp')


def return_primary(data):
    retdict = defaultdict(list)
    spt = data.spectype
    retdict['temperature'].append(MS.Interpolate('temperature', spt))
    retdict['radius'].append(MS.Interpolate('radius', spt))
    retdict['mass'].append(MS.Interpolate('mass', spt))
    return retdict


mult_filename = '{}/Dropbox/School/Research/Databases/A_star/SB9andWDS.csv'.format(os.environ['HOME'])
multiples = pd.read_csv(mult_filename)
def get_companions(starname, sep_max=1.5):
    data = StarData.GetData(starname, safe_spt=True)

    # Search for the given star in the database
    match = multiples.loc[multiples.main_id == data.main_id]
    print('{} matches with the same name'.format(len(match)))
    if len(match) < 1:
        return return_primary(data)

    # Now, only keep close companions
    match = match.loc[(match.separation < sep_max) | (match.separation.isnull())]
    print('{} matches that are within {}"'.format(len(match), sep_max))
    if len(match) < 1:
        return return_primary(data)

    # Finally, only keep stars we can figure something out with
    match = match.loc[((match.Sp1.notnull()) & (match.mag1.notnull()) & match.mag2.notnull()) | (
    (match.K1.notnull()) & match.K2.notnull())]
    print('{} matches with sufficient information'.format(len(match)))
    if len(match) < 1:
        return return_primary(data)

    # Get the spectral type for each match
    match['Sp2'] = match.apply(get_sec_spt, axis=1)

    # Only keep the companions that are early type for this
    match = match.loc[(match.Sp2.str.startswith('O')) | (match.Sp2.str.startswith('B'))
                      | (match.Sp2.str.startswith('A')) | (match.Sp2.str.startswith('F'))]
    print('{} matches with early type companions'.format(len(match)))
    if len(match) < 1:
        return return_primary(data)


    # Get the temperature, mass, and radius of the companions
    # Split by the components in the system
    match = match.fillna('AB')
    components = split_by_component(match.copy())

    # Fix spectral type
    components['SpT'] = components['SpT'].map(lambda s: s.replace('m', '5'))
    print(components)
    print(components['SpT'])
    components['companion_mass'] = components['SpT'].map(lambda s: MS.Interpolate('mass', s))
    components['companion_teff'] = components['SpT'].map(lambda s: MS.Interpolate('temperature', s))
    components['companion_radius'] = components['SpT'].map(lambda s: MS.Interpolate('radius', s))

    retdict = {'temperature': list(components['companion_teff']),
               'mass': list(components['companion_mass']),
               'radius': list(components['companion_radius'])}
    return retdict


def Analyze(fileList,
            primary_vsini,
            badregions=[],
            interp_regions=[],
            extensions=True,
            resolution=None,
            trimsize=1,
            vsini_values=(10,),
            Tvalues=range(3000, 6100, 100),
            metal_values=(0.0,),
            logg_values=(4.5,),
            max_vsini=None,
            modeldir=StellarModel.modeldir,
            hdf5_file=StellarModel.HDF5_FILE,
            addmode="ML",
            output_mode='hdf5',
            output_file='Sensitivity.hdf5',
            vel_list=range(-400, 450, 50),
            tolerance=5.0,
            rerun=False,
            debug=False,
            makeplots=False):
    """
    This function runs a sensitivity analysis using the same methodology as slow_companion_search
    :param fileList: The list of fits data files
    :param primary_vsini: A list of the same length as fileList, which contains the vsini for each star (in km/s)
    :param badregions: A list of wavelength regions to ignore in the CCF (contaminated by tellurics, etc)
    :param interp_regions: A list of wavelength regions to interpolate over in the data. Generally, badregions should be on the edge of the orders or contain the whole order, and interp_regions should contain a small wavelength range.
    :param resolution: The detector resolution in lam/dlam. The default is now to use a pre-broadened grid; do not give a value for resolution unless the grid is un-broadened!
    :param trimsize: The number of pixels to cut from both sides of each order. This is because the  order edges are usually pretty noisy.
    :param vsini_values: A list of vsini values (in km/s) to apply to each model spectrum before correlation.
    :param Tvalues: A list of model temperatures (in K) to correlate the data against.
    :param metal_values: A list of [Fe/H] values to correlate the model against
    :param logg_values: A list of log(g) values (in cgs units) to correlate the model against
    :param max_vsini: What is the maximum vsini (in km/s) that we search? If it is given and less than 
                      any of the vsini_values, then the model we correlate against has this vsini. For example,
                      if one of the vsini_values is 150 km/s and the max_vsini is 40 km/s, then a 150 km/s model
                      will be added to the data, but we use a 40 km/s model to correlate against the result.
    :param modeldir: The model directory. This is no longer used by default!
    :param hdf5_file: The path to the hdf5 file containing the pre-broadened model grid.
    :param addmode: The way to add the CCFs for each order. Options are:
         1: 'simple': Do a simple average
         2: 'weighted': Do a weighted average: C = \sum_i{w_i C_i^2}
         3: 'ml': The maximum likelihood estimate. See Zucker 2003, MNRAS, 342, 1291
         4: 'all': Does all of the above.
    :param output_mode: How to output. Valid options are:
         1: text, which is just ascii data with a filename convention.
         2: hdf5, which ouputs a single hdf5 file with all the metadata necessary to classify the output
    :param output_file: The name of the HDF5 file to to save the results to. Only used if output_mode='hdf5'.
    :param vel_list: The list of velocities to add the model to the data with
    :param tolerance: How close the highest CCF peak needs to be to the correct velocity, to count as a detection
    :param rerun: If output_mode=hdf5, check to see if the current parameters have already been checked before running.
    :param debug: Flag to print a bunch of information to screen, and save some intermediate data files
    :param makeplots: A 'higher level' of debug. Will make a plot of the data and model orders for each model.
    """

    model_list = StellarModel.GetModelList(type='hdf5',
                                           hdf5_file=hdf5_file,
                                           temperature=Tvalues,
                                           metal=metal_values,
                                           logg=logg_values)
    modeldict, processed = StellarModel.MakeModelDicts(model_list, type='hdf5', hdf5_file=hdf5_file,
                                                       vsini_values=vsini_values, vac2air=True, logspace=True)

    get_weights = True if addmode.lower() == "weighted" else False
    orderweights = None

    MS = SpectralTypeRelations.MainSequence()

    # Do the cross-correlation
    datadict = defaultdict(list)
    temperature_dict = defaultdict(float)
    vbary_dict = defaultdict(float)
    alpha = 0.0
    for temp in sorted(modeldict.keys()):
        for gravity in sorted(modeldict[temp].keys()):
            for metallicity in sorted(modeldict[temp][gravity].keys()):
                for vsini_sec in vsini_values:
                    if debug:
                        logging.info('T: {}, logg: {}, [Fe/H]: {}, vsini: {}'.format(temp, gravity,
                                                                                     metallicity, vsini_sec))
                    # broaden the model
                    model = modeldict[temp][gravity][metallicity][alpha][vsini_sec].copy()
                    broadened = Broaden.RotBroad(model, vsini_sec * units.km.to(units.cm), linear=True)
                    if resolution is not None:
                        broadened = FittingUtilities.ReduceResolutionFFT(broadened, resolution)
                    if max_vsini is not None and max_vsini < vsini_sec:
                        search_model = Broaden.RotBroad(model, vsini_sec * units.km.to(units.cm), linear=True)
                        if resolution is not None:
                            search_model = FittingUtilities.ReduceResolutionFFT(search_model, resolution)
                    else:
                        search_model = broadened.copy()

                    # Make an interpolator function
                    bb_flux = blackbody_lambda(broadened.x * units.nm, temp)
                    idx = np.where(broadened.x > 700)[0]
                    s = np.median(broadened.y[idx] / bb_flux[idx])
                    broadened.cont = bb_flux * s
                    modelfcn = interp(broadened.x, broadened.y / broadened.cont)

                    for i, (fname, vsini_prim) in enumerate(zip(fileList, primary_vsini)):
                        # Read in data
                        process_data = False if fname in datadict else True
                        if process_data:
                            orders_original = HelperFunctions.ReadExtensionFits(fname)
                            orders_original = GenericSearch.Process_Data(orders_original,
                                                                         badregions=badregions, interp_regions=[],
                                                                         trimsize=trimsize, vsini=None,
                                                                         reject_outliers=False, logspacing=False)

                            datadict[fname] = orders_original
                        else:
                            orders_original = datadict[fname]

                        header = fits.getheader(fname)
                        starname = header['OBJECT']
                        date = header['DATE-OBS'].split('T')[0]

                        components = get_companions(starname)
                        print(components)
                        primary_temp = components['temperature']
                        primary_radius = components['radius']
                        primary_mass = components['mass']
                        secondary_spt = MS.GetSpectralType('temperature', temp)[0]
                        secondary_radius = MS.Interpolate('radius', secondary_spt)
                        secondary_mass = MS.Interpolate('mass', secondary_spt)

                        for rv in vel_list:
                            # Check if these parameters already exist
                            params = {'velocity': rv, 'primary_temps': primary_temp, 'secondary_temp': temp,
                                      'object': starname, 'date': date,
                                      'primary_vsini': vsini_prim, 'secondary_vsini': vsini_sec,
                                      'primary_masses': primary_mass, 'secondary_mass': secondary_mass,
                                      'logg': gravity, '[Fe/H]': metallicity, 'addmode': addmode}
                            if output_mode == 'hdf5' and not rerun and check_existence(output_file, params):
                                continue

                            # Make a copy of the data orders
                            orders = [order.copy() for order in orders_original]

                            for ordernum, order in enumerate(orders):
                                # Get the flux ratio
                                prim_flux = 0.0
                                for ptemp, pR in zip(primary_temp, primary_radius):
                                    prim_flux += blackbody_lambda(order.x * units.nm, ptemp).cgs.value * pR
                                sec_flux = blackbody_lambda(order.x * units.nm, temp).cgs.value * secondary_radius
                                scale = sec_flux / prim_flux

                                # Add the model to the data
                                model_segment = (modelfcn(order.x * (1.0 - rv / lightspeed)) - 1.0) * scale
                                order.y += model_segment * order.cont

                                orders[ordernum] = order

                            # Process the data and model
                            orders = GenericSearch.Process_Data(orders,
                                                                badregions=[], interp_regions=interp_regions,
                                                                extensions=extensions, trimsize=0,
                                                                vsini=vsini_prim, logspacing=True,
                                                                reject_outliers=True)
                            model_orders = GenericSearch.process_model(search_model.copy(), orders,
                                                                       vsini_model=vsini_sec, vsini_primary=vsini_prim,
                                                                       debug=debug, logspace=False)

                            # Do the correlation
                            corr = Correlate.Correlate(orders, model_orders, addmode=addmode, outputdir='Sensitivity/',
                                                       get_weights=get_weights, prim_teff=max(primary_temp),
                                                       debug=debug)
                            if debug:
                                corr, ccf_orders = corr

                            # Determine if we found the companion, and output
                            check_detection(corr, params, mode='hdf5', tol=tolerance, hdf5_file=output_file)





                    # Delete the model. We don't need it anymore and it just takes up ram.
                    modeldict[temp][gravity][metallicity][alpha][vsini_sec] = []

    return


def check_detection(corr, params, mode='text', tol=5, update=True, hdf5_file='Sensitivity.hdf5', backup=True):
    """
    Check if we detected the companion, and output to a summary file.
    :param: corr: The DataStructures object holding the cross-correlation function
    :param params: A dictionary describing the metadata to include
    :param mode: See docstring for slow_companion_search, param output_mode
    :keyword tol: Tolerance (in km/s) to count a peak as the 'correct' one.\
    :keyword backup: Should we write to 2 files instead of just one? That way things we are safe from crashes
                     corrupting the HDF5 file (but it takes twice the disk space...)
    """
    # Loop through the add-modes if addmode=all
    if params['addmode'].lower() == 'all':
        for am in corr.keys():
            p = params
            p['addmode'] = am
            check_detection(corr[am], p, mode=mode, tol=tol, update=update, hdf5_file=hdf5_file)
        return

    idx = np.argmax(corr.y)
    vmax = corr.x[idx]
    detected = True if abs(vmax - params['velocity']) < tol else False

    # Find the significance
    if detected:
        logging.info('Companion detected!')
        fit = FittingUtilities.Continuum(corr.x, corr.y, fitorder=2, lowreject=3, highreject=2.5)
        corr.y -= fit
        goodindices = np.where(np.abs(corr.x - params['velocity']) > 100)[0]
        mean = corr.y[goodindices].mean()
        std = corr.y[goodindices].std()
        significance = (corr.y[idx] - mean) / std
    else:
        logging.info('Companion not detected...')
        significance = np.nan

    # Output
    if mode.lower() == 'text':
        outfile = open('Sensitivity.txt', 'a')
        if detected:
            outfile.write("{0:s}\t{1:s}\t{2:d}\t\t\t{3:d}\t\t\t\t{4:.2f}\t\t"
                          "{5:.4f}\t\t{6:d}\t\tyes\t\t{7:.2f}\n".format(params['object'], params['date'],
                                                                        max(params['primary_temps']),
                                                                        params['secondary_temp'],
                                                                        params['secondary_mass'],
                                                                        params['secondary_mass'] / sum(
                                                                            params['primary_masses']),
                                                                        params['velocity'],
                                                                        significance))
        else:
            outfile.write("{0:s}\t{1:s}\t{2:d}\t\t\t{3:d}\t\t\t\t{4:.2f}\t\t"
                          "{5:.4f}\t\t{6:d}\t\tno\t\t{7:.2f}\n".format(params['object'], params['date'],
                                                                       max(params['primary_temps']),
                                                                       params['secondary_temp'],
                                                                       params['secondary_mass'],
                                                                       params['secondary_mass'] / sum(
                                                                           params['primary_masses']),
                                                                       params['velocity'],
                                                                       significance))

    elif mode.lower() == 'hdf5':
        # Get the hdf5 file
        print('Saving CCF to {}'.format(hdf5_file))
        f = h5py.File(hdf5_file, 'a')

        # Star name and date
        star = params['object']
        date = params['date']

        # Get or create star in file
        print(star, date)
        print(params)
        if star in f.keys():
            s = f[star]
        else:
            star_data = StarData.GetData(star)
            s = f.create_group(star)
            s.attrs['vsini'] = params['primary_vsini']
            s.attrs['RA'] = star_data.ra
            s.attrs['DEC'] = star_data.dec
            s.attrs['SpT'] = star_data.spectype
            s.attrs['n_companions'] = len(params['primary_temps'])
            for i, (pT, pM) in enumerate(zip(params['primary_temps'], params['primary_masses'])):
                s.attrs['comp{}_Teff'.format(i + 1)] = pT
                s.attrs['comp{}_Mass'.format(i + 1)] = pM

        # Get or create date in star
        d = s[date] if date in s.keys() else s.create_group(date)

        # Get or create a group for the secondary star temperature
        Tsec = str(int(params['secondary_temp']))
        if Tsec in d.keys():
            T = d[Tsec]
        else:
            T = d.create_group(Tsec)
            T.attrs['mass'] = params['secondary_mass']


        # Add a new dataset. The name doesn't matter
        current_datasets = T.keys()
        #attr_pars = ['vbary'] if 'vbary' in params else []
        #attr_pars.extend(['vsini', 'T', 'logg', '[Fe/H]', 'addmode', 'fname'])
        attr_pars = ['vsini', 'logg', '[Fe/H]', 'rv', 'addmode']
        params['vsini'] = params['secondary_vsini']
        params['rv'] = params['velocity']
        for ds_name, ds_test in T.iteritems():
            if all([HelperFunctions.is_close(ds_test.attrs[a], params[a]) for a in attr_pars]):
                if update:
                    ds = ds_test
                    new_data = np.array((corr.x, corr.y))
                    try:
                        ds.resize(new_data.shape)
                    except TypeError:
                        # Hope for the best...
                        pass
                    ds[:] = np.array((corr.x, corr.y))
                    idx = np.argmax(corr.y)
                    #ds.attrs['vel_max'] = corr.x[idx]
                    #ds.attrs['ccf_max'] = corr.y[idx]
                    ds.attrs['vsini'] = params['secondary_vsini']
                    ds.attrs['logg'] = params['logg']
                    ds.attrs['[Fe/H]'] = params['[Fe/H]']
                    ds.attrs['rv'] = params['velocity']
                    ds.attrs['addmode'] = params['addmode']
                    ds.attrs['detected'] = detected
                    ds.attrs['significance'] = significance
                    f.flush()
                    f.close()
                return

        # If we get here, the dataset does not yet exist so create it.
        ds_num = len(current_datasets) + 1
        ds = T.create_dataset('ds{}'.format(ds_num), data=np.array((corr.x, corr.y)), maxshape=(2, None))

        # Add attributes to the dataset
        ds.attrs['vsini'] = params['secondary_vsini']
        ds.attrs['logg'] = params['logg']
        ds.attrs['[Fe/H]'] = params['[Fe/H]']
        ds.attrs['rv'] = params['velocity']
        ds.attrs['addmode'] = params['addmode']
        ds.attrs['detected'] = detected
        ds.attrs['significance'] = significance

        f.flush()
        f.close()

        # Write the second file if backup = True
        if backup:
            check_detection(corr, params, mode=mode, tol=tol, update=update,
                            hdf5_file='{}_autobackup.hdf5'.format(hdf5_file.split('.h')[0]), backup=False)

    else:
        raise ValueError('output mode ({}) not supported!'.format(mode))


def check_existence(hdf5_file, params):
    # Check if an entry already exists for the given parameters in the given hdf5 file.
    if not os.path.exists(hdf5_file):
        return False

    starname = params['object']
    date = params['date']
    teff = str(int(params['secondary_temp']))
    with h5py.File(hdf5_file, 'r') as f:
        if (starname in f.keys() and date in f[starname].keys() and teff in f[starname][date].keys()):
            logging.debug('Checking datasets...')
            retval = False
            vsini = params['secondary_vsini']
            logg = params['logg']
            feh = params['[Fe/H]']
            addmode = params['addmode']
            rv = params['velocity']
            for _, ds in f[starname][date][teff].iteritems():
                logging.debug(ds.name)
                attrs = ds.attrs
                for a in attrs:
                    logging.debug('{}: {}'.format(a, attrs[a]))
                if all([key in attrs for key in ['vsini', 'logg', '[Fe/H]', 'rv', 'addmode'] ]):
                        
                    if (np.isclose(vsini, attrs['vsini']) and np.isclose(logg, attrs['logg']) and
                            np.isclose(feh, attrs['[Fe/H]']) and np.isclose(rv, attrs['rv']) and
                                (addmode == attrs['addmode'] or addmode == 'all')):
                        logging.debug('Match found!')
                        retval = True
                else:
                    for a in attrs:
                        logging.info('{}: {}'.format(a, attrs[a]))
                    raise KeyError('Keys not all found in ds {}'.format(ds.name))
            return retval

        else:
            return False


class HDF5_Interface(object):
    def __init__(self, filename):
        self.hdf5 = h5py.File(filename, 'r')


    def list_stars(self, print2screen=False):
        """
        List the stars available in the HDF5 file, and the dates available for each
        :return: A list of the stars
        """
        if print2screen:
            for star in sorted(self.hdf5.keys()):
                print(star)
                for date in sorted(self.hdf5[star].keys()):
                    print('\t{}'.format(date))
        return sorted(self.hdf5.keys())


    def list_dates(self, star, print2screen=False):
        """
        List the dates available for the given star
        :param star: The name of the star
        :return: A list of dates the star was observed
        """
        if print2screen:
            for date in sorted(self.hdf5[star].keys()):
                print(date)
        return sorted(self.hdf5[star].keys())


    def to_df(self, starname=None, date=None):
        """
        This reads in all the datasets for the given star and date.
        If star/date is not given, it reads in all the datesets in the hdf5 file.
        :param starname: the name of the star. Must be in self.hdf5
        :param date: The date to search. Must be in self.hdf5[star]
        :return: a pandas DataFrame with the columns:
                  - star (primary)
                  - primary masses (a list of masses for the primary and any known early-type companions)
                  - primary temps (a list of temperatures for the primary and any known early-type companions)
                  - date
                  - temperature
                  - secondary mass
                  - log(g)
                  - [Fe/H]
                  - vsini (of the secondary)
                  - addmode
                  - rv
                  - significance
        """
        df_list = []
        if starname is None:
            starnames = self.list_stars()
            for starname in starnames:
                dates = self.list_dates(starname)
                for date in dates:
                    df_list.append(self.to_df(starname=starname, date=date))
        elif starname is not None and date is None:
            # Get every date for the requested star
            dates = self.list_dates(starname)
            for date in dates:
                df_list.append(self.to_df(starname=starname, date=date))
        else:
            # Get the primary information
            logging.info(starname)
            g = self.hdf5[starname]
            for a in g.attrs:
                logging.debug(a, g.attrs[a])
            try:
                prim_spt = self.hdf5[starname].attrs['SpT']
                prim_vsini = self.hdf5[starname].attrs['vsini']
                n_comps = self.hdf5[starname].attrs['n_companions']
                pmass = []
                ptemp = []
                prad = []
                for n in range(n_comps):
                    pmass.append(self.hdf5[starname].attrs['comp{}_Mass'.format(n + 1)])
                    ptemp.append(self.hdf5[starname].attrs['comp{}_Teff'.format(n + 1)])
                    spt = MS.GetSpectralType('temperature', ptemp[-1], prec=0.01)
                    prad.append(MS.Interpolate('radius', spt)[0])

                # Get the detection information
                temperatures = self.hdf5[starname][date].keys()
                for T in temperatures:
                    datasets = self.hdf5[starname][date][T].items()
                    logg = [ds[1].attrs['logg'] for ds in datasets]
                    metal = [ds[1].attrs['[Fe/H]'] for ds in datasets]
                    vsini = [ds[1].attrs['vsini'] for ds in datasets]
                    addmode = [ds[1].attrs['addmode'] for ds in datasets]
                    rv = [ds[1].attrs['rv'] for ds in datasets]
                    significance = [ds[1].attrs['significance'] for ds in datasets]
                    temp = [T] * len(logg)
                    try:
                        mass = [self.hdf5[starname][date][T].attrs['mass']] * len(logg)
                    except:
                        sec_spt = MS.GetSpectralType('temperature', float(T), prec=0.01)
                        mass = [MS.Interpolate('mass', sec_spt)] * len(logg)
                    df = pd.DataFrame(data={'star': [starname] * len(logg), 'primary masses': [pmass] * len(logg),
                                            'primary temps': [ptemp] * len(logg), 'primary radii': [prad] * len(logg),
                                            'primary SpT': [prim_spt] * len(logg),
                                            'primary vsini': [prim_vsini] * len(logg), 'date': [date] * len(logg),
                                            'addmode': addmode, 'mass': mass,
                                            'temperature': temp, 'logg': logg, '[Fe/H]': metal,
                                            'vsini': vsini, 'significance': significance, 'rv': rv})
                    df_list.append(df)

            except KeyError:
                logging.warn('Something weird happened with {}. Check the HDF5 file manually!'.format(starname))
                df = pd.DataFrame(data={'star': [], 'primary masses': [],
                                        'primary temps': [], 'primary radii': [],
                                        'primary SpT': [],
                                        'primary vsini': [], 'date': [],
                                        'addmode': [], 'mass': [],
                                        'temperature': [], 'logg': [], '[Fe/H]': [],
                                        'vsini': [], 'significance': [], 'rv': []})
                df_list = [df]

        # Convert things to float, when possible
        df = pd.concat(df_list, ignore_index=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df
        
        

"""
=================================================
    Scripts for analyzing the HDF5 output
=================================================
"""


def get_luminosity_ratio(row):
    """
    Given a row in the overall dataframe, figure out the luminosity ratio. This is meant to be called via df.map
    :param row:
    :return:
    """
    # Get luminosity ratio
    lum_prim = 0
    for T, R in zip(row['primary temps'], row['primary radii']):
        lum_prim += T ** 4 * R ** 2
    T_sec = float(row['temperature'])
    s_spt = MS.GetSpectralType('temperature', T_sec)
    lum_sec = T_sec ** 4 * MS.Interpolate('radius', s_spt)[0] ** 2

    return lum_prim / lum_sec


def get_contrast(row, band='V'):
    """
    Given a row in the overall dataframe, work out the contrast ratio in the requested magnitude filter
    :param row:
    :param band: The Johnson filter to get the contrast ratio in
    :return:
    """
    # pri_spts = [MS.GetSpectralType(MS.Temperature, T, prec=1e-3) for T in row['primary temps']]
    #pri_spts = [MS.GetSpectralType('temperature', T, prec=1e-3) for T in row['primary temps']]
    #pri_mags = [MS.GetAbsoluteMagnitude(s, color=band) for s in pri_spts]
    pri_spts = MS.GetSpectralType('temperature', row['primary temps'], prec=1e-3)
    pri_mags = MS.GetAbsoluteMagnitude(pri_spts, color=band)
    pri_total_mag = HelperFunctions.add_magnitudes(*pri_mags)

    Tsec = float(row['temperature'])
    # sec_mag = MS.GetAbsoluteMagnitude(MS.GetSpectralType(MS.Temperature, Tsec, prec=1e-3), color=band)
    sec_mag = MS.GetAbsoluteMagnitude(MS.GetSpectralType('temperature', Tsec, prec=1e-3), color=band)

    return float(sec_mag - pri_total_mag)


def read_hdf5(hdf5_file):
    """
    Reads the hdf5 file into a dataframe. Assumes a very specific format!
    :param hdf5_file: the full path to the hdf5 file.
    :return: a pandas dataframe containing summary information
    """
    logging.info('Reading HDF5 file {}'.format(hdf5_file))
    hdf5_int = HDF5_Interface(hdf5_file)
    df = hdf5_int.to_df()

    # Get the luminosity ratio
    logging.info('Estimating the luminosity ratio for each trial')
    df['lum_ratio'] = df.apply(get_luminosity_ratio, axis=1)
    df['logL'] = np.log10(df.lum_ratio)

    # Get the contrast. Split by group and then merge to limit the amount of calculation needed
    logging.info('Estimating the V-band contrast ratio for each trial')
    keys = [u'primary temps', u'temperature']
    test_vsini = df.vsini.unique()[0]
    temp = df.loc[(df.rv == 0) & (df.vsini == test_vsini)].drop_duplicates(subset=['star', 'temperature'])
    # temp = df.groupby(('star')).apply(lambda df: df.loc[(df.rv == 0) & (df.vsini == test_vsini)][keys]).reset_index()
    # logging.debug(temp[['star', 'primary_temps', 'temperature']])
    temp['contrast'] = temp.apply(lambda r: get_contrast(r, band='V'), axis=1)
    df = pd.merge(df, temp[['star', 'temperature', 'contrast']], on=['star', 'temperature'], how='left')

    return df


def heatmap(df, ax=None, make_cbar=True, make_labels=True, **plot_kws):
    """
    Make a heatmap of the pandas dataframe using the first three columns
    """
    xcol, ycol, color_col = df.columns[:3]
    x_range = (df[xcol].min(), df[xcol].max())
    y_range = (df[ycol].min(), df[ycol].max())
    logging.debug(df[[xcol, ycol, color_col]])
    aspect_ratio = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0])
    plot_extents = np.hstack((x_range, y_range[::-1]))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        aspect_ratio = 'auto'
    im = ax.matshow(df.pivot(ycol, xcol, color_col), aspect=aspect_ratio, extent=plot_extents, **plot_kws)
    ax.xaxis.set_ticks_position('bottom')
    if make_cbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(color_col)

    if make_labels:
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

    return im


def summarize_sensitivity(sens_df):
    """
    Summarize the sensitivity analysis by finding the detection rate and average significance
    as a function of teff and vsini

    :param sens_df:   The DataFrame generated by read_hdf5 (and probably available in Sensitivity_Dataframe.csv)
    :return: A pandas dataframe with the summary
    """
    cols = ['star', 'date', '[Fe/H]', 'logg', 'addmode', 'temperature', 'vsini']
    detrate = sens_df.groupby(cols).apply(lambda d: d.significance.notnull().sum() / float(len(d)))
    detrate = detrate.reset_index().rename(columns={0: 'detrate'})
    significance = sens_df.groupby(cols).apply(lambda d: np.nanmean(d.significance))
    significance = significance.reset_index().rename(columns={0: 'significance'})
    detrate['significance'] = significance['significance']
    return detrate


def analyze_sensitivity(hdf5_file='Sensitivity.hdf5', interactive=True, update=True, combine=False, **heatmap_kws):
    """
    This uses the output of a previous run of check_sensitivity, and makes plots
    :keyword interactive: If True, the user will pick which stars to plot
    :keyword update: If True, always update the Sensitivity_Dataframe.csv file.
                     Otherwise, try to load that file instead of reading the hdf5 file
    :keyword combine: If True, combine the sensitivity matrix for all stars to get an average sensitivity
    :return:
    """
    if not update and os.path.isfile('Sensitivity_Dataframe.csv'):
        df = pd.read_csv('Sensitivity_Dataframe.csv')
    else:
        if hdf5_file.endswith('hdf5'):
            df = read_hdf5(hdf5_file)

            # Save the dataframe for later use
            df.to_csv('Sensitivity_Dataframe.csv', index=False)
        elif hdf5_file.endswith('csv'):
            # Treat the input as a csv file
            df = pd.read_csv(hdf5_file)

    # Group by a bunch of keys that probably don't change, but could
    groups = df.groupby(('star', 'date', '[Fe/H]', 'logg', 'addmode', 'primary SpT'))

    # Have the user choose keys
    if interactive:
        for i, key in enumerate(groups.groups.keys()):
            print('[{}]: {}'.format(i + 1, key))
        inp = raw_input('Enter the numbers of the keys you want to plot (, or - delimited): ')
        chosen = parse_input(inp)
        keys = [k for i, k in enumerate(groups.groups.keys()) if i + 1 in chosen]
    else:
        keys = groups.groups.keys()

    # Compile dataframes for each star
    dataframes = defaultdict(lambda: defaultdict(pd.DataFrame))
    for key in keys:
        logging.info(key)
        g = groups.get_group(key)
        detrate = g.groupby(('temperature', 'vsini', 'logL', 'contrast')).apply(
            lambda df: float(sum(df.significance.notnull())) / float(len(df)))
        significance = g.groupby(('temperature', 'vsini', 'logL', 'contrast')).apply(
            lambda df: np.nanmean(df.significance))
        dataframes['detrate'][key] = detrate.reset_index().rename(columns={0: 'detection rate'})
        dataframes['significance'][key] = significance.reset_index().rename(columns={0: 'significance'})

    # Make heatmap plots for each key.
    HelperFunctions.ensure_dir('Figures/')
    for i, key in enumerate(keys):
        star = key[0]
        date = key[1]
        addmode = key[4]
        spt = key[5]
        logging.info('Making figures for {} observed on {} with addmode {}'.format(star, date, addmode))
        plt.figure(i * 3 + 1)
        if len(dataframes['detrate'][key]) == 0:
            dataframes['detrate'].pop(key)
            dataframes['significance'].pop(key)
            continue

        #sns.heatmap(dataframes['detrate'][key].pivot('temperature', 'vsini', 'detection rate'))
        heatmap(dataframes['detrate'][key][['vsini', 'temperature', 'detection rate']], **heatmap_kws)
    
        plt.title('Detection Rate for {} ({}) on {}'.format(star, spt, date))
        plt.savefig('Figures/T_vsini_Detrate_{}.{}.addmode-{}.pdf'.format(star.replace(' ', '_'), date, addmode))

        plt.figure(i * 3 + 2)
        #sns.heatmap(dataframes['significance'][key].pivot('temperature', 'vsini', 'significance'),
        #            robust=True)
        heatmap(dataframes['significance'][key][['vsini', 'temperature', 'significance']], **heatmap_kws)
        plt.title('Detection Significance for {} ({}) on {}'.format(star, spt, date))
        plt.savefig('Figures/T_vsini_Significance_{}.{}.addmode-{}.pdf'.format(star.replace(' ', '_'), date, addmode))

        plt.figure(i * 3 + 3)
        #p = dataframes['detrate'][key].pivot('vsini', 'contrast', 'detection rate')
        #ylabels = [round(float(L), 2) for L in p.index]
        #sns.heatmap(p, yticklabels=ylabels)
        heatmap(dataframes['detrate'][key][['vsini', 'contrast', 'detection rate']], **heatmap_kws)
        plt.title('Detection Rate for {} ({}) on {}'.format(star, spt, date))
        plt.savefig('Figures/contrast_vsini_Detrate_{}.{}.addmode-{}.pdf'.format(star.replace(' ', '_'), date, addmode))

        if not interactive:
            plt.close('all')

    if interactive:
        plt.show()
    
    return dataframes


def marginalize_sensitivity(infilename='Sensitivity_Dataframe.csv'):
    """
    Take the csv output of analyze_sensitivity above, and make 1d plots of temperature vs detection rate
    :param infilename: The name of the csv file to read in. This CAN be the original HDF5 file, but why repeat work?
    :return: A figure and axes instance containing the sensitivity plot
    """
    # Set seaborn settings
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_style('ticks')
    plot_styles = ['-', '--', ':', '-.']
    ps_cycler = itertools.cycle(plot_styles)

    # marginalize over vsini to get 1d temperature vs. detection rate plots
    HelperFunctions.ensure_dir('Marginalized_Sensitivity')
    df = EstimateDetection.read_detection_rate('Sensitivity_Dataframe.csv')
    detrate = df['detrate']
    fig, ax = plt.subplots()
    for key in detrate.keys():
        if len(detrate[key]) == 0:
            continue

        # Get ages from either the measured ages in David & Hillenbrand 2015, or from main sequence ages
        starname = key[0].replace(' ', '')
        ages = EstimateDetection.get_ages(starname.replace(' ', ''), N_age=300)

        # marginalize over vsini
        marg_df = EstimateDetection.marginalize_vsini(detrate[key], age=ages)

        # Plot
        ls = ps_cycler.next()
        ax.plot(marg_df.temperature, marg_df['detection rate'], ls, label='{} ({})'.format(key[0], key[-1]))

        # Save the marginalized dataframe
        df_outname = 'Marginalized_Sensitivity/{}_{}_{}.csv'.format(starname, key[1], key[2])
        logging.info('Saving marginalized dataframe to {}'.format(df_outname))
        marg_df.to_csv(df_outname, index=False)

    # Add a spectral type axis on top
    top = add_top_axis(ax)

    # Axis labels
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Detection Rate')
    top.set_xlabel('Spectral Type')

    # Add a legend
    leg = ax.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.45)

    # Return
    return fig, ax



def add_top_axis(axis, spt_values=('M5', 'M0', 'K5', 'K0', 'G5', 'G0')):
    # get the full range of the axis.
    xlim = axis.get_xlim()

    # Find the temperatures at each spectral type
    MS = SpectralTypeRelations.MainSequence()
    MT = Mamajek_Table.MamajekTable()
    sptnums = np.array([MS.SpT_To_Number(spt) for spt in spt_values])
    sptnum2teff = MT.get_interpolator('SpTNum', 'Teff')
    temp_values = sptnum2teff(sptnums)

    # make the axis
    top = axis.twiny()
    top.set_xticks(temp_values)
    top.set_xlim(xlim)
    top.set_xticklabels(spt_values)
    return top


def parse_input(inp, sort_output=True, ensure_unique=True):
    """
    Parse the user input to get a list of integers
    :param inp: Can be in the form 'a-b', 'a,b,c', 'a-b,c-d', etc.
                '-' means an inclusive list of every number between a and b
                ',' means the numbers a and b
    :return: A list of integers
    """
    sublists = inp.split(',')
    final_list = []
    for l in sublists:
        if '-' in l:
            first, last = l.split('-')
            for i in range(int(first), int(last) + 1):
                final_list.append(i)
        else:
            final_list.append(int(l))

    if ensure_unique:
        final_list = pd.unique(final_list)
    if sort_output:
        final_list = sorted(final_list)
    return final_list


def plot_expected(orders, prim_spt, Tsec, instrument, vsini=None, rv=0.0, twoaxes=False):
    """
    Plot the orders, with a model spectrum added at appropriate flux ratio
    :param orders: A list of Datastructures.xypoint instances
    :param prim_spt: The primary star spectral type
    :param Tsec: The secondary temperature
    :param instrument: The name of the instrument the observation came from
    :param vsini: the vsini of the companion
    :param rv: the rv shift of the companion
    :return:
    """

    sns.set_context('paper', font_scale=2.0)
    sns.set_style('white')
    sns.set_style('ticks')

    # First, get the model
    dir_prefix = '/media/ExtraSpace'
    if not os.path.exists(dir_prefix):
        dir_prefix = '/Volumes/DATADRIVE'
    inst2hdf5 = {'TS23': '{}/PhoenixGrid/TS23_Grid.hdf5'.format(dir_prefix),
                 'HRS': '{}/PhoenixGrid/HRS_Grid.hdf5'.format(dir_prefix),
                 'CHIRON': '{}/PhoenixGrid/CHIRON_Grid.hdf5'.format(dir_prefix),
                 'IGRINS': '{}/PhoenixGrid/IGRINS_Grid.hdf5'.format(dir_prefix)}
    hdf5_int = StellarModel.HDF5Interface(inst2hdf5[instrument])
    wl = hdf5_int.wl
    pars = {'temp': Tsec, 'logg': 4.5, 'Z': 0.0, 'alpha': 0.0}
    fl = hdf5_int.load_flux(pars)

    # Broaden, if requested
    if vsini is not None:
        m = DataStructures.xypoint(x=wl, y=fl)
        m = Broaden.RotBroad(m, vsini * units.km.to(units.cm))
        wl, fl = m.x, m.y

    # get model continuum
    c = FittingUtilities.Continuum(wl, fl, fitorder=5, lowreject=2, highreject=10)

    # Interpolate the model
    x = wl * units.angstrom
    plt.plot(wl, fl)
    plt.plot(wl, c)
    plt.show()
    modelfcn = interp(x.to(units.nm), fl / c)

    # Get the wavelength-specific flux ratio between the primary and secondary star
    MS = SpectralTypeRelations.MainSequence()
    Tprim = MS.Interpolate('temperature', prim_spt)
    Rprim = MS.Interpolate('radius', prim_spt)
    sec_spt = MS.GetSpectralType('temperature', Tsec, prec=1e-3)
    Rsec = MS.Interpolate('radius', sec_spt)
    flux_ratio = blackbody_lambda(x, Tprim) / blackbody_lambda(x, Tsec) * (Rprim / Rsec) ** 2
    fluxratio_fcn = interp(x.to(units.nm), 1.0 / flux_ratio)

    # Loop over the orders:
    if twoaxes:
        fig, axes = plt.subplots(2, 1, sharex=True)
        top, bottom = axes
        for order in orders:
            order.cont = FittingUtilities.Continuum(order.x, order.y, fitorder=3, lowreject=2, highreject=5)
            top.plot(order.x, order.y, 'k-', alpha=0.4)
            top.plot(order.x, order.cont, 'r--')

            total = order.copy()

            xarr = total.x * (1 + rv / constants.c.to(units.km / units.s).value)
            model = (modelfcn(xarr) - 1.0) * fluxratio_fcn(xarr)
            total.y += total.cont * model
            top.plot(total.x, total.y, 'g-', alpha=0.4)

            bottom.plot(total.x, total.y - order.y, 'k-', alpha=0.4)

        return fig, [top, bottom], orders

    fig, ax = plt.subplots(1, 1)
    for order in orders:
        order.cont = FittingUtilities.Continuum(order.x, order.y, fitorder=3, lowreject=2, highreject=5)
        ax.plot(order.x, order.y, 'k-', alpha=0.4)

        total = order.copy()

        xarr = total.x * (1 + rv / constants.c.to(units.km / units.s).value)
        model = (modelfcn(xarr) - 1.0) * fluxratio_fcn(xarr)
        total.y += total.cont * model
        ax.plot(total.x, total.y, 'g-', alpha=0.4)

    # Label
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Flux (arbitrary units)')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([9e9], [9e9], 'k-', label='Actual data')
    ax.plot([9e9], [9e9], 'g-', label='Expected data')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    leg = ax.legend(loc='best', fancybox=True)

    return fig, ax, orders

