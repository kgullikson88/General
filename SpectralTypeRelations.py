from collections import defaultdict
import warnings
import sys

from scipy.interpolate import UnivariateSpline, griddata

import DataStructures
import pandas
import os
import re
import pickle

# Provides relations temperature, luminosity, radius, and mass for varius spectral types
#Data comes from Carroll and Ostlie book, or interpolated from it
#ALL RELATIONS ARE FOR MAIN SEQUENCE ONLY!

"""
  Usage:
         Make instance of class (currently only MainSequence class available
         call instance.Interpolate(instance.dict, SpT) where dict is the name of the dictionary you want to interpolate (Temperature, Radius, or Mass) and SpT is the spectral type of what you wish to interpolate to.
"""

def fill_dict(row, d, key, makefloat=True):
    val = row[key].strip()
    if makefloat:
        if val != '':
            d[row['SpT'].strip()[:-1]] = float(val)
    else:
        d[row['SpT'].strip()[:-1]] = val


class FitVals():
    def __init__(self, coeffs, xmean, xscale, logscale=False, intercept=0.0):
        self.coeffs = coeffs
        self.order = len(coeffs) - 1.0
        self.xmean = xmean
        self.xscale = xscale
        self.log = logscale
        self.intercept = intercept


class FunctionFits():
    def __init__(self, MS=None):
        self.MS = MainSequence() if MS is None else MS
        # Fit to the spectral type lookup table in Pecaut & Mamajek 2013
        self.sptnum_to_teff = FitVals(coeffs=np.array([ 0.02122169, -0.03002895, -0.12733951,  0.15767385,
                                                        0.27973801, -0.28587626, -0.23513699,  0.09748218,
                                                        0.11926096, -0.13911846, -0.02927537]),
                                      xmean=36.515384615384619,
                                      xscale=18.170195640536857,
                                      logscale=True,
                                      intercept=3.8265827919657864)

    def evaluate(self, fv, spt):
        """
        Evaluate the function defined by fv (which is a FitVals instance) for the given spectral type
        """
        if HelperFunctions.IsListlike(spt):
            sptnum = np.array([self.MS.SpT_To_Number(s) for s in spt])
        else:
            sptnum = self.MS.SpT_To_Number(spt)

        # Normalize the sptnum
        x = (sptnum - fv.xmean)/fv.xscale

        # Evaluate
        retval = np.poly1d(fv.coeffs)(x) + fv.intercept
        if fv.log:
            retval = 10**retval
        return retval


class Interpolator():
    def __init__(self, MS=None):
        self.MS = MainSequence() if MS is None else MS

        # Spectral type to temperature converter
        fname = '{}/Dropbox/School/Research/Databases/SpT_Relations/sptnum_to_teff.interp'.format(os.environ['HOME'])
        fileid = open(fname, 'r')
        self.sptnum_to_teff = pickle.load(fileid)
        fileid.close()

    def evaluate(self, interp, spt):
        if HelperFunctions.IsListlike(spt):
            sptnum = np.array([self.MS.SpT_To_Number(s) for s in spt])
        else:
            sptnum = self.MS.SpT_To_Number(spt)

        return interp(sptnum)



class MainSequence:
    def __init__(self):
        self.Temperature = defaultdict(float)
        self.Radius = defaultdict(float)
        self.Mass = defaultdict(float)
        self.Lifetime = defaultdict(float)
        self.BC = defaultdict(float)
        self.BmV = defaultdict(float)  #B-V color
        self.UmV = defaultdict(float)  #U-V color
        self.VmR = defaultdict(float)  #V-R color
        self.VmI = defaultdict(float)  #V-I color
        self.VmJ = defaultdict(float)  #V-J color
        self.VmH = defaultdict(float)  #V-H color
        self.VmK = defaultdict(float)  #V-K color
        self.AbsMag = defaultdict(float)  #Absolute Magnitude in V band

        # Read in the data from Pecaut & Mamajek 2013 for Teff and color indices
        pfilename = "{:s}/Dropbox/School/Research/Databases/SpT_Relations/Pecaut2013.tsv".format(os.environ['HOME'])
        pdata = pandas.read_csv(pfilename, skiprows=55, sep="|")[2:-1]
        pdata.apply(fill_dict, axis=1, args=(self.Temperature, 'Teff', True))
        pdata.apply(fill_dict, axis=1, args=(self.UmV, 'U-B', True))
        pdata.apply(fill_dict, axis=1, args=(self.BmV, 'B-V', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmR, 'V-Rc', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmI, 'V-Ic', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmJ, 'V-J', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmH, 'V-H', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmK, 'V-Ks', True))

        self.Radius['O5'] = 13.4
        self.Radius['O6'] = 12.2
        self.Radius['O7'] = 11.0
        self.Radius['O8'] = 10.0
        self.Radius['B0'] = 6.7
        self.Radius['B1'] = 5.4  #Malkov et al. 2007
        self.Radius['B2'] = 4.9  #Malkov et al. 2007
        self.Radius['B3'] = 3.9  #Malkov et al. 2007
        self.Radius['B4'] = 3.6  #Malkov et al. 2007
        self.Radius['B5'] = 3.3  #Malkov et al. 2007
        self.Radius['B6'] = 3.1  #Malkov et al. 2007
        self.Radius['B7'] = 2.85  #Malkov et al. 2007
        self.Radius['B8'] = 2.57  #Malkov et al. 2007
        self.Radius['B9'] = 2.3
        self.Radius['A0'] = 2.2
        self.Radius['A1'] = 2.1
        self.Radius['A2'] = 2.0
        self.Radius['A5'] = 1.8
        self.Radius['A8'] = 1.5
        self.Radius['F0'] = 1.4
        self.Radius['F2'] = 1.3
        self.Radius['F5'] = 1.2
        self.Radius['F8'] = 1.1
        self.Radius['G0'] = 1.06
        self.Radius['G2'] = 1.03
        self.Radius['G8'] = 0.96
        self.Radius['K0'] = 0.93
        self.Radius['K1'] = 0.92
        self.Radius['K3'] = 0.86
        self.Radius['K4'] = 0.83
        self.Radius['K5'] = 0.80
        self.Radius['K7'] = 0.74
        self.Radius['M0'] = 0.63
        self.Radius['M1'] = 0.56
        self.Radius['M2'] = 0.48
        self.Radius['M3'] = 0.41
        self.Radius['M4'] = 0.35
        self.Radius['M5'] = 0.29
        self.Radius['M6'] = 0.24
        self.Radius['M7'] = 0.20

        self.Mass['O5'] = 60
        self.Mass['O6'] = 37
        self.Mass['O8'] = 23
        self.Mass['B0'] = 17.5
        self.Mass['B1'] = 10.5  #Malkov et al. 2007
        self.Mass['B2'] = 8.9  #Malkov et al. 2007
        self.Mass['B3'] = 6.4  #Malkov et al. 2007
        self.Mass['B4'] = 5.4  #Malkov et al. 2007
        self.Mass['B5'] = 4.5  #Malkov et al. 2007
        self.Mass['B6'] = 4.0  #Malkov et al. 2007
        self.Mass['B7'] = 3.5  #Malkov et al. 2007
        self.Mass['B8'] = 3.2  #Malkov et al. 2007
        self.Mass['A0'] = 2.9
        self.Mass['A5'] = 2.0
        self.Mass['F0'] = 1.6
        self.Mass['F5'] = 1.4
        self.Mass['G0'] = 1.05
        self.Mass['K0'] = 0.79
        self.Mass['K5'] = 0.67
        self.Mass['M0'] = 0.51
        self.Mass['M2'] = 0.40
        self.Mass['M5'] = 0.21

        self.Lifetime['O9.1'] = 8
        self.Lifetime['B1.1'] = 11
        self.Lifetime['B2.5'] = 16
        self.Lifetime['B4.2'] = 26
        self.Lifetime['B5.3'] = 43
        self.Lifetime['B6.7'] = 94
        self.Lifetime['B7.7'] = 165
        self.Lifetime['B9.7'] = 350
        self.Lifetime['A1.6'] = 580
        self.Lifetime['A5'] = 1100
        self.Lifetime['A8.4'] = 1800
        self.Lifetime['F2.6'] = 2700

        #From Allen's Astrophysical Quantities and Binney & Merrifield (marked with 'BM')
        self.AbsMag['O5'] = -5.7
        self.AbsMag['O8'] = -4.9  #BM
        self.AbsMag['O9'] = -4.5
        self.AbsMag['B0'] = -4.0
        self.AbsMag['B2'] = -2.45
        self.AbsMag['B3'] = -1.6  #BM
        self.AbsMag['B5'] = -1.2
        self.AbsMag['B8'] = -0.25
        self.AbsMag['A0'] = 0.65
        self.AbsMag['A2'] = 1.3
        self.AbsMag['A5'] = 1.95
        self.AbsMag['F0'] = 2.7
        self.AbsMag['F2'] = 3.6
        self.AbsMag['F5'] = 3.5
        self.AbsMag['F8'] = 4.0
        self.AbsMag['G0'] = 4.4
        self.AbsMag['G2'] = 4.7
        self.AbsMag['G5'] = 5.1
        self.AbsMag['G8'] = 5.5
        self.AbsMag['K0'] = 5.9
        self.AbsMag['K2'] = 6.4
        self.AbsMag['K5'] = 7.35
        self.AbsMag['M0'] = 8.8
        self.AbsMag['M2'] = 9.9
        self.AbsMag['M5'] = 12.3




    def SpT_To_Number(self, SpT):
        if SpT[1:] == "":
            basenum = 5.0
        else:
            basenum = float(SpT[1:])
        SpectralClass = SpT[0]
        if SpectralClass == "O":
            return basenum
        elif SpectralClass == "B":
            return basenum + 10
        elif SpectralClass == "A":
            return basenum + 20
        elif SpectralClass == "F":
            return basenum + 30
        elif SpectralClass == "G":
            return basenum + 40
        elif SpectralClass == "K":
            return basenum + 50
        elif SpectralClass == "M":
            return basenum + 60
        else:
            print "Something weird happened! Spectral type = ", SpT
            return -1

    def Number_To_SpT(self, number):
        tens_index = 0
        num = float(number)
        while num > 0:
            num -= 10
            tens_index += 1
        tens_index = tens_index - 1
        if num == 0:
            tens_index += 1
            number = 10 * tens_index
        if tens_index == 0:
            spt_class = "O"
        elif tens_index == 1:
            spt_class = "B"
        elif tens_index == 2:
            spt_class = "A"
        elif tens_index == 3:
            spt_class = "F"
        elif tens_index == 4:
            spt_class = "G"
        elif tens_index == 5:
            spt_class = "K"
        elif tens_index == 6:
            spt_class = "M"
        subclass = str(number - 10 * tens_index)
        return spt_class + subclass

    def Interpolate(self, dictionary, SpT):
        #First, we must convert the relations above into a monotonically increasing system
        #Just add ten when we get to each new spectral type
        relation = DataStructures.xypoint(len(dictionary))

        # Strip the spectral type of the luminosity class information
        SpT = re.search('[A-Z]([0-9]\.?[0-9]*)', SpT).group()

        xpoints = []
        ypoints = []
        for key, index in zip(dictionary, range(len(dictionary))):
            #Convert key to a number
            xpoints.append(self.SpT_To_Number(key))
            ypoints.append(dictionary[key])

        sorting_indices = [i[0] for i in sorted(enumerate(xpoints), key=lambda x: x[1])]
        for index in range(len(dictionary)):
            i = sorting_indices[index]
            relation.x[index] = xpoints[i]
            relation.y[index] = ypoints[i]

        RELATION = UnivariateSpline(relation.x, relation.y, s=0)

        spnum = self.SpT_To_Number(SpT)
        if spnum > 0:
            return RELATION(spnum)
        else:
            return np.nan

    def GetAbsoluteMagnitude(self, spt, color='V'):
        Vmag = self.Interpolate(self.AbsMag, spt)
        if color.upper() == "V":
            return Vmag
        else:
            if color.upper() in ['U', 'B']:
                string = "color_diff = self.Interpolate(self.%smV, spt)" % color
                exec (string)
                return color_diff + Vmag
            elif color.upper() in ['R', 'I', 'J', 'H', 'K']:
                string = "color_diff = self.Interpolate(self.Vm%s, spt)" % color
                exec (string)
                return Vmag - color_diff
            else:
                raise ValueError("Color %s not known!" % color)

    def GetSpectralType_FromAbsMag(self, value, color='V'):
        diff = np.inf
        best_spt = 'O9'
        for spt_num in range(10, 70):
            spt = self.Number_To_SpT(spt_num)
            absmag = self.GetAbsoluteMagnitude(spt, color=color)
            dm = abs(absmag - value)
            if dm < diff:
                diff = dm
                best_spt = spt
        return best_spt

    def GetSpectralType(self, dictionary, value, interpolate=False):
        #Returns the spectral type that is closest to the value (within 0.1 subtypes)
        testgrid = np.arange(self.SpT_To_Number("O1"), self.SpT_To_Number("M9"), 0.1)
        besttype = "O1"
        best_difference = 9e9
        for num in testgrid:
            num = round(num, 2)
            spt = self.Number_To_SpT(num)
            difference = np.abs(value - self.Interpolate(dictionary, spt))
            if difference < best_difference:
                best_difference = difference
                besttype = spt
        if not interpolate:
            return besttype
        else:
            bestvalue = self.Interpolate(dictionary, besttype)
            num = self.SpT_To_Number(besttype)
            spt = self.Number_To_SpT(num - 0.1)
            secondvalue = self.Interpolate(dictionary, spt)
            slope = 0.1 / (bestvalue - secondvalue)
            num2 = slope * (bestvalue - value) + num
            return self.Number_To_SpT(num2)


########################################################
########               Pre-Main Sequence         #######
########################################################
import os
import HelperFunctions
import numpy as np

homedir = os.environ["HOME"] + "/"
tracksfile = homedir + "Dropbox/School/Research/Stellar_Evolution/Padova_Tracks.dat"


class PreMainSequence:
    def __init__(self, pms_tracks_file=tracksfile, track_source="Padova", minimum_stage=0, maximum_stage=1):
        #We need an instance of MainSequence to get temperature from spectral type
        self.MS = MainSequence()

        #Now, read in the evolutionary tracks
        if track_source.lower() == "padova":
            self.Tracks = self.ReadPadovaTracks(pms_tracks_file, minimum_stage=minimum_stage,
                                                maximum_stage=maximum_stage)
        elif track_source.lower() == "baraffe":
            self.Tracks = self.ReadBaraffeTracks(pms_tracks_file)


    def ReadPadovaTracks(self, pms_tracks_file, minimum_stage, maximum_stage):
        infile = open(pms_tracks_file)
        lines = infile.readlines()
        infile.close()
        Tracks = defaultdict(lambda: defaultdict(list))
        self.Mass = []
        self.InitialMass = []
        self.Luminosity = []
        self.Gravity = []
        self.Age = []
        self.Temperature = []
        for line in lines:
            if not line.startswith("#"):
                segments = line.split()
                age = float(segments[1])
                m_initial = float(segments[2])  #Initial mass
                mass = float(segments[3])
                Lum = float(segments[4])  #Luminosity
                Teff = float(segments[5])  #Effective temperature
                logg = float(segments[6])  #gravity
                evol_stage = int(segments[-1])

                if (minimum_stage <= evol_stage <= maximum_stage and
                        (len(Tracks[age]["Mass"]) == 0 or Tracks[age]["Mass"][-1] < mass)):
                    Tracks[age]["Initial Mass"].append(m_initial)
                    Tracks[age]["Mass"].append(mass)
                    Tracks[age]["Temperature"].append(Teff)
                    Tracks[age]["Luminosity"].append(Lum)
                    Tracks[age]["Gravity"].append(logg)
                    self.Mass.append(mass)
                    self.InitialMass.append(m_initial)
                    self.Luminosity.append(Lum)
                    self.Gravity.append(logg)
                    self.Temperature.append(Teff)
                    self.Age.append(age)

        return Tracks


    def ReadBaraffeTracks(self, pms_tracks_file):
        infile = open(pms_tracks_file)
        lines = infile.readlines()
        infile.close()
        Tracks = defaultdict(lambda: defaultdict(list))
        for i, line in enumerate(lines):
            if "log t (yr)" in line:
                age = float(line.split()[-1])
                j = i + 4
                while "----" not in lines[j] and lines[j].strip() != "":
                    segments = lines[j].split()
                    mass = float(segments[0])
                    Teff = float(segments[1])
                    logg = float(segments[2])
                    Lum = float(segments[3])
                    Tracks[age]["Mass"].append(mass)
                    Tracks[age]["Temperature"].append(np.log10(Teff))
                    Tracks[age]["Luminosity"].append(Lum)
                    Tracks[age]["Gravity"].append(logg)
                    j += 1

        return Tracks


    def GetEvolution(self, mass, key='Temperature'):
        #Need to find the first and last ages that have the requested mass
        first_age = 9e9
        last_age = 0.0
        Tracks = self.Tracks
        ages = sorted(Tracks.keys())
        ret_ages = []
        ret_value = []
        for age in ages:
            if min(Tracks[age]["Mass"]) < mass and max(Tracks[age]["Mass"]) > mass:
                T = self.GetTemperature(mass, 10 ** age)
                if key == "Temperature":
                    ret_value.append(T)
                else:
                    ret_value.append(self.GetFromTemperature(10 ** age, T, key=key))
                ret_ages.append(10 ** age)
        return ret_ages, ret_value


    def GetFromTemperature(self, age, temperature, key='Mass'):
        # Check that the user gave a valid key
        valid = ["Initial Mass", "Mass", "Luminosity", "Gravity", "Radius"]
        if key not in valid:
            print "Error! 'key' keyword must be one of the following"
            for v in valid:
                print "\t%s" % v
            sys.exit()
        elif key == "Radius":
            #We need to get this from the luminosity and temperature
            lum = self.GetFromTemperature(age, temperature, key="Luminosity")
            return np.sqrt(lum) / (temperature / 5780.0) ** 2


        # Otherwise, interpolate
        Tracks = self.Tracks
        ages = sorted([t for t in Tracks.keys() if 10 ** t >= 0.5 * age and 10 ** t <= 2.0 * age])
        points = []
        values = []
        for t in ages:
            temps = sorted(Tracks[t]['Temperature'])
            val = sorted(Tracks[t][key])
            for T, V in zip(temps, val):
                points.append((t, T))
                values.append(V)
        xi = np.array([[np.log10(age), np.log10(temperature)], ])
        points = np.array(points)
        values = np.array(values)

        val = griddata(points, values, xi, method='linear')
        if np.isnan(val):
            warnings.warn("Requested temperature (%g) at this age (%g) is outside of grid!" % (temperature, age))
            val = griddata(points, values, xi, method='nearest')

        if key == "Luminosity" or key == "Temperature":
            val = 10 ** val
        return float(val)


    def Interpolate(self, SpT, age, key="Mass"):
        Teff = self.MS.Interpolate(self.MS.Temperature, SpT)
        return self.GetFromTemperature(age, Teff, key)


    def GetTemperature(self, mass, age):
        Tracks = self.Tracks
        ages = sorted([t for t in Tracks.keys() if 10 ** t >= 0.5 * age and 10 ** t <= 2.0 * age])
        points = []
        values = []
        for t in ages:
            temps = sorted(Tracks[t]['Temperature'])
            masses = sorted(Tracks[t]['Mass'])
            for T, M in zip(temps, masses):
                points.append((t, M))
                values.append(T)
        xi = np.array([[np.log10(age), mass], ])
        points = np.array(points)
        values = np.array(values)
        val = griddata(points, values, xi, method='linear')
        if np.isnan(val):
            warnings.warn("Requested temperature (%g) at this age (%g) is outside of grid!" % (temperature, age))
            val = griddata(points, values, xi, method='nearest')
        return 10 ** float(val)


    def GetMainSequenceAge(self, mass, key='Mass'):
        Tracks = self.Tracks
        ages = sorted(Tracks.keys())
        if key.lower() == "temperature":
            age = 100e6
            old_age = 0
            while abs(age - old_age) / age > 0.05:
                old_age = age
                m = self.GetFromTemperature(old_age, mass, key="Mass")
                age = self.GetMainSequenceAge(m) * 0.2
            return age
        elif key.lower() != 'mass':
            raise ValueError("Error! key = %s not supported in GetMainSequenceAge!" % key)

        #Find the masses that are common to at least the first few ages
        common_masses = list(Tracks[ages[0]]["Mass"])
        tol = 0.001
        for i in range(1, 3):
            age = ages[i]
            masses = np.array(Tracks[age]["Mass"])
            length = len(common_masses)
            badindices = []
            for j, m in enumerate(common_masses[::-1]):
                if np.min(np.abs(m - masses)) > tol:
                    badindices.append(length - 1 - j)
            for idx in badindices:
                common_masses.pop(idx)


        #Find the mass closest to the requested one.
        m1, m2 = HelperFunctions.GetSurrounding(common_masses, mass)
        if m1 < mass and m1 == common_masses[-1]:
            warnings.warn(
                "Requested mass ( %g ) is above the highest common mass in the evolutionary tracks ( %g )" % (mass, m1))
        elif m1 > mass and m1 == common_masses[0]:
            warnings.warn(
                "Requested mass ( %g ) is below the lowest common mass in the evolutionary tracks ( %g )" % (mass, m1))
        age1 = 0.0
        age2 = 0.0

        done = False
        i = 1
        while not done and i < len(ages):
            age = ages[i]
            masses = np.array(Tracks[age]["Mass"])
            done = True
            if np.min(np.abs(m1 - masses)) <= tol:
                age1 = age
                done = False
            if np.min(np.abs(m2 - masses)) <= tol:
                age2 = age
                done = False
            i += 1

        return 10 ** ((age1 - age2) / (m1 - m2) * (mass - m1) + age1)


    def GetSpectralType(self, temperature, interpolate=False):
        return self.MS.GetSpectralType(self, self.MS.Temperature, value, interpolate)


    #Get the factor you would need to multiply these tracks by to make the given star agree with MS relations
    def GetFactor(self, temperature, key='Mass'):
        MS_age = self.GetMainSequenceAge(temperature, key="Temperature")
        tracks_value = self.GetFromTemperature(MS_age, temperature, key=key)

        #Get the value from main sequence relations. The call signature is different, so need if statements
        spt = self.MS.GetSpectralType(self.MS.Temperature, temperature)
        if key.lower() == "mass":
            msr_value = self.MS.Interpolate(self.MS.Mass, spt)
        elif key.lower() == "radius":
            msr_value = self.MS.Interpolate(self.MS.Radius, spt)
        else:
            raise ValueError("Error! Key %s not allowed!" % key)

        return msr_value / tracks_value


if __name__ == "__main__":
    sptr = MainSequence()
    pms = PreMainSequence()
    for spt in ["K9", "K5", "K0", "G5", "G0"]:
        temp = sptr.Interpolate(sptr.Temperature, spt)
        radius = sptr.Interpolate(sptr.Radius, spt)
        print "%s:  T=%g\tR=%g" % (spt, temp, radius)
        print pms.Interpolate(spt, 1000, "radius")

    
    
    
    
    
    
    
    
    
    
    
    
    
