"""
This class was written by Stuart Littlefair to do barycentric velocity correction.

"""
from __future__ import division

from math import pi
import warnings

from numpy import cos, sin
from astropy import time
from astropy import constants as const
from astropy import units as u
from astropy.utils.iers import IERS
from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
from astropy import _erfa as erfa
import numpy as np



# mean sidereal rate (at J2000) in radians per (UT1) second
SR = 7.292115855306589e-5

''' time class: inherits astropy time object, and adds heliocentric, barycentric
    correction utilities.'''


class Time(time.Time):
    def __init__(self, *args, **kwargs):
        super(Time, self).__init__(*args, **kwargs)
        self.height = kwargs.get('height', 0.0)

    def _pvobs(self):
        '''calculates position and velocity of the observatory
           returns position/velocity in AU and AU/d in GCRS reference frame
        '''

        # convert obs position from WGS84 (lat long) to ITRF geocentric coords in AU
        xyz = self.location.to(u.AU).value

        # now we need to convert this position to Celestial Coords
        # specifically, the GCRS coords.
        # conversion from celestial to terrestrial coords given by
        # [TRS] = RPOM * R_3(ERA) * RC2I * [CRS]
        # where:
        # [CRS] is vector in GCRS (geocentric celestial system)
        # [TRS] is vector in ITRS (International Terrestrial Ref System)
        # ERA is earth rotation angle
        # RPOM = polar motion matrix

        tt = self.tt
        mjd = self.utc.mjd

        # we need the IERS values to correct for the precession/nutation of the Earth
        iers_tab = IERS.open()

        # Find UT1, which is needed to calculate ERA
        # uses IERS_B by default , for more recent times use IERS_A download
        try:
            ut1 = self.ut1
        except:
            try:
                iers_a_file = download_file(IERS_A_URL, cache=True)
                iers_a = IERS_A.open(iers_a_file)
                self.delta_ut1_utc = self.get_delta_ut1_utc(iers_a)
                ut1 = self.ut1
            except:
                # fall back to UTC with degraded accuracy
                warnings.warn('Cannot calculate UT1: using UTC with degraded accuracy')
                ut1 = self.utc

        # Gets x,y coords of Celestial Intermediate Pole (CIP) and CIO locator s
        # CIO = Celestial Intermediate Origin
        # Both in GCRS
        X, Y, S = erfa.xys00a(tt.jd1, tt.jd2)

        # Get dX and dY from IERS B
        dX = np.interp(mjd, iers_tab['MJD'], iers_tab['dX_2000A']) * u.arcsec
        dY = np.interp(mjd, iers_tab['MJD'], iers_tab['dY_2000A']) * u.arcsec

        # Get GCRS to CIRS matrix
        # can be used to convert to Celestial Intermediate Ref Sys
        # from GCRS.
        rc2i = erfa.c2ixys(X + dX.to(u.rad).value, Y + dY.to(u.rad).value, S)

        # Gets the Terrestrial Intermediate Origin (TIO) locator s'
        # Terrestrial Intermediate Ref Sys (TIRS) defined by TIO and CIP.
        # TIRS related to to CIRS by Earth Rotation Angle
        sp = erfa.sp00(tt.jd1, tt.jd2)

        # Get X and Y from IERS B
        # X and Y are
        xp = np.interp(mjd, iers_tab['MJD'], iers_tab['PM_x']) * u.arcsec
        yp = np.interp(mjd, iers_tab['MJD'], iers_tab['PM_y']) * u.arcsec

        # Get the polar motion matrix. Relates ITRF to TIRS.
        rpm = erfa.pom00(xp.to(u.rad).value, yp.to(u.rad).value, sp)

        # multiply ITRF position of obs by transpose of polar motion matrix
        # Gives Intermediate Ref Frame position of obs
        x, y, z = np.array([rpmMat.T.dot(xyz) for rpmMat in rpm]).T

        # Functions of Earth Rotation Angle, theta
        # Theta is angle bewtween TIO and CIO (along CIP)
        # USE UT1 here.
        theta = erfa.era00(ut1.jd1, ut1.jd2)
        S, C = np.sin(theta), np.cos(theta)

        # Position #GOT HERE
        pos = np.asarray([C * x - S * y, S * x + C * y, z]).T

        # multiply by inverse of GCRS to CIRS matrix
        # different methods for scalar times vs arrays
        if pos.ndim > 1:
            pos = np.array([np.dot(rc2i[j].T, pos[j]) for j in range(len(pos))])
        else:
            pos = np.dot(rc2i.T, pos)

        # Velocity
        vel = np.asarray([SR * (-S * x - C * y), SR * (C * x - S * y), np.zeros_like(x)]).T
        # multiply by inverse of GCRS to CIRS matrix
        if vel.ndim > 1:
            vel = np.array([np.dot(rc2i[j].T, vel[j]) for j in range(len(pos))])
        else:
            vel = np.dot(rc2i.T, vel)

        # return position and velocity
        return pos, vel

    def _obs_pos(self):
        '''calculates heliocentric and barycentric position of the earth in AU and AU/d'''
        tdb = self.tdb

        # get heliocentric and barycentric position and velocity of Earth
        # BCRS reference frame
        h_pv, b_pv = erfa.epv00(tdb.jd1, tdb.jd2)

        # h_pv etc can be shape (ntimes,2,3) or (2,3) if given a scalar time
        if h_pv.ndim == 2:
            h_pv = h_pv[np.newaxis, :]
        if b_pv.ndim == 2:
            b_pv = b_pv[np.newaxis, :]

        # unpack into position and velocity arrays
        h_pos = h_pv[:, 0, :]
        h_vel = h_pv[:, 1, :]

        # unpack into position and velocity arrays
        b_pos = b_pv[:, 0, :]
        b_vel = b_pv[:, 1, :]

        # now need position and velocity of observing station
        pos_obs, vel_obs = self._pvobs()

        #add this to heliocentric and barycentric position of center of Earth
        h_pos += pos_obs
        b_pos += pos_obs
        h_vel += vel_obs
        b_vel += vel_obs
        return (h_pos, h_vel, b_pos, b_vel)

    def _vect(self, coord):
        '''get unit vector pointing to star, and modulus of vector, in AU
           coordinate of star supplied as astropy.coordinate object

           assume zero proper motion, parallax and radial velocity'''
        pmra = pmdec = px = rv = 0.0

        rar = coord.ra.radian
        decr = coord.dec.radian
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore warnings about 0 parallax
            pos, vel = erfa.starpv(rar, decr, pmra, pmdec, px, rv)

        modulus = np.sqrt(pos.dot(pos))
        unit = pos / modulus
        modulus /= const.au.value
        return modulus, unit

    def hcor(self, coord):
        mod, spos = self._vect(coord)
        # get helio/bary-centric position and velocity of observatory, in AU, AU/d
        h_pos, h_vel, b_pos, b_vel = self._obs_pos()

        # heliocentric light travel time, s
        tcor_hel = const.au.value * np.array([np.dot(spos, hpos) for hpos in h_pos]) / const.c.value
        # print 'Correction to add to get time at heliocentre = %.7f s' % tcor_hel
        dt = time.TimeDelta(tcor_hel, format='sec', scale='tdb')
        return self.utc + dt

    def bcor(self, coord):
        mod, spos = self._vect(coord)
        # get helio/bary-centric position and velocity of observatory, in AU, AU/d
        h_pos, h_vel, b_pos, b_vel = self._obs_pos()

        # barycentric light travel time, s
        tcor_bar = const.au.value * np.array([np.dot(spos, bpos) for bpos in b_pos]) / const.c.value
        # print 'Correction to add to get time at barycentre  = %.7f s' % tcor_bar
        dt = time.TimeDelta(tcor_bar, format='sec', scale='tdb')
        return self.tdb + dt


# This is a direct port of x_keckhelio.pro from XIDL, written by Erik Tollerud

def x_keckhelio(ra, dec, epoch=2000.0, jd=None, tai=None,
                longitude=None, latitude=None, altitude=None, obs='keck'):
    """
    `ra` and `dec` in degrees
    Returns `vcorr`: "Velocity correction term, in km/s, to add to measured
                    radial velocity to convert it to the heliocentric frame."
    but the sign seems to be backwards of what that says:
    helio_shift = -1. * x_keckhelio(RA, DEC, 2000.0)
    uses barvel and ct2lst functions from idlastro, also ported below
    #NOTE: this seems to have some jitter about the IDL version at the .1 km/s level
    """

    if longitude is not None and latitude is not None and altitude is not None:
        print 'using long/lat/alt instead of named observatory'
    elif obs == 'keck':
        longitude = 360. - 155.47220
        latitude = 19.82656886
        altitude = 4000.  #meters
    else:
        print 'Using observatory', obs
        if obs == 'vlt':
            longitude = 360. - 70.40322
            latitude = -24.6258
            altitude = 2635.  #meters
        if obs == 'mmt':
            longitude = 360. - 110.88456
            latitude = 31.688778
            altitude = 2600.  #meters
        if obs == 'lick':
            longitude = 360. - 121.637222
            latitude = 37.343056
            altitude = 1283.  #meters
        if obs == 'mcdonald':
            longitude = 360. - 104.0226111111
            latitude = 30.6715
            altitude = 2076.
        else:
            raise ValueError('unrecognized observatory' + obs)

    if jd is None and tai is not None:
        jd = 2400000.5 + tai / (24. * 3600.)
    elif tai is None and jd is not None:
        pass
    else:
        raise ValueError('Must specify either JD or TAI')

    DRADEG = 180.0 / pi

    # ----------
    # Compute baryocentric velocity (Accurate only to 1m/s)
    dvelh, dvelb = baryvel(jd, epoch)

    #Project velocity toward star
    vbarycen = dvelb[0] * cos(dec / DRADEG) * cos(ra / DRADEG) + \
               dvelb[1] * cos(dec / DRADEG) * sin(ra / DRADEG) + dvelb[2] * sin(dec / DRADEG)

    #----------
    #Compute rotational velocity of observer on the Earth

    #LAT is the latitude in radians.
    latrad = latitude / DRADEG

    #Reduction of geodetic latitude to geocentric latitude (radians).
    #DLAT is in arcseconds.

    dlat = -(11. * 60. + 32.743000) * sin(2. * latrad) + \
           1.163300 * sin(4. * latrad) - 0.002600 * sin(6. * latrad)
    latrad = latrad + (dlat / 3600.) / DRADEG

    #R is the radius vector from the Earth's center to the observer (meters).
    #VC is the corresponding circular velocity
    #(meters/sidereal day converted to km / sec).
    #(sidereal day = 23.934469591229 hours (1986))

    r = 6378160.0 * (0.998327073 + 0.00167643800 * cos(2. * latrad) - \
                     0.00000351 * cos(4. * latrad) + 0.000000008 * cos(6. * latrad)) \
        + altitude
    vc = 2. * pi * (r / 1000.) / (23.934469591229 * 3600.)

    #Compute the hour angle, HA, in degrees
    LST = 15. * ct2lst(longitude, 'junk', jd)  #  convert from hours to degrees
    HA = LST - ra

    #Project the velocity onto the line of sight to the star.
    vrotate = vc * cos(latrad) * cos(dec / DRADEG) * sin(HA / DRADEG)

    return (-vbarycen + vrotate)


def ct2lst(lng, tz, jd, day=None, mon=None, year=None):
    """
    # NAME:
    #     CT2LST
    # PURPOSE:
    #     To convert from Local Civil Time to Local Mean Sidereal Time.
    #
    # CALLING SEQUENCE:
    #     CT2LST, Lst, Lng, Tz, Time, [Day, Mon, Year] #NOT SUPPORTED IN PYTHON PORT!
    #                       or
    #     CT2LST, Lst, Lng, dummy, JD
    #
    # INPUTS:
    #     Lng  - The longitude in degrees (east of Greenwich) of the place for
    #            which the local sidereal time is desired, scalar.   The Greenwich
    #            mean sidereal time (GMST) can be found by setting Lng = 0.
    #     Tz  - The time zone of the site in hours, positive East  of the Greenwich
    #           meridian (ahead of GMT).  Use this parameter to easily account
    #           for Daylight Savings time (e.g. -4=EDT, -5 = EST/CDT), scalar
    #           This parameter is not needed (and ignored) if Julian date is
    #           supplied.    ***Note that the sign of TZ was changed in July 2008
    #           to match the standard definition.***
    #     Time or JD  - If more than four parameters are specified, then this is
    #               the time of day of the specified date in decimal hours.  If
    #               exactly four parameters are specified, then this is the
    #               Julian date of time in question, scalar or vector
    #
    # OPTIONAL INPUTS:
    #      Day -  The day of the month (1-31),integer scalar or vector
    #      Mon -  The month, in numerical format (1-12), integer scalar or vector
    #      Year - The 4 digit year (e.g. 2008), integer scalar or vector
    #
    # OUTPUTS:
    #       Lst   The Local Sidereal Time for the date/time specified in hours.
    #
    # RESTRICTIONS:
    #       If specified, the date should be in numerical form.  The year should
    #       appear as yyyy.
    #
    # PROCEDURE:
    #       The Julian date of the day and time is question is used to determine
    #       the number of days to have passed since 0 Jan 2000.  This is used
    #       in conjunction with the GST of that date to extrapolate to the current
    #       GST# this is then used to get the LST.    See Astronomical Algorithms
    #       by Jean Meeus, p. 84 (Eq. 11-4) for the constants used.
    #
    # EXAMPLE:
    #       Find the Greenwich mean sidereal time (GMST) on 2008 Jul 30 at 15:53 pm
    #       in Baltimore, Maryland (longitude=-76.72 degrees).   The timezone is
    #       EDT or tz=-4
    #
    #       IDL> CT2LST, lst, -76.72, -4,ten(15,53), 30, 07, 2008
    #
    #               ==> lst =  11.356505  hours  (= 11h 21m 23.418s)
    #
    #       The Web site  http://tycho.usno.navy.mil/sidereal.html contains more
    #       info on sidereal time, as well as an interactive calculator.
    # PROCEDURES USED:
    #       jdcnv - Convert from year, month, day, hour to julian date
    #
    # MODIFICATION HISTORY:
    #     Adapted from the FORTRAN program GETSD by Michael R. Greason, STX,
    #               27 October 1988.
    #     Use IAU 1984 constants Wayne Landsman, HSTX, April 1995, results
    #               differ by about 0.1 seconds
    #     Longitudes measured *east* of Greenwich   W. Landsman    December 1998
    #     Time zone now measure positive East of Greenwich W. Landsman July 2008
    #     Remove debugging print statement  W. Landsman April 2009
    """

    # IF N_params() gt 4 THEN BEGIN
    # time = tme - tz
    # jdcnv, year, mon, day, time, jd

    # ENDIF ELSE jd = double(tme)

    #
    #                            Useful constants, see Meeus, p.84
    #
    c = [280.46061837, 360.98564736629, 0.000387933, 38710000.0]
    jd2000 = 2451545.0
    t0 = jd - jd2000
    t = t0 / 36525
    #
    #                            Compute GST in seconds.
    #
    theta = c[0] + (c[1] * t0) + t ** 2 * (c[2] - t / c[3])
    #
    #                            Compute LST in hours.
    #
    lst = np.array((theta + lng) / 15.0)
    neg = lst < 0
    if np.sum(neg) > 0:
        if neg.shape == tuple():
            lst = 24. + idl_like_mod(lst, 24.)
        else:
            lst[neg] = 24. + idl_like_mod(lst[neg], 24.)
    return idl_like_mod(lst, 24.)


def baryvel(dje, deq):
    #+
    # NAME:
    #       BARYVEL
    # PURPOSE:
    #       Calculates heliocentric and barycentric velocity components of Earth.
    #
    # EXPLANATION:
    #       BARYVEL takes into account the Earth-Moon motion, and is useful for
    #       radial velocity work to an accuracy of  ~1 m/s.
    #
    # CALLING SEQUENCE:
    #       BARYVEL, dje, deq, dvelh, dvelb, [ JPL =  ]
    #
    # INPUTS:
    #       DJE - (scalar) Julian ephemeris date.
    #       DEQ - (scalar) epoch of mean equinox of dvelh and dvelb. If deq=0
    #               then deq is assumed to be equal to dje.
    # OUTPUTS:
    #       DVELH: (vector(3)) heliocentric velocity component. in km/s
    #       DVELB: (vector(3)) barycentric velocity component. in km/s
    #
    #       The 3-vectors DVELH and DVELB are given in a right-handed coordinate
    #       system with the +X axis toward the Vernal Equinox, and +Z axis
    #       toward the celestial pole.
    #
    # OPTIONAL KEYWORD SET:
    #       JPL - if /JPL set, then BARYVEL will call the procedure JPLEPHINTERP
    #             to compute the Earth velocity using the full JPL ephemeris.
    #             The JPL ephemeris FITS file JPLEPH.405 must exist in either the
    #             current directory, or in the directory specified by the
    #             environment variable ASTRO_DATA.   Alternatively, the JPL keyword
    #             can be set to the full path and name of the ephemeris file.
    #             A copy of the JPL ephemeris FITS file is available in
    #                 http://idlastro.gsfc.nasa.gov/ftp/data/
    # PROCEDURES CALLED:
    #       Function PREMAT() -- computes precession matrix
    #       JPLEPHREAD, JPLEPHINTERP, TDB2TDT - if /JPL keyword is set
    # NOTES:
    #       Algorithm taken from FORTRAN program of Stumpff (1980, A&A Suppl, 41,1)
    #       Stumpf claimed an accuracy of 42 cm/s for the velocity.    A
    #       comparison with the JPL FORTRAN planetary ephemeris program PLEPH
    #       found agreement to within about 65 cm/s between 1986 and 1994
    #
    #       If /JPL is set (using JPLEPH.405 ephemeris file) then velocities are
    #       given in the ICRS system# otherwise in the FK4 system.
    # EXAMPLE:
    #       Compute the radial velocity of the Earth toward Altair on 15-Feb-1994
    #          using both the original Stumpf algorithm and the JPL ephemeris
    #
    #       IDL> jdcnv, 1994, 2, 15, 0, jd          #==> JD = 2449398.5
    #       IDL> baryvel, jd, 2000, vh, vb          #Original algorithm
    #               ==> vh = [-17.07243, -22.81121, -9.889315]  #Heliocentric km/s
    #               ==> vb = [-17.08083, -22.80471, -9.886582]  #Barycentric km/s
    #       IDL> baryvel, jd, 2000, vh, vb, /jpl   #JPL ephemeris
    #               ==> vh = [-17.07236, -22.81126, -9.889419]  #Heliocentric km/s
    #               ==> vb = [-17.08083, -22.80484, -9.886409]  #Barycentric km/s
    #
    #       IDL> ra = ten(19,50,46.77)*15/!RADEG    #RA  in radians
    #       IDL> dec = ten(08,52,3.5)/!RADEG        #Dec in radians
    #       IDL> v = vb[0]*cos(dec)*cos(ra) + $   #Project velocity toward star
    #               vb[1]*cos(dec)*sin(ra) + vb[2]*sin(dec)
    #
    # REVISION HISTORY:
    #       Jeff Valenti,  U.C. Berkeley    Translated BARVEL.FOR to IDL.
    #       W. Landsman, Cleaned up program sent by Chris McCarthy (SfSU) June 1994
    #       Converted to IDL V5.0   W. Landsman   September 1997
    #       Added /JPL keyword  W. Landsman   July 2001
    #       Documentation update W. Landsman Dec 2005
    #-
    #Define constants
    dc2pi = 2 * pi
    cc2pi = dc2pi
    dc1 = 1.0
    dcto = 2415020.0
    dcjul = 36525.0  #days in Julian year
    dcbes = 0.313
    dctrop = 365.24219572  #days in tropical year (...572 insig)
    dc1900 = 1900.0
    AU = 1.4959787e8

    #Constants dcfel(i,k) of fast changing elements.
    dcfel = [1.7400353e00, 6.2833195099091e02, 5.2796e-6 \
        , 6.2565836e00, 6.2830194572674e02, -2.6180e-6 \
        , 4.7199666e00, 8.3997091449254e03, -1.9780e-5 \
        , 1.9636505e-1, 8.4334662911720e03, -5.6044e-5 \
        , 4.1547339e00, 5.2993466764997e01, 5.8845e-6 \
        , 4.6524223e00, 2.1354275911213e01, 5.6797e-6 \
        , 4.2620486e00, 7.5025342197656e00, 5.5317e-6 \
        , 1.4740694e00, 3.8377331909193e00, 5.6093e-6]
    dcfel = np.array(dcfel).reshape(8, 3)

    #constants dceps and ccsel(i,k) of slowly changing elements.
    dceps = [4.093198e-1, -2.271110e-4, -2.860401e-8]
    ccsel = [1.675104E-2, -4.179579E-5, -1.260516E-7 \
        , 2.220221E-1, 2.809917E-2, 1.852532E-5 \
        , 1.589963E00, 3.418075E-2, 1.430200E-5 \
        , 2.994089E00, 2.590824E-2, 4.155840E-6 \
        , 8.155457E-1, 2.486352E-2, 6.836840E-6 \
        , 1.735614E00, 1.763719E-2, 6.370440E-6 \
        , 1.968564E00, 1.524020E-2, -2.517152E-6 \
        , 1.282417E00, 8.703393E-3, 2.289292E-5 \
        , 2.280820E00, 1.918010E-2, 4.484520E-6 \
        , 4.833473E-2, 1.641773E-4, -4.654200E-7 \
        , 5.589232E-2, -3.455092E-4, -7.388560E-7 \
        , 4.634443E-2, -2.658234E-5, 7.757000E-8 \
        , 8.997041E-3, 6.329728E-6, -1.939256E-9 \
        , 2.284178E-2, -9.941590E-5, 6.787400E-8 \
        , 4.350267E-2, -6.839749E-5, -2.714956E-7 \
        , 1.348204E-2, 1.091504E-5, 6.903760E-7 \
        , 3.106570E-2, -1.665665E-4, -1.590188E-7]
    ccsel = np.array(ccsel).reshape(17, 3)

    #Constants of the arguments of the short-period perturbations.
    dcargs = [5.0974222, -7.8604195454652e2 \
        , 3.9584962, -5.7533848094674e2 \
        , 1.6338070, -1.1506769618935e3 \
        , 2.5487111, -3.9302097727326e2 \
        , 4.9255514, -5.8849265665348e2 \
        , 1.3363463, -5.5076098609303e2 \
        , 1.6072053, -5.2237501616674e2 \
        , 1.3629480, -1.1790629318198e3 \
        , 5.5657014, -1.0977134971135e3 \
        , 5.0708205, -1.5774000881978e2 \
        , 3.9318944, 5.2963464780000e1 \
        , 4.8989497, 3.9809289073258e1 \
        , 1.3097446, 7.7540959633708e1 \
        , 3.5147141, 7.9618578146517e1 \
        , 3.5413158, -5.4868336758022e2]
    dcargs = np.array(dcargs).reshape(15, 2)

    #Amplitudes ccamps(n,k) of the short-period perturbations.
    ccamps = \
        [-2.279594E-5, 1.407414E-5, 8.273188E-6, 1.340565E-5, -2.490817E-7 \
            , -3.494537E-5, 2.860401E-7, 1.289448E-7, 1.627237E-5, -1.823138E-7 \
            , 6.593466E-7, 1.322572E-5, 9.258695E-6, -4.674248E-7, -3.646275E-7 \
            , 1.140767E-5, -2.049792E-5, -4.747930E-6, -2.638763E-6, -1.245408E-7 \
            , 9.516893E-6, -2.748894E-6, -1.319381E-6, -4.549908E-6, -1.864821E-7 \
            , 7.310990E-6, -1.924710E-6, -8.772849E-7, -3.334143E-6, -1.745256E-7 \
            , -2.603449E-6, 7.359472E-6, 3.168357E-6, 1.119056E-6, -1.655307E-7 \
            , -3.228859E-6, 1.308997E-7, 1.013137E-7, 2.403899E-6, -3.736225E-7 \
            , 3.442177E-7, 2.671323E-6, 1.832858E-6, -2.394688E-7, -3.478444E-7 \
            , 8.702406E-6, -8.421214E-6, -1.372341E-6, -1.455234E-6, -4.998479E-8 \
            , -1.488378E-6, -1.251789E-5, 5.226868E-7, -2.049301E-7, 0.E0 \
            , -8.043059E-6, -2.991300E-6, 1.473654E-7, -3.154542E-7, 0.E0 \
            , 3.699128E-6, -3.316126E-6, 2.901257E-7, 3.407826E-7, 0.E0 \
            , 2.550120E-6, -1.241123E-6, 9.901116E-8, 2.210482E-7, 0.E0 \
            , -6.351059E-7, 2.341650E-6, 1.061492E-6, 2.878231E-7, 0.E0]
    ccamps = np.array(ccamps).reshape(15, 5)

    #Constants csec3 and ccsec(n,k) of the secular perturbations in longitude.
    ccsec3 = -7.757020E-8
    ccsec = [1.289600E-6, 5.550147E-1, 2.076942E00 \
        , 3.102810E-5, 4.035027E00, 3.525565E-1 \
        , 9.124190E-6, 9.990265E-1, 2.622706E00 \
        , 9.793240E-7, 5.508259E00, 1.559103E01]
    ccsec = np.array(ccsec).reshape(4, 3)

    #Sidereal rates.
    dcsld = 1.990987e-7  #sidereal rate in longitude
    ccsgd = 1.990969E-7  #sidereal rate in mean anomaly

    #Constants used in the calculation of the lunar contribution.
    cckm = 3.122140E-5
    ccmld = 2.661699E-6
    ccfdi = 2.399485E-7

    #Constants dcargm(i,k) of the arguments of the perturbations of the motion
    # of the moon.
    dcargm = [5.1679830, 8.3286911095275e3 \
        , 5.4913150, -7.2140632838100e3 \
        , 5.9598530, 1.5542754389685e4]
    dcargm = np.array(dcargm).reshape(3, 2)

    #Amplitudes ccampm(n,k) of the perturbations of the moon.
    ccampm = [1.097594E-1, 2.896773E-7, 5.450474E-2, 1.438491E-7 \
        , -2.223581E-2, 5.083103E-8, 1.002548E-2, -2.291823E-8 \
        , 1.148966E-2, 5.658888E-8, 8.249439E-3, 4.063015E-8]
    ccampm = np.array(ccampm).reshape(3, 4)

    #ccpamv(k)=a*m*dl,dt (planets), dc1mme=1-mass(earth+moon)
    ccpamv = [8.326827E-11, 1.843484E-11, 1.988712E-12, 1.881276E-12]
    dc1mme = 0.99999696

    #Time arguments.
    dt = (dje - dcto) / dcjul
    tvec = np.array([1., dt, dt * dt])

    #Values of all elements for the instant(aneous?) dje.
    temp = idl_like_mod(idl_like_pound(tvec, dcfel), dc2pi)
    #PROBLEM: the mod here is where the 100 m/s error slips in
    dml = temp[:, 0]
    forbel = temp[:, 1:8]
    g = forbel[:, 0]  #old fortran equivalence

    deps = idl_like_mod(np.sum(tvec * dceps), dc2pi)
    sorbel = idl_like_mod(idl_like_pound(tvec, ccsel), dc2pi)
    e = sorbel[:, 0]  #old fortran equivalence

    #Secular perturbations in longitude.
    dummy = cos(2.0)
    sn = sin(idl_like_mod(idl_like_pound(tvec.ravel()[0:2], ccsec[:, 1:3]), cc2pi))

    #Periodic perturbations of the emb (earth-moon barycenter).
    pertl = np.sum(ccsec[:, 0] * sn) + dt * ccsec3 * sn.ravel()[2]
    pertld = 0.0
    pertr = 0.0
    pertrd = 0.0

    for k in range(14):
        a = idl_like_mod((dcargs[k, 0] + dt * dcargs[k, 1]), dc2pi)
        cosa = cos(a)
        sina = sin(a)
        pertl = pertl + ccamps[k, 0] * cosa + ccamps[k, 1] * sina
        pertr = pertr + ccamps[k, 2] * cosa + ccamps[k, 3] * sina
        if k < 11:
            pertld = pertld + (ccamps[k, 1] * cosa - ccamps[k, 0] * sina) * ccamps[k, 4]
            pertrd = pertrd + (ccamps[k, 3] * cosa - ccamps[k, 2] * sina) * ccamps[k, 4]

    #Elliptic part of the motion of the emb.
    phi = (e * e / 4) * (((8 / e) - e) * sin(g) + 5 * sin(2 * g) + (13 / 3) * e * sin(3 * g))
    f = g + phi
    sinf = sin(f)
    cosf = cos(f)
    dpsi = (dc1 - e * e) / (dc1 + e * cosf)
    phid = 2 * e * ccsgd * ((1 + 1.5 * e * e) * cosf + e * (1.25 - 0.5 * sinf * sinf))
    psid = ccsgd * e * sinf * (dc1 - e * e) ** -0.5

    #Perturbed heliocentric motion of the emb.
    d1pdro = dc1 + pertr
    drd = d1pdro * (psid + dpsi * pertrd)
    drld = d1pdro * dpsi * (dcsld + phid + pertld)
    dtl = idl_like_mod((dml + phi + pertl), dc2pi)
    dsinls = sin(dtl)
    dcosls = cos(dtl)
    dxhd = drd * dcosls - drld * dsinls
    dyhd = drd * dsinls + drld * dcosls

    #Influence of eccentricity, evection and variation on the geocentric
    # motion of the moon.
    pertl = 0.0
    pertld = 0.0
    pertp = 0.0
    pertpd = 0.0
    for k in range(2):
        a = idl_like_mod((dcargm[k, 0] + dt * dcargm[k, 1]), dc2pi)
        sina = sin(a)
        cosa = cos(a)
        pertl = pertl + ccampm[k, 0] * sina
        pertld = pertld + ccampm[k, 1] * cosa
        pertp = pertp + ccampm[k, 2] * cosa
        pertpd = pertpd - ccampm[k, 3] * sina

    #Heliocentric motion of the earth.
    tl = forbel.ravel()[1] + pertl
    sinlm = sin(tl)
    coslm = cos(tl)
    sigma = cckm / (1.0 + pertp)
    a = sigma * (ccmld + pertld)
    b = sigma * pertpd
    dxhd = dxhd + a * sinlm + b * coslm
    dyhd = dyhd - a * coslm + b * sinlm
    dzhd = -sigma * ccfdi * cos(forbel.ravel()[2])

    #Barycentric motion of the earth.
    dxbd = dxhd * dc1mme
    dybd = dyhd * dc1mme
    dzbd = dzhd * dc1mme
    for k in range(3):
        plon = forbel.ravel()[k + 3]
        pomg = sorbel.ravel()[k + 1]
        pecc = sorbel.ravel()[k + 9]
        tl = idl_like_mod((plon + 2.0 * pecc * sin(plon - pomg)), cc2pi)
        dxbd = dxbd + ccpamv[k] * (sin(tl) + pecc * sin(pomg))
        dybd = dybd - ccpamv[k] * (cos(tl) + pecc * cos(pomg))
        dzbd = dzbd - ccpamv[k] * sorbel.ravel()[k + 13] * cos(plon - sorbel.ravel()[k + 5])

    #Transition to mean equator of date.
    dcosep = cos(deps)
    dsinep = sin(deps)
    dyahd = dcosep * dyhd - dsinep * dzhd
    dzahd = dsinep * dyhd + dcosep * dzhd
    dyabd = dcosep * dybd - dsinep * dzbd
    dzabd = dsinep * dybd + dcosep * dzbd

    #Epoch of mean equinox (deq) of zero implies that we should use
    # Julian ephemeris date (dje) as epoch of mean equinox.
    if deq == 0:
        dvelh = AU * ([dxhd, dyahd, dzahd])
        dvelb = AU * ([dxbd, dyabd, dzabd])
        return dvelh, dvelb

    #General precession from epoch dje to deq.
    deqdat = (dje - dcto - dcbes) / dctrop + dc1900
    prema = premat(deqdat, deq, FK4=True)

    dvelh = AU * idl_like_pound(prema, [dxhd, dyahd, dzahd])
    dvelb = AU * idl_like_pound(prema, [dxbd, dyabd, dzabd])

    return dvelh, dvelb


def premat(equinox1, equinox2, FK4=False):
    """
    #+
    # NAME:
    #       PREMAT
    # PURPOSE:
    #       Return the precession matrix needed to go from EQUINOX1 to EQUINOX2.
    # EXPLANTION:
    #       This matrix is used by the procedures PRECESS and BARYVEL to precess
    #       astronomical coordinates
    #
    # CALLING SEQUENCE:
    #       matrix = PREMAT( equinox1, equinox2, [ /FK4 ] )
    #
    # INPUTS:
    #       EQUINOX1 - Original equinox of coordinates, numeric scalar.
    #       EQUINOX2 - Equinox of precessed coordinates.
    #
    # OUTPUT:
    #      matrix - double precision 3 x 3 precession matrix, used to precess
    #               equatorial rectangular coordinates
    #
    # OPTIONAL INPUT KEYWORDS:
    #       /FK4   - If this keyword is set, the FK4 (B1950.0) system precession
    #               angles are used to compute the precession matrix.   The
    #               default is to use FK5 (J2000.0) precession angles
    #
    # EXAMPLES:
    #       Return the precession matrix from 1950.0 to 1975.0 in the FK4 system
    #
    #       IDL> matrix = PREMAT( 1950.0, 1975.0, /FK4)
    #
    # PROCEDURE:
    #       FK4 constants from "Computational Spherical Astronomy" by Taff (1983),
    #       p. 24. (FK4). FK5 constants from "Astronomical Almanac Explanatory
    #       Supplement 1992, page 104 Table 3.211.1.
    #
    # REVISION HISTORY
    #       Written, Wayne Landsman, HSTX Corporation, June 1994
    #       Converted to IDL V5.0   W. Landsman   September 1997
    #-
    """

    deg_to_rad = pi / 180.0
    sec_to_rad = deg_to_rad / 3600.

    T = 0.001 * ( equinox2 - equinox1)

    if not FK4:  # FK5
        ST = 0.001 * ( equinox1 - 2000.)
        #  Compute 3 rotation angles
        A = sec_to_rad * T * (23062.181 + ST * (139.656 + 0.0139 * ST) \
                              + T * (30.188 - 0.344 * ST + 17.998 * T))

        B = sec_to_rad * T * T * (79.280 + 0.410 * ST + 0.205 * T) + A

        C = sec_to_rad * T * (20043.109 - ST * (85.33 + 0.217 * ST) \
                              + T * (-42.665 - 0.217 * ST - 41.833 * T))

    else:
        ST = 0.001 * ( equinox1 - 1900.)
        #  Compute 3 rotation angles

        A = sec_to_rad * T * (23042.53 + ST * (139.75 + 0.06 * ST) \
                              + T * (30.23 - 0.27 * ST + 18.0 * T))

        B = sec_to_rad * T * T * (79.27 + 0.66 * ST + 0.32 * T) + A

        C = sec_to_rad * T * (20046.85 - ST * (85.33 + 0.37 * ST) \
                              + T * (-42.67 - 0.37 * ST - 41.8 * T))

    sina = sin(A)
    sinb = sin(B)
    sinc = sin(C)
    cosa = cos(A)
    cosb = cos(B)
    cosc = cos(C)

    r = np.empty([3, 3])
    r[:, 0] = [cosa * cosb * cosc - sina * sinb, sina * cosb + cosa * sinb * cosc, cosa * sinc]
    r[:, 1] = [-cosa * sinb - sina * cosb * cosc, cosa * cosb - sina * sinb * cosc, -sina * sinc]
    r[:, 2] = [-cosb * sinc, -sinb * sinc, cosc]

    return r


def idl_like_pound(a, b):
    a = np.array(a, copy=False)
    b = np.array(b, copy=False)

    if len(a.shape) == 2 and len(b.shape) == 1:
        return np.dot(a.T, b)
    if len(a.shape) == 1 and len(b.shape) == 2:
        res = np.dot(a, b.T)
        return res.reshape(1, res.size)
    else:
        return np.dot(a, b)


def idl_like_mod(a, b):
    a = np.array(a, copy=False)
    b = np.array(b, copy=False)
    res = np.abs(a) % b
    if a.shape == tuple():
        if a < 0:
            return -res
        else:
            return res
    else:
        res[a < 0] *= -1
        return res