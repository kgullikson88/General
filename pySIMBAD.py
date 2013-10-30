#!/usr/bin/python
# -*- coding: utf-8 -*-

### PySIMBAD -----------
#
#  M. Emre Aydın
#  VERSION : 0.0.7
#  10 Dec 2010, 01:30 AM
#
### --------------------

import urllib2
import sys
import re

sirius="http://simbad.u-strasbg.fr/simbad/sim-id?Ident=sirius"

def buildLink(value, cfa_mirror=False) :
	value = value.replace("+","%2B")
	value = value.replace("#","%23")
	value = value.replace(" ","+")
	if cfa_mirror:
	  value = "http://simbad.cfa.harvard.edu/simbad/sim-id?Ident=" + value
	else:
	  value = "http://simbad.u-strasbg.fr/simbad/sim-id?Ident=" + value
	return value

class simbad(object) :
	def __init__(self, link) :

		req=urllib2.Request(link)
		try :
			response=urllib2.urlopen(req)
			self.page = response.read()
		except :
			raise ConnectionError(link)

		if re.search("Identifier not found in the database :",self.page,re.IGNORECASE) : raise SimbadError("Identifier not found")
		if re.search("this identifier has an incorrect format",self.page,re.IGNORECASE) : raise SimbadError("Incorrect identifier format")
		if re.search("For querying by bibcode",self.page,re.IGNORECASE) : raise SimbadError("You searched for bibcode")
		if re.search("For querying by coordinates",self.page,re.IGNORECASE) : raise SimbadError("You searched for coordinates")
		if re.search("No known catalog could be found",self.page,re.IGNORECASE) : raise SimbadError("You searched for catalog")

	def flux_num(self) :
		"""Returns the number of fluxes of the object. Does NOT count the result of flux(), it reads it directly from the HTML. So if flux_num() and flux().__len__() aren't equal, than something's wrong."""
		# Gets the HTML Tag which says Fluxes (#)
		htmlTag = "Fluxes\s\(.*\)\s:"
		try :
			tagSpan = re.search(htmlTag,self.page,re.IGNORECASE).span()
			return self.page[tagSpan[0]+8:tagSpan[1]-3]
		except : raise HtmlTagError("Fluxes")


	def objectTypes(self) :
		"""Returns a list of other object types as coded by simbad"""
		try :
			start = re.search("Other object types:",self.page,re.IGNORECASE).span()[0]
			end = start+re.search("<TD NOWRAP>",self.page[start:],re.IGNORECASE).span()[0]
			types = re.findall("(?!\()(?<=<TT>\s).*",self.page[start:end],re.IGNORECASE)
			return types
		except :
			raise HtmlTagError("Object Types")

	def mainType(self) :
		"""Returns the string value of the main type of the object"""
		try :
			return re.search("(?<=--\s\s).*(?=\s\s\s\s\s</FONT>)",self.page,re.IGNORECASE).group()
		except : raise HtmlTagError("Main Object Type")


	def SpectralType(self):
		"""Returns the string value of the spectral type for the object"""
		try :
			start = re.search("Spectral type:", self.page).start()
			end = start + 200
			return self.page[start:end].split("\n")[5].split("<")[0].strip()
	        except : raise HtmlTagError("Spectral Type")
	        
	        
	def Parallax(self):
	        """Returns the parallax of the object, in mas"""
	        try:
	                start = re.search("Parallaxes", self.page).start()
	                end = start + 200
	                return float(self.page[start:end].split("\n")[5].split("[")[0].strip())
	        except : raise HtmlTagError("Parallax")
	        	



	def flux(self) :
		"""Returns a dictionary of Fluxes according to their Colors.
		Example :
		>>> S.flux()
		{'H': '-1.391', 'J': '-1.391', 'B': '-1.46', 'K': '-1.390', 'V': '-1.47'}	
		>>> S.flux()['J']
		'-1.391'
		"""
		# The regular expression for the color and magnitude values.
		fluxTag = "<tr>\s*<td>\s*<b>\s*<tt>\s*[^<,^\s]\s*.*\["
		try :
			fluxAll_temp = re.findall(fluxTag,self.page,re.IGNORECASE)
			fluxes = {}
			# Splitting the values from the HTML and appending them to a dictionary.
			for i in fluxAll_temp : fluxes[i[:-1].split("\n")[-1].split()[0]] = i[:-1].split("\n")[-1].split()[1]
			return fluxes
		except : raise HtmlTagError("Flux for each color")

	def names_num(self) :
		""" Returns the number of identifiers for the object. This does NOT return the lenght of the arguments returned from names(), instead, it is read directly from the HTML. So if they don't match somehow, it means something's wrong."""
		# Gets the HTML Tag which says Identifiers (#)
		htmlTag = "Identifiers\s\(.*\)"
		try :
			tagSpan = re.search(htmlTag,self.page,re.IGNORECASE).span()
			return self.page[tagSpan[0]+13:tagSpan[1]-1]
		except : raise HtmlTagError("Identifiers")

	def names(self) :
		"""Returns a list with every Identifier avabilable on Simbad."""

		# The regular expression for the names.
		namesTag = "<tt>\s*<a href.*Dic-Simbad.*>.*</a>\s*.*\s*</tt>"
		try :
			namesAll_temp = re.findall(namesTag,self.page,re.IGNORECASE)
			namesAll = []
			# Splitting the values from the HTML and appending them to the list.
			for i in namesAll_temp : namesAll.append(i.split("\n")[1].split("</A>")[0].split(">")[1] + i.split("\n")[1].split("</A>")[1])
			return namesAll
		except : raise HtmlTagError("Dic-Simbad")

	def getCoord(self,coordType):
		"""This method is developed for getting the coordinates easily. You can specify the type of coordinate to get as a string. the fk4, fk5, icrs, gal methods also use this method. Example usage : getCoord('fk5')"""
		# Develop the regular expression to search. (the BOLD HTML tag of the coordinate type)
		htmlTag = "<b>\s*%s\s*</b>" % coordType
		try :
			# The END position of the HTML tag.
			tagEnd = re.search(htmlTag,self.page,re.IGNORECASE).end()
			# The position of the Coordinates AFTER the HTML tag.
			coord_pos = re.search("<tt>\s*.*\s*\(",self.page[tagEnd:],re.IGNORECASE).span()
			return self.page[coord_pos[0]+tagEnd+4:coord_pos[1]+tagEnd-1].strip()
		except :
			raise HtmlTagError(coordType)

	def fk5(self) :
		"""Returns the FK5 Coordinates as a string if they exist."""
		return self.getCoord("fk5")

	def icrs(self) :
		"""Returns the ICRS Coordinates as a string if they exist."""
		return self.getCoord("icrs")

	def gal(self) :
		"""Returns the Galactic Coordinates as a string if they exist."""
		return self.getCoord("gal")

	def fk4(self) : 
		"""Returns the FK4 Coordinates as a string if they exist."""
		return self.getCoord("fk4")

	def ra(self,coord) :
		"""Returns a string RA joined by ':'"""
		try : return ":".join(coord.split()[:3])
		except : pass

	def dec(self,coord) :
		"""Returns a string DEC joined by ':'"""
		try : return ":".join(coord.split()[3:])
		except : pass

	def refs(self) :
		"""Returns the number of References found by Simbad."""
		tag="<b>\s*References\s\(\d*\sbetween"
		try : return re.findall(tag,self.page,re.IGNORECASE)[0].split("(")[1].split()[0]
		except : raise HtmlTagError("References")


## Definitions of the object types by Simbad.

objectDictionary = {	
		'?':'Object of unknown nature',
		'Rad':'Radio-source',
		'mR':'metric Radio-source',
		'cm':'centimetric Radio-source',
		'mm':'millimetric Radio-source',
		'smm':'sub-millimetric source',
		'Mas':'Maser',
		'IR':'Infra-Red source',
		'FIR':'Far-IR source (λ >= 30 µm)',
		'NIR':'Near-IR source (λ < 10 µm)',
		'red':'Very red source',
		'ERO':'Extremely Red Object',
		'blu':'Blue object',
		'UV':'UV-emission source',
		'X':'X-ray source',
		'gam':'gamma-ray source',
		'gB':'gamma-ray Burst',
		'grv':'Gravitational Source',
		'Lev':'(Micro)Lensing Event',
		'Le?':'Possible gravitationally lensed image',
		'gLe':'Gravitational Lens',
		'..?':'Candidate objects',
		'Cl?':'Possible Cluster of Galaxies',
		'Gr?':'Possible Group of Galaxies',
		'**?':'Interacting Binary Candidate',
		'EB?':'Eclipsing Binary Candidate',
		'CV?':'Cataclysmic Binary Candidate',
		'XB?':'X-ray binary Candidate',
		'LX?':'Low-Mass X-ray binary Candidate',
		'HX?':'High-Mass X-ray binary Candidate',
		'Pec?':'Possible Peculiar Star',
		'Y*?':'Young Stellar Object Candidate',
		'pr?':'Pre-main sequence Star Candidate',
		'TT?':'T Tau star Candidate',
		'C*?':'Possible Carbon Star',
		'S*?':'Possible S Star',
		'OH?':'Possible Star with envelope of OH/IR type',
		'CH?':'Possible Star with envelope of CH type',
		'WR?':'Possible Wolf-Rayet Star',
		'HB?':'Possible Horizontal Branch Star',
		'AB?':'Possible Asymptotic Giant Branch Star',
		'pA?':'Post-AGB Star Candidate',
		'WD?':'White Dwarf Candidate',
		'N*?':'Neutron Star Candidate',
		'BH?':'Black Hole Candidate',
		'BD?':'Brown Dwarf Candidate',
		'mul':'Composite object',
		'reg':'Region defined in the sky',
		'vid':'Underdense region of the Universe',
		'SCG':'Supercluster of Galaxies',
		'Clg':'Cluster of Galaxies',
		'GrG':'Group of Galaxies',
		'CGG':'Compact Group of Galaxies',
		'PaG':'Pair of Galaxies',
		'IG':'Interacting Galaxies',
		'Gl?':'Possible Globular Cluster',
		'Cl*':'Cluster of Stars',
		'GlC':'Globular Cluster',
		'OpC':'Open (galactic) Cluster',
		'As*':'Association of Stars',
		'**':'Double or multiple star',
		'EB*':'Eclipsing binary',
		'Al*':'Eclipsing binary of Algol type',
		'bL*':'Eclipsing binary of beta Lyr type',
		'WU*':'Eclipsing binary of W UMa type',
		'EP*':'Star showing eclipses by its planet',
		'SB*':'Spectroscopic binary',
		'CV*':'Cataclysmic Variable Star',
		'DQ*':'Cataclysmic Var. DQ Her type',
		'AM*':'Cataclysmic Var. AM Her type',
		'NL*':'Nova-like Star',
		'No*':'Nova',
		'DN*':'Dwarf Nova',
		'XB*':'X-ray Binary',
		'LXB':'Low Mass X-ray Binary',
		'HXB':'High Mass X-ray Binary',
		'Neb':'Nebula of unknown nature',
		'PoC':'Part of Cloud',
		'PN?':'Possible Planetary Nebula',
		'CGb':'Cometary Globule',
		'EmO':'Emission Object',
		'HH':'Herbig-Haro Object',
		'Cld':'Cloud of unknown nature',
		'GNe':'Galactic Nebula',
		'BNe':'Bright Nebula',
		'DNe':'Dark Nebula',
		'RNe':'Reflection Nebula',
		'HI':'HI (neutral) region',
		'MoC':'Molecular Cloud',
		'HVC':'High-velocity Cloud',
		'HII':'HII (ionized) region',
		'PN':'Planetary Nebula',
		'sh':'HI shell',
		'SR?':'SuperNova Remnant Candidate',
		'SNR':'SuperNova Remnant',
		'*':'Star',
		'*iC':'Star in Cluster',
		'*iN':'Star in Nebula',
		'*iA':'Star in Association',
		'*i*':'Star in double system',
		'V*?':'Star suspected of Variability',
		'Pe*':'Peculiar Star',
		'HB*':'Horizontal Branch Star',
		'Y*O':'Young Stellar Object',
		'Em*':'Emission-line Star',
		'Be*':'Be Star',
		'AB*':'Asymptotic Giant Branch Star',
		'pA*':'Post-AGB Star',
		'WD*':'White Dwarf',
		'ZZ*':'Pulsating White Dwarf',
		'BD*':'Brown Dwarf',
		'C*':'Carbon Star',
		'S*':'S Star',
		'OH*':'Star with envelope of OH/IR type',
		'CH*':'Star with envelope of CH type',
		'pr*':'Pre-main sequence Star (optically detected)',
		'TT*':'T Tau-type Star',
		'WR*':'Wolf-Rayet Star',
		'PM*':'High proper-motion Star',
		'HV*':'High-velocity Star',
		'V*':'Variable Star',
		'Ir*':'Variable Star of irregular type',
		'Or*':'Variable Star of Orion Type',
		'RI*':'Variable Star with rapid variations',
		'Er*':'Eruptive variable Star',
		'Fl*':'Flare Star',
		'FU*':'Variable Star of FU Ori type',
		'RC*':'Variable Star of R CrB type',
		'Ro*':'Rotationally variable Star',
		'a2*':'Variable Star of alpha2 CVn type',
		'El*':'Ellipsoidal variable Star',
		'Psr':'Pulsar',
		'BY*':'Variable of BY Dra type',
		'RS*':'Variable of RS CVn type',
		'Pu*':'Pulsating variable Star',
		'Mi*':'Variable Star of Mira Cet type',
		'RR*':'Variable Star of RR Lyr type',
		'Ce*':'Cepheid variable Star',
		'sr*':'Semi-regular pulsating Star',
		'dS*':'Variable Star of delta Sct type',
		'RV*':'Variable Star of RV Tau type',
		'WV*':'Variable Star of W Vir type',
		'bC*':'Variable Star of beta Cep type',
		'cC*':'Classical Cepheid (delta Cep type)',
		'gD*':'Variable Star of gamma Dor type',
		'SN*':'SuperNova',
		'Sy*':'Symbiotic Star',
		'su*':'Sub-stellar object',
		'Pl?':'Extra-solar Planet Candidate',
		'G':'Galaxy',
		'PoG':'Part of a Galaxy',
		'GiC':'Galaxy in Cluster of Galaxies',
		'GiG':'Galaxy in Group of Galaxies',
		'GiP':'Galaxy in Pair of Galaxies',
		'HzG':'Galaxy with high redshift',
		'ALS':'Absorption Line system',
		'LyA':'Ly alpha Absorption Line system',
		'DLA':'Damped Ly-alpha Absorption Line system',
		'mAL':'metallic Absorption Line system',
		'LLS':'Lyman limit system',
		'BAL':'Broad Absorption Line system',
		'rG':'Radio Galaxy',
		'H2G':'HII Galaxy',
		'LSB':'Low Surface Brightness Galaxy',
		'AG?':'Possible Active Galaxy Nucleus',
		'Q?':'Possible Quasar',
		'EmG':'Emission-line galaxy',
		'SBG':'Starburst Galaxy',
		'BCG':'Blue compact Galaxy',
		'LeI':'Gravitationally Lensed Image',
		'LeG':'Gravitationally Lensed Image of a Galaxy',
		'LeQ':'Gravitationally Lensed Image of a Quasar',
		'AGN':'Active Galaxy Nucleus',
		'LIN':'LINER-type Active Galaxy Nucleus',
		'SyG':'Seyfert Galaxy',
		'Sy1':'Seyfert 1 Galaxy',
		'Sy2':'Seyfert 2 Galaxy',
		'Bla':'Blazar',
		'BLL':'BL Lac - type object',
		'OVV':'Optically Violently Variable object',
		'QSO':'Quasar'
		}

# Custom Exceptions.

class HtmlTagError(Exception) :
	def __init__(self, value) :	self.value = value
	def __str__(self) :	return repr(self.value)

class ConnectionError(Exception) :
	def __init__(self, value) :	self.value=value
	def __str__(self) :	return repr(self.value)

class SimbadError(Exception) :
	def __init__(self,value) : self.value = value
	def __str__(self) : return repr(self.value)	

# If PySIMBAD is run as a standalone tool.

if __name__ == '__main__' :

	if sys.argv.__len__() > 1 :
	    for arg in sys.argv[1:]:
		print "SIMBOT Running..."
		print "Input Object Name : %s" % arg
		link = buildLink(arg)
		print "HTTP Address : %s" % link
		try : s=simbad(link)
		except ConnectionError :
			print "A connection error has occured..."
			sys.exit()
		except SimbadError as e : 
			print "The Simbad services have returned an error : %s" % e.value
			sys.exit()

		print "-"*15
		print "ICRS Coord. : %s %s" % (s.ra(s.icrs()),s.dec(s.icrs()))
		print "FK5 Coord. : %s %s" % (s.ra(s.fk5()),s.dec(s.fk5()))
		print "FK4 Coord. : %s %s" % (s.ra(s.fk4()),s.dec(s.fk4()))
		print "Galactic Coord. : %s %s" % (s.gal().split()[0],s.gal().split()[1])

		print "-"*15
		print "Main Object Type : %s" % s.mainType()
		print "Other Object Types : " 
		for i in s.objectTypes() :
				try : print "\t%s : %s" % (i,objectDictionary[i])
				except : print "\t%s : %s" % (i,"--> Object Type Unknown by PySIMBAD, please report this.")
		print "-"*15
		print "Number of Fluxes : %s" % s.flux_num()
		print "Number of Identifiers : %s" % s.names_num()
		print "Number of Refs. : %s" % s.refs()


		print "-"*15
		print "Colors and Magnitudes : "
		for i in s.flux() : print "\t%s : %s" % (i,s.flux()[i])

		print "-"*15
		print "Names : "
		for i in s.names() : print "\t"+i
