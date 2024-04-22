import h5py 
import numpy as np
from sys import argv
import pickle

def dx_wrap(dx,box):
	#wraps to account for period boundary conditions. This mutates the original entry
	idx = dx > +box/2.0
	dx[idx] -= box
	idx = dx < -box/2.0
	dx[idx] += box 
	return dx

def dist2(dx,dy,dz,box):
	#Calculates distance taking into account periodic boundary conditions
	return dx_wrap(dx,box)**2 + dx_wrap(dy,box)**2 + dx_wrap(dz,box)**2

def set_snap_directories(filename,snapnum,**kwargs):
	"""
	Sets paths HDF5 from directory
	
	Arguments: 
        filename (str): directory of snap-groupordered file
		snapnum (float): Snap file number
		foffilename (str, Optional): path to directory containing fof_subhalo_tab file. 
            Default: same directory as snap-groupordered
	
	Returns:
        (tuple): path to groupordered file, path to fof file
	"""
	gofilename = filename  + "/snap-groupordered_" + str(snapnum).zfill(3)
	if kwargs['foffilename']:
		foffile = str(kwargs['foffilename']) + "/fof_subhalo_tab_"+str(snapnum).zfill(3)
	else: 
		foffile = filename + "/fof_subhalo_tab_"+str(snapnum).zfill(3)
	return gofilename, foffile

def open_hdf5(gofilename, foffilename):
	"""
	Opens hdf5 files with h5py given paths as defined in set_snap_directories
	
	Arguments: 
	    gofilename (str): directory of snap-groupordered file
		goffilename (str): directory of fof table file
	Returns:
        (tuple): snap-groupordered hdf5, fof table hdf5
	"""
	snap = h5py.File(gofilename+'.hdf5')
	fof = h5py.File(foffilename+'.hdf5')
	return snap, fof

def get_headerprops(f):
	""""
	extract header properties from hdf5file
	Parameters:
        f (snap): snap Hdf5 file with header
	Returns:
        (list): boxsize, redshift, massDMParticle
	"""
	return f['Header'].attrs['BoxSize'], f['Header'].attrs['Redshift'],  f['Header'].attrs['MassTable'][1]

def get_cosmo_props(f):
	"""
	Create dictionary of cosmological parameters based on snapfile header. 
	"""
	cos = {}
	G = 6.672e-8
	cos['H0'] = f['Header'].attrs['HubbleParam']* 100 # hubble constant
	cos['a'] =   f['Header'].attrs['Time'] #scale factor @ z of snap
	cos['Om0'] = f['Header'].attrs['Omega0'] #z=0 matter fraction
	cos['Om'] = cos['Om0']* (cos['a'])**(-3.) # matter fraction at z of snap
	cos['OL0'] = f['Header'].attrs['OmegaLambda'] #z=0 DE fraction
	cos['OL'] = cos['OL0'] # DE fraction at z of snap
	cos['H2'] = cos['H0']**2 * (cos['Om']+cos['OL'] ) #Hubble parameter at z of snap
	cos['rhocrit0'] = 3 * cos['H0']**2/(8*np.pi* G) * 1/((3.086e19)**2) # unit conversion to g/cm^3
	cos['rhodm'] = cos['Om']*cos['rhocrit0'] 
	return cos

class subfindGroup():
	def __init__(self,f):
		"""
		initializes the subfind structure
		Parameters: 
            f (HDF5 file): fof table snap file 
		"""
		self.GroupCM = f['Group/GroupCM']
		self.GroupLen = f['Group/GroupLen']
		self.GroupLenType = f['Group/GroupLenType']
		self.GroupPos = f['Group/GroupPos']
		self.Group_R_Crit200 = f['Group/Group_R_Crit200']
		self.GroupVel = f['Group/GroupVel']
		
		
def print_group_properties(f):
	print("Group properties")
	print(f['Group'].keys())
	print("subhalo properties")
	print(f['Subhalo'].keys())
		

def set_subfind_catalog(f):
	return subfindGroup(f)


def set_config(f):
	conf = f['Config']
	prim_type = conf.attrs['FOF_PRIMARY_LINK_TYPES'] 
	if prim_type == 1: 
		prim = "gas"
	elif prim_type ==2: 
		prim = "DM"
	elif prim_type ==16: 
		prim = "stars"
	elif prim_type == 17: 
		prim = "stars+gas"
	else: 
		raise AttributeError("Unknown Primary FOF type. Add support for your FOF!")
	if 'FOF_SECONDARY_LINK_TYPES' in conf.attrs:
		sec_type = conf.attrs['FOF_SECONDARY_LINK_TYPES'] 
		if sec_type == 1: 
			sec = "gas"
		elif sec_type ==16:
			sec = "stars"
		elif sec_type ==17: 
			sec ="stars+gas"
		elif sec_type ==2: 
			sec = "DM"
		else: 
			raise AttributeError("Unknown Secondary FOF type. Add support for your FOF!")
	else: 
		sec = "none"
	return prim, sec

def get_Halos(cat):
	"""
	halos of greater than 300 particles AND nonzero virial radius
	"""
	over300idx, = np.where(np.logical_and(np.greater(cat.GroupLenType[:,1],300),np.not_equal(cat.Group_R_Crit200,0.)))
	return over300idx

def get_gasGroups(cat):
	"""
	Gas clumps with greater than 100 particles
	"""
	halo100_indices= np.where(cat.GroupLenType[:,0] >100)[0]

	return halo100_indices


def get_starGroups(cat):
	""""
	Star clumps with greater than 100 particles
	"""
	halo100_indices= np.where(cat.GroupLenType[:,4] >100)[0]		
	return halo100_indices

def get_gasIDgroups(cat, halo100_indices):
	"""
	given a list of indices, return the beginning and end of the grourp in gas indices
	"""
	startAllGas = []
	endAllGas   = []
	for i in halo100_indices:
		startAllGas += [np.sum(cat.GroupLenType[:i,0])]
		endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]

	return startAllGas, endAllGas

def get_starIDgroups(cat, halo100_indices):
	"""
	given a list of indices, return the beginning and end of the grourp in gas indices
	"""
	startAllStars = []
	endAllStars   = []
	for i in halo100_indices:
		startAllStars += [np.sum(cat.GroupLenType[:i,4])]
		endAllStars   += [startAllStars[-1] + cat.GroupLenType[i,4]]

	return startAllStars, endAllStars

def get_DMIDgroups(cat, halo100_indices):
	"""
	given a list of indices, return the beginning and end of the grourp in gas indices
	"""
	startAllDM = []
	endAllDM   = []
	for i in halo100_indices:
		startAllDM += [np.sum(cat.GroupLenType[:i,1])]
		endAllDM   += [startAllDM[-1] + cat.GroupLenType[i,1]]

	return startAllDM, endAllDM


def get_starIDs(f):
	"""
	Get particle IDs (groupordered snap)
	"""
	allStarIDs = f['PartType4/ParticleIDs']
	allStarMasses = f['PartType4/Masses']
	allStarPositions = f['PartType4/Coordinates']

	return allStarIDs, allStarMasses, allStarPositions

def get_gasIDs(f):
	"""
	Get particle IDs (groupordered snap)
	"""
	allGasIDs = f['PartType0/ParticleIDs']
	allGasMasses = f['PartType0/Masses']
	allGasPositions = f['PartType0/Coordinates']

	return allGasIDs, allGasMasses, allGasPositions


def get_DMIDs(f):
	"""
	Get particle IDs (groupordered snap)
	"""
	allDMIDs = f['PartType1/ParticleIDs']
	allDMPositions = f['PartType1/Coordinates']
	
	return allDMIDs, allDMPositions
	
def get_obj_properties(cat, boxsize, halo100_indices, allStarIDs,allStarMasses, allStarPositions, startAllStars,endAllStars, r200 =True, Gas = False):
	"""
	Find all the star particles in each object within r200 and sums to give the mass
	Parameters: 
		cat
		boxsize
		halo100_indices
		r200 (bool): whether or not to check whether particles lie inside r200 or not
		Gas (bool): function can also be used for gas. If True, adds "gasMass" and "gasIndices" to dict instead of star keys. 
					Make sure to use gas positions, indices, etc as inputs instead. 
			Default: False

	Returns: 
		(dict): dictionary containing star IDs in each group and stellar Mass
	"""
	starIDs_inGroup = np.empty(np.size(cat.GroupLenType),dtype=list)
	mStars_inGroup = np.empty(np.size(cat.GroupLenType),dtype=list)
	mStar_Group = np.zeros(np.size(cat.GroupLenType))
	#print("In testing mode! - not the full set")
	Stars = True
	if Gas:
		Stars = False
	if r200:  #Check whether the stars are located inside r200 or not
		r_200 = cat.Group_R_Crit200
		for i, j in enumerate(halo100_indices): #0-10 for testing mode only!
			# starIDs_inGroup[j] = allStarIDs[startAllStars[i]:endAllStars[i]]
			starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
			goodStars  = np.where(dist2(np.array(starPos_inGroup[:,0]- cat.GroupPos[j][0]),np.array(starPos_inGroup[:,1]- cat.GroupPos[j][1]),np.array(starPos_inGroup[:,2]- cat.GroupPos[j][2]), boxsize)<r_200[j])[0]
			starIDs_inGroup[j] = allStarIDs[startAllStars[i]:endAllStars[i]][goodStars]
			if Stars: #only need mass of each star particle for SFR
				mStars_inGroup[j] = allStarMasses[startAllStars[i]:endAllStars[i]][goodStars] 
			mStar_Group[j] = np.sum(allStarMasses[startAllStars[i]:endAllStars[i]][goodStars])
	else:
		for i, j in enumerate(halo100_indices): #0-10 for testing mode only!
					starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
					#dont include the R200 condition
					#goodStars  = np.where(dist2(np.array(starPos_inGroup[:,0]- cat.GroupPos[j][0]),np.array(starPos_inGroup[:,1]- cat.GroupPos[j][1]),np.array(starPos_inGroup[:,2]- cat.GroupPos[j][2]), boxsize))[0]
					starIDs_inGroup[j] = allStarIDs[startAllStars[i]:endAllStars[i]]
					if Stars: 
						mStars_inGroup[j] = allStarMasses[startAllStars[i]:endAllStars[i]]
					mStar_Group[j] = np.sum(allStarMasses[startAllStars[i]:endAllStars[i]])
	objs = {}
	if Gas: 
		objs['gasIDs'] = np.array(starIDs_inGroup[halo100_indices])
		#objs['gasMasses']= np.array(mStars_inGroup[halo100_indices]) # the masses of each star particle in the group
		objs['gasMass']= np.array(mStar_Group[halo100_indices]) # total stellar mass in the group
	else: 
		objs['starIDs'] = np.array(starIDs_inGroup[halo100_indices])
		objs['starMasses']= np.array(mStars_inGroup[halo100_indices]) # the masses of each star particle in the group
		objs['stellarMass']= np.array(mStar_Group[halo100_indices]) # total stellar mass in the group
	return objs

def get_DM(cat, boxsize, halo100_indices, massDMParticle,allStarIDs, allStarPositions, startAllStars,endAllStars, r200 =True):
	"""
	Find all the DM particles in each object within r200 and sums to give the mass
	SORRY THAT EVERYTHING SAYS STARS
	Parameters: 
		cat
		boxsize
		halo100_indices
		r200 (bool): whether or not to check whether particles lie inside r200 or not
		
	Returns: 
		(dict): dictionary containing DM IDs in each group and DM Mass
	"""
	starIDs_inGroup = np.empty(np.size(cat.GroupLenType),dtype=list)
	# mStars_inGroup = np.empty(np.size(cat.GroupLenType),dtype=list)
	mDM_Group = np.zeros(np.size(cat.GroupLenType))
	#print("In testing mode! - not the full set")
	if r200:  #Check whether the stars are located inside r200 or not
		r_200 = cat.Group_R_Crit200
		for i, j in enumerate(halo100_indices): #0-10 for testing mode only!
			# starIDs_inGroup[j] = allStarIDs[startAllStars[i]:endAllStars[i]]
			starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
			goodStars  = np.where(dist2(np.array(starPos_inGroup[:,0]- cat.GroupPos[j][0]),np.array(starPos_inGroup[:,1]- cat.GroupPos[j][1]),np.array(starPos_inGroup[:,2]- cat.GroupPos[j][2]), boxsize)<r_200[j])[0]
			starIDs_inGroup[j] = allStarIDs[startAllStars[i]:endAllStars[i]][goodStars]
			mDM_Group[j] = len(starIDs_inGroup[j])*massDMParticle
	else:
		for i, j in enumerate(halo100_indices): #0-10 for testing mode only!
			starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
			#dont include the R200 condition
			#goodStars  = np.where(dist2(np.array(starPos_inGroup[:,0]- cat.GroupPos[j][0]),np.array(starPos_inGroup[:,1]- cat.GroupPos[j][1]),np.array(starPos_inGroup[:,2]- cat.GroupPos[j][2]), boxsize))[0]
			starIDs_inGroup[j] = allStarIDs[startAllStars[i]:endAllStars[i]]
			mDM_Group[j] = len(starIDs_inGroup[j])*massDMParticle
	objs = {}
	objs['DMIDs'] = np.array(starIDs_inGroup[halo100_indices])
	objs['DMMass']= np.array(mDM_Group[halo100_indices]) # total stellar mass in the group
	return objs

def find_newStars(objs,newstars, snapnum):
	"""
	Compare IDs in each object to the new stars and see which stars are new
	Parameters: 
		objs (dict): dictionary containing the star IDs and stellar mass of each group
		newstars (dict): dictionary containing the new stars in each snap
		snapnum (float, str, or int): HDF5 snap number
	Returns: 
		dict: dictionary containing the stellar mass, particle IDs, and new stellar mass of each object
	"""
	newmStar_Group = np.zeros(np.size(objs['stellarMass']))
	for i in range(np.size(objs['stellarMass'])): # should be len of the objs list
		zipped = np.c_[objs['starIDs'][i],objs['starMasses'][i]] #Stars particles and their masses in each object
		zipped_IDsMasses = zipped[zipped[:,0].argsort()] #sort by the IDs in the 0th column
		newstars_inSnap = set(newstars[str(snapnum)]) #these were already sorted in the saved file. If you don't save them as sorted they need to get sorted before here
		IDs = zipped_IDsMasses[:,0] #only the IDs
		ind_dict = dict((k,i) for i,k in enumerate(IDs))
		masses = zipped_IDsMasses[:,1] #only the masses
		newIDs = set(IDs).intersection(newstars_inSnap)
		newstaridx = [ind_dict[x] for x in newIDs]
		newmStar_Group[i] = sum(masses[newstaridx])
	return newmStar_Group

def calc_all_stellarprops(filename, snapnum, newstars, group="Stars", SFR = True, r200 = True):
	"""
	Routine to do everything
	Parameters: 
		group (str): which group to use for halos. Stars = at least 100 star particles, gas = at least 100 gas cells, DM = at least 300 DM particles. 
			Default: "Stars"
			Options: "Stars", "Gas", "DM"
		SFR (bool): whether or not to calculate the new stars 
			Default: True  
		r200 (bool): whether or not to calculate material inside r200
	"""
	print('opening files')
	gofilename = str(filename)
	gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
	snap, fof = open_hdf5(gofilename, foffilename)
	boxsize, redshift, massDMParticle = get_headerprops(snap)
	print('redshift is '+str(redshift))
	cat = set_subfind_catalog(fof)
	prim, sec = set_config(fof)
	print("I detected that the FOF has "+str(prim)+" primary and "+str(sec)+" secondary.")
	print("getting fof groups")
	if group == "Stars":
		print("used groups of 100 or more stars")
		halo100_indices=get_starGroups(cat)
	elif group == "Gas":
		print("used groups of 100 or more gas")
		halo100_indices=get_gasGroups(cat)
	elif group == "DM":
		print("used groups of 300 or more DM")
		halo100_indices=get_Halos(cat)
	objs = {}
	if prim == "stars" or prim == "stars+gas" or sec == "stars" or sec == "stars+gas":
		print("have stars, calculating the stellar properties")
		allStarIDs, allStarMasses,allStarPositions = get_starIDs(snap)
		startAllStars, endAllStars = get_starIDgroups(cat,halo100_indices)
		print("calculating stellar mass")
		objs = get_obj_properties(cat, boxsize ,halo100_indices, allStarIDs,allStarMasses, allStarPositions, startAllStars,endAllStars, r200=r200)
		print("determining new star formation")
		if SFR: 
			objs['new_mStar'] = find_newStars(objs,newstars, snapnum)
		else: 
			print("Warning: SFR is off. Not calculating new stars")
	else: 
		print("Warning: no stars found. I will not calculate SFRs")
	if prim == "gas" or prim == "stars+gas" or sec =="gas" or sec =="stars+gas":
		print("have gas, calculating the gas properties")
		allGasIDs, allGasMasses, allGasPositions = get_gasIDs(snap)
		startAllGas, endAllGas = get_gasIDgroups(cat,halo100_indices)
		print("calculating gas mass")
		gasobjs = get_obj_properties(cat, boxsize ,halo100_indices, allGasIDs,allGasMasses, allGasPositions, startAllGas,endAllGas, Gas =True, r200=r200)
		objs['gasIDs'] = gasobjs['gasIDs']
		objs['gasMass'] = gasobjs['gasMass']
	else:
		print("Warning: no gas found. I will not calculate gas properties")
	if prim =="DM" or sec =="DM":
		print("have DM, calculating the DM properties")
		allDMIDs, allDMPositions = get_DMIDs(snap)
		startAllDM, endAllDM = get_DMIDgroups(cat,halo100_indices)
		dmobjs = get_DM(cat, boxsize, halo100_indices, massDMParticle,allDMIDs, allDMPositions, startAllDM,endAllDM, r200 =r200)
		objs['DMMass'] = dmobjs['DMMass']
		objs['DMIDs'] = dmobjs['DMIDs']
		objs['r200']= np.array(cat.Group_R_Crit200)[halo100_indices]
	else: 
		print("Warning: No DM found. I will not calculate any halo properties")
	objs['prim'] = prim
	objs['sec'] = sec
	print("done")
	return objs

if __name__=="__main__":
	"""
	Routine if running as a script

	Arguments: 
		gofilename path to directory containing groupordered file + fof table
		# foffilename 
		snapnum (float)
	"""
	script, gofilename, snapnum = argv
	with open("/home/x-cwilliams/FOF_calculations/newstars_Sig2_25Mpc.dat",'rb') as f:
		newstars = pickle.load(f,encoding = "latin1")
	objs = calc_all_stellarprops(str(gofilename), snapnum, newstars, group="Gas", SFR=True, r200 =False)
	print(objs['gasMass'][0:10]*10.**10/0.71)
	# gofilename = str(gofilename)
	# #foffilename = str(gofilename)
	# gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
	# snap, fof = open_hdf5(gofilename, foffilename)
	# boxsize, redshift, massDMParticle = get_headerprops(snap)
	# cat = set_subfind_catalog(fof)
	# prim, sec = set_config(fof)
	# #print(cat.Group_R_Crit200[0])
	# halo100_indices=get_starGroups(cat)
	# # allStarIDs, allStarMasses,allStarPositions = get_starIDs(snap, prim, sec)
	# with open("/u/home/c/clairewi/project-snaoz/SF_MolSig2/newstars_Sig2_25Mpc.dat",'rb') as f:
	# 	newstars = pickle.load(f,encoding = "latin1")
	# allStarIDs, allStarMasses,allStarPositions = get_starIDs(snap, prim, sec)
	# startAllStars, endAllStars = get_starIDgroups(cat,halo100_indices)
	# objs = get_obj_properties(cat, boxsize ,halo100_indices, allStarIDs,allStarMasses, allStarPositions, startAllStars,endAllStars)
	# #print(objs['stellarMass'][0:10]*1e10)
	# new_ones = find_newStars(objs,newstars, snapnum)
	# print(new_ones[0:10]*1e10)


	