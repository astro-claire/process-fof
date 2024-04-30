import h5py 
import numpy as np
from sys import argv
import pickle
from fof_process import get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config,get_gasGroups, get_cosmo_props,get_starIDgroups

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

def get_GroupRadii(cat, halo100_indices):
    """
    Return Group COM
    """
    return cat.Group_R_Crit200[halo100_indices]

def get_GroupVel(cat, halo100_indices):
    """
    Return Group COM
    """
    return cat.GroupVel[halo100_indices]


def get_starIDs(f):
    """
    Get particle IDs (groupordered snap)
    """
    allStarIDs = f['PartType4/ParticleIDs']
    allStarPositions = f['PartType4/Coordinates']
    allStarVelocities = f['PartType4/Velocities']
    return allStarIDs, allStarPositions, allStarVelocities


def calc_stellar_rotation():
	"""
	Calculate rotation curve of stellar component 
	"""
	pass

def iterate_galaxies(halo100_indices):
	"""
	iterate all the galaxies in the FOF and find their rotation curves
	"""
	objs = {}
	rotation = np.array(len(halo100_indices))
	radii = np.array(len(halo100_indices))
	for i in halo100_indices:
            
		stellar_rotation_curve, rotation_radii = calc_stellar_rotation()
		rotation[i] = stellar_rotation_curve
		radii[i] = rotation_radii
	objs['rot_curves'] = rotation
	objs['rot_radii'] = radii
	return objs

#Alternatively, let's just do this
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


def add_rotation_curves(filename, snapnum, group = "Stars"):
    """
    wrapper function
    """
    print('opening files')
    gofilename = str(filename)
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, redshift, massDMParticle = get_headerprops(snap)
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
        print("Not supported!")
    print("Loading star particles")
    allStarIDs, allStarMasses,allStarPositions = get_starIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    objs = iterate_galaxies(halo100_indices)
    objs =get_obj_properties(cat, boxSize, halo100_indices, allStarIDs,allStarMasses, allStarPositions, startAllStars,endAllStars, r200 =True, Gas = False)
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
	# with open("/home/x-cwilliams/FOF_calculations/newstars_Sig2_25Mpc.dat",'rb') as f:
	# 	newstars = pickle.load(f,encoding = "latin1")
	objs = add_rotation_curves(gofilename, snapnum)
