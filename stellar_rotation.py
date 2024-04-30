import h5py 
import numpy as np
from sys import argv
import pickle
from fof_process import get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config,get_gasGroups, get_cosmo_props

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
	objs = calc_stellar_rotation()
