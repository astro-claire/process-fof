"""
For a Star or Gas primary object, associate dark matter particles with those particles. 

"""
import h5py 
import numpy as np
from sys import argv
import pickle
import fof_process
	prim, sec = set_config(fof)
from fof_process import get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config

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



def files_and_groups(filename, snapnum, newstars, group="Stars")
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
		print("No need to add DM to a DM primary! Exiting now.")
		break
	objs = {}
	if prim == "stars" or prim == "stars+gas" or sec == "stars" or sec == "stars+gas":
		print("have stars, getting the star groups")
		allStarIDs, allStarMasses,allStarPositions = get_starIDs(snap)
		startAllStars, endAllStars = get_starIDgroups(cat,halo100_indices)
		print(" ")
	else: 
		print("Warning: no stars found. I will not be including them!")
	if prim == "gas" or prim == "stars+gas" or sec =="gas" or sec =="stars+gas":
		print("have gas, getting the gas groups")
		allGasIDs, allGasMasses, allGasPositions = get_gasIDs(snap)
		startAllGas, endAllGas = get_gasIDgroups(cat,halo100_indices)
	else:
		print("Warning: no gas found. I will not be including them!")
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
    
	#objs = calc_all_stellarprops(str(gofilename), snapnum, newstars, group="Gas", SFR=True, r200 =False)
	print(objs['gasMass'][0:10]*10.**10/0.71)
