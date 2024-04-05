"""
For a Star or Gas primary object, associate dark matter particles with those particles. 

"""
import h5py 
import numpy as np
from sys import argv
import pickle
import fof_process
from fof_process import get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config,get_gasGroups, get_gasIDs, get_starIDs,get_starIDgroups, get_gasIDgroups

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

def get_GroupPos(cat, halo100_indices):
    """
    Return Group COM"""
    return np.array(cat.GroupPos)[halo100_indices]

def get_DMIDs(f):
    """
    Get particle IDs (groupordered snap)
    """
    allDMIDs = f['PartType1/ParticleIDs']
    allDMPositions = f['PartType1/Coordinates']
    allDMVelocities = f['PartType1/Velocities']
    return allDMIDs, allDMPositions, allDMVelocities

def find_DM_shells(f,pDM,cm):
    """
    This function will calculate the amount of DM inside spherical shells around a position x, y, z
    Parameters: 
        f (h5py): snapshot
        pDM (list): list of 3D positions of each DM particle in snapshot
        cm (array or list): 3 element array containing x, y, z position from which to calculate the shells. 
    """
    
    tempPosDM = dx_wrap(pDM-cm,boxSize)			
    tempAxis = 20. #draw a 20 kpc sphere around the object
    nearidx, = np.where(dist2(pDM[:,0]-cm[0],pDM[:,1]-cm[1],pDM[:,2]-cm[2],boxSize)<=tempAxis**2)
    shell_width = 0.5 # steps of 0.5 kpc
    if len(nearidx)==0: 
        mDM_shells = np.zeros(int(tempAxis/shell_width))


def files_and_groups(filename, snapnum, newstars, group="Stars"):
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
    print("Now getting all DM particles and their 6D vectors")
    allDMIDs, allDMPositions, allDMVelocities =  get_DMIDs(snap)
    print("Getting group COM!")
    halo100_pos = get_GroupPos(cat, halo100_indices)
    print(halo100_pos[0])
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
    objs = files_and_groups(gofilename, snapnum, newstars, group="Stars")
    print("Done!")
