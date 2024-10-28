import numpy as np
import numpy as np
import pickle
from concatenateclass import processedFOF
from sys import argv
from boundedness import get_starIDs, get_DMIDs
from fof_process import dx_wrap, dist2, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, get_Halos

def set_up_DM_fofs(filename, snapnum,sv):
    """
    Grabs FOF data using processed FOF class
    
    Parameters: 
        fileame (str): path to FOF directory
        snapnum (str or int): 
    
    Returns:
        tuple: centers of bounded objs, radii (from boundedness) of bounded objects

    """
    fof = processedFOF(snapnum,filename,sv, path = "/u/home/c/clairewi/project-snaoz/FOF_project") #call processed fof class 
    return fof['DMIDs'], fof['starIDs'], fof['starMasses']

def get_fof_particles(filename, snapnum, sv):
    """
    Gets the particles from the snapshot
    """
    gofilename = str(filename)
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxsize, redshift, massDMParticle = get_headerprops(snap)
    atime = 1./(1.+redshift)
    print('redshift is '+str(redshift))
    cat = set_subfind_catalog(fof)
    print("used groups of 300 or more DM")
    halo100_indices=get_Halos(cat)
    allStarIDs, allStarMasses, allStarPositions, allStarVelocities = get_starIDs(snap)
    allDMIDs, allDMPositions, allDMVelocities = get_DMIDs(snap)
    groupPos =  cat.GroupPos[halo100_indices]
    groupVel = cat.GroupVel[halo100_indices]
    DMIDs_inGroup, starIDs_inGroup, starMasses_inGroup = set_up_DM_fofs(str(filename),snapnum, sv)



