import numpy as np
import numpy as np
import pickle
from concatenateclass import processedFOF
from sys import argv
import sys
sys.path.append('../')
from boundedness import calc_max_radius, get_GroupPos
from fof_process import set_subfind_catalog, get_starIDs,get_starIDgroups,get_starGroups, set_snap_directories, open_hdf5, get_headerprops

def get_maxradii(allStarPositions,startAllStars,endAllStars,baryon_centers,boxSize):
    """
    iterate through the bounded and virialized objects and add their max radius parameter
    """
    N = len(baryon_centers)
    maxradii = np.empty(N,dtype = np.ndarray)
    for i in range(N):
        com = baryon_centers[i]
        starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
        maxradii[i] = calc_max_radius(starPos_inGroup,com,boxSize)
    return maxradii


def wrapper(directory, sv,snapnum, save = True, boxSize = 1775., path = '/u/home/c/clairewi/project-snaoz/FOF_project/'):   
    """
    Wrapper function. Currently there are some unused parameters nad the 

    Parameters: 
        directory (str): name of directory where baryon output located (ie. "SP-", "SGP-")
        sv (str): Value of stream velocity - options "Sig0" or "Sig2" 
        snapnum (str or int): hdf5 snap output. 
        save (bool): whether or not to save the output at path + directory + sv

    """ 
    gofilename = str(path)+str(directory)+str(sv)
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, redshift, massDMParticle = get_headerprops(snap)
    cat = set_subfind_catalog(fof)
    halo100_indices=get_starGroups(cat)
    _,_, allStarPositions, _= get_starIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    radii = get_maxradii(allStarPositions,startAllStars,endAllStars,halo100_pos,boxSize)
    if save ==True: 
        objs = {}
        objs['maxradii'] = np.array(radii)
        print("Saving output!")
        with open(gofilename+"/maxradii_"+str(snapnum)+"_V1.dat",'wb') as f:
            pickle.dump(objs, f)