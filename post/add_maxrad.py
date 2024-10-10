import numpy as np
import numpy as np
import pickle
from concatenateclass import processedFOF
from sys import argv
import sys
sys.path.append('../')
from boundedness import get_allHalos, set_up_DM, calc_max_radius
from fof_process import dx_wrap, dist2, set_snap_directories, open_hdf5, get_headerprops
from environment import set_up_baryon_fofs

def get_maxradii(allStarPositions,startAllStars,endAllStars,baryon_centers,boxSize):
    """
    iterate through the bounded and virialized objects and add their max radius parameter
    """
    starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
    for center in baryon_centers:
        maxradius = calc_max_radius(starPos_inGroup,baryon_centers,boxSize)

def wrapper(directory, sv,snapnum, save = True, boxSize = 1775., path = '/u/home/c/clairewi/project-snaoz/FOF_project/'):   
    """
    Wrapper function. Currently there are some unused parameters nad the 

    Parameters: 
        directory (str): name of directory where baryon output located (ie. "SP-", "SGP-")
        sv (str): Value of stream velocity - options "Sig0" or "Sig2" 
        snapnum (str or int): hdf5 snap output. 
        save (bool): whether or not to save the output at path + directory + sv

    """ 
    baryon_centers, _ = set_up_baryon_fofs(str(directory), str(snapnum), str(sv))