import numpy as np
import numpy as np
import pickle
from sys import argv
import sys
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.boundedness import calc_max_radius, get_GroupPos, get_starIDs
from modules.fof_process import set_subfind_catalog,get_cosmo_props,get_starIDgroups,get_starGroups, set_snap_directories, open_hdf5, get_headerprops
import config.configuration as config

constants = config.load_constants()
UnitLength_in_cm = constants['UnitLength_in_cm']
# UnitLength_in_cm = 3.085678e21 
inputdir = config.set_directories()[0]

def get_maxradii(allStarPositions,startAllStars,endAllStars,baryon_centers,boxSize,cosmo):
    """
    iterate through the bounded and virialized objects and add their max radius parameter

    Parameters:
        allStarPositions (np.ndarray): positions of all the star particles in the simulation
        startAllStars (np.ndarray): Starting indices for star particles in each halo.
        endAllStars (np.ndarray): Ending indices for star particles in each halo.
        baryon_centers (np.ndarray): Center of masses of baryon primary groups
        boxSize (float): size of box in code units for periodic boundary conditions correcitons
        cosmo (dict): dictionary of cosmological parameters. Here we only need H0 (hubble constant) and a (scale factor)
    
    Returns: 
        np.ndarray: radius of maximum star particle in each object (in physical units) 
        
    """ 
    # hubbleparam = cosmo['H0']/100.
    #atime = cosmo ['a']
    hubbleparam = 0.71
    atime = cosmo['afromsnap']
    baryon_centers = baryon_centers *atime / hubbleparam 
    N = len(baryon_centers)
    print(str(N)+" objects")
    maxradii = np.empty(N,dtype = float)
    boxSize = boxSize * atime/hubbleparam
    for i in range(N):
        com = baryon_centers[i]
        starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
        starPos_inGroup= np.array(starPos_inGroup) *atime / hubbleparam
        maxradii[i] = calc_max_radius(starPos_inGroup,com,boxSize)/UnitLength_in_cm
    print(maxradii)
    return maxradii

def wrapper(directory, sv,snapnum, save = True, boxSize = 1775., path = str(inputdir)):   
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
    boxSize, redshift, _ = get_headerprops(snap)
    cat = set_subfind_catalog(fof)
    halo100_indices=get_starGroups(cat)
    _,_, allStarPositions, _= get_starIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    cos = get_cosmo_props(snap)
    cos['afromsnap'] = 1./(1.+redshift)
    print("calculating the radii")
    radii = get_maxradii(allStarPositions,startAllStars,endAllStars,halo100_pos,boxSize,cos)
    print("Got the radii")
    if save ==True: 
        objs = {}
        objs['maxradii'] = np.array(radii)
        print("Saving output!")
        with open(str(path)+str(directory)+str(sv)+"/maxradii_"+str(snapnum)+"_V3.dat",'wb') as f:
            pickle.dump(objs, f)

if __name__=="__main__":
    """
    Routine if running as a script

    Arguments: 
    gofilename path to directory containing groupordered file + fof table
    # foffilename 
    snapnum (float)
    """
    script, directory, sv, snapnum = argv
    wrapper(directory, sv,snapnum, save =True)
    # baryon_centers, baryon_radii = set_up_baryon_fofs("SP-", str(snapnum), 'Sig2')
    # groupMass, groupDMmass, groupPos, groupRadii= set_up_dm_fofs(str(snapnum), 'Sig2')
    # compare_baryon_env(baryon_centers)
    # compare_baryon_dm_fof(baryon_centers, groupPos, groupDMmass, groupRadii)
