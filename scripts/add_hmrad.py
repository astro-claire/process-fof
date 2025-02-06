import numpy as np
import numpy as np
import pickle
from sys import argv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.boundedness import get_GroupPos, get_starIDs
from modules.fof_process import dist2, set_subfind_catalog,get_cosmo_props,get_starIDgroups,get_starGroups, set_snap_directories, open_hdf5, get_headerprops

UnitLength_in_cm = 3.085678e21 

def get_hmradii(allStarPositions,allStarMasses,startAllStars,endAllStars,baryon_centers,boxSize,cosmo):
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
        np.ndarray: radius of half mass star particle in each object (in physical units) 
        
    """ 
    hubbleparam = cosmo['H0']/100.
    atime = cosmo ['a']
    baryon_centers = baryon_centers *atime / hubbleparam 
    N = len(baryon_centers)
    print(str(N)+" objects")
    hmradii = np.empty(N,dtype = np.ndarray)
    boxSize = boxSize * atime/hubbleparam
    for i in range(N):
        com = baryon_centers[i]
        starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
        starMass_inGroup = allStarMasses[startAllStars[i]:endAllStars[i]]
        starPos_inGroup= np.array(starPos_inGroup) *atime / hubbleparam
        hmradii[i] = calc_half_mass_radius(starPos_inGroup,starMass_inGroup,com,boxSize)/UnitLength_in_cm
    return hmradii

def calc_half_mass_radius(starPos_inGroup, starMass_inGroup, groupPos, boxSize):
    """
    Calculate the minimum radius that encloses half the stellar mass of a star cluster. Used GPT.

    Parameters:
        starPos_inGroup (np.ndarray): The positions of stars in the group (shape: N x 3).
        starMass_inGroup (np.ndarray): The masses of stars in the group (shape: N).
        groupPos (np.ndarray): The position of the group's center (shape: 3).
        boxSize (float): The size of the simulation box.

    Returns:
        float: The minimum radius enclosing half the stellar mass of the star cluster, in cm.
    """
    # Calculate distances of stars from the group center
    distances = dist2(starPos_inGroup[:,0] - groupPos[0],
                      starPos_inGroup[:,1] - groupPos[1],
                      starPos_inGroup[:,2] - groupPos[2], boxSize)
    distances = np.sqrt(distances)  # Convert squared distances to actual distances
    
    # Sort distances and masses by distance
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_masses = starMass_inGroup[sorted_indices]
    
    # Calculate cumulative mass and find the radius enclosing half the total mass
    cumulative_mass = np.cumsum(sorted_masses)
    half_mass = cumulative_mass[-1] / 2
    
    # Find the index where cumulative mass reaches or exceeds half of total mass
    half_mass_index = np.searchsorted(cumulative_mass, half_mass)
    half_mass_radius = sorted_distances[half_mass_index]
    
    return UnitLength_in_cm * half_mass_radius




def wrapper(directory, sv,snapnum, save = True, boxSize = 1775., path = '/u/home/c/clairewi/project-snaoz/FOF_project/'):   
    """
    Wrapper function. Currently there are some unused parameters

    Parameters: 
        directory (str): name of directory where baryon output located (ie. "SP-", "SGP-")
        sv (str): Value of stream velocity - options "Sig0" or "Sig2" 
        snapnum (str or int): hdf5 snap output. 
        save (bool): whether or not to save the output at path + directory + sv

    """ 
    gofilename = str(path)+str(directory)+str(sv)
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, _, _ = get_headerprops(snap)
    cat = set_subfind_catalog(fof)
    halo100_indices=get_starGroups(cat)
    _,allStarMasses, allStarPositions, _ = get_starIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    cos = get_cosmo_props(snap)
    print("calculating the radii")
    radii = get_hmradii(allStarPositions,allStarMasses,startAllStars,endAllStars,halo100_pos,boxSize,cos)
    print("Got the radii")
    if save ==True: 
        objs = {}
        objs['hmradii'] = np.array(radii)
        print("Saving output!")
        with open(str(path)+str(directory)+str(sv)+"/hmradii_"+str(snapnum)+"_V2.dat",'wb') as f:
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