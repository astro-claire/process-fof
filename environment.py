import numpy as np
import h5py 
import numpy as np
import pickle
import os
from concatenateclass import processedFOF
from sys import argv
from boundedness import get_allHalos, set_up_DM
from fof_process import dx_wrap, dist2, set_snap_directories, open_hdf5, get_headerprops

def set_up_baryon_fofs(filename, snapnum,sv):
    """
    Grabs FOF data using processed FOF class
    
    Parameters: 
        fileame (str): path to FOF directory
        snapnum (str or int): 
    
    Returns:
        tuple: centers of bounded objs, radii (from boundedness) of bounded objects

    """
    fof = processedFOF(snapnum,filename,sv, path = "/u/home/c/clairewi/project-snaoz/FOF_project") #call processed fof class 
    fof.chopUnBounded() #remove any unbounded objects 
    if 'recalcRadii' in fof.properties.keys():
        return fof.properties['centers'], fof.properties['recalcRadii']
    elif 'fofradii' in fof.properties.keys():
        return fof.properties['centers'], fof.properties['fofradii']
    else: 
        print("ERROR: No radii associated with this fof! This should not happen.")

def set_up_dm_fofs(snapnum, sv, path = '/u/home/c/clairewi/project-snaoz/FOF_project/DMP-GS-'):
    """
    Grabs halo data corresponding to DM primary FOF with same snapnum and SV
    Parameters: 
        snapnum (int or str)
        sv (str)
        path (str): path of DM primary files WITHOUT the stream velocity indicator
    Return:
        np.ndarray: array of total group mass, np.ndarray: array of dm group mass, array of group position [x,y,z]
    """
    #halo100_indices, halo_positions, startAllDM, endAllDM, dmsnap = set_up_DM(sv, snapnum)
    gofilename = path + str(sv) 
    dmgofile, dmfoffile  = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename) )
    dmsnap, dmfof = open_hdf5(dmgofile, dmfoffile)
    boxSize, _, massDMParticle = get_headerprops(dmsnap)
    print(boxSize)
    mask = dmfof['Group/GroupLenType'][:,1] > 32
    groupMass = np.array(dmfof['Group/GroupMass'][mask])
    groupDMmass = np.array(dmfof['Group/GroupLenType'][:,1][mask])
    groupPos = np.array(dmfof['Group/GroupPos'])[mask]
    return groupMass, groupDMmass * massDMParticle, groupPos

def compare_baryon_dm_fof(baryon_centers, dm_centers, dmmass, boxSize = 1775.): 
    """
    For every baryon FOF object, finds the nearest DM halos (closest one and within 5, 10 ckpc)
    
    Parameters: 
        baryon_centers (arr Nx3): centers of mass of the baryon primary objects
        dm_centers (arr, Mx3): centers of mass of the DM objects
        dmmmass (arr, M): dark matter mass of the DM objects
        boxSize (float): box size in code units

    Returns:

    """
    # for i in range(len(baryon_centers)): 
    N= len(baryon_centers)
    N= 10 # testing mode
    masks10 = np.empty(N,dtype = np.ndarray)
    masks5 = np.empty(N,dtype = np.ndarray)
    closestdm = np.empty(N,dtype = np.ndarray)
    closestdm_dist = np.empty(N,dtype = float)
    closestdm_dmmass = np.empty(N,dtype= float)
    num_within10 =  np.empty(N,dtype = float)
    num_within5 =  np.empty(N,dtype = float)
    for i in range(N):
        com = baryon_centers[i]
        distances = dist2(com[0]-dm_centers[:,0], com[1]-dm_centers[:,1], com[2]-dm_centers[:,2], boxSize)
        mask = np.where(distances<100.), #find all halos within 10 kpc
        masks10[i]= mask
        num_within10[i]= len(mask)
        mask = np.where(distances<25.), # find all halos within 5 kpc
        masks5[i] = mask
        num_within5[i] = len(mask)
        closestdm[i] = np.argmin(distances) # find closest DM halo
        closestdm_dist[i] = distances[closestdm[i]]
        closestdm_dmmass[i] = dmmass[closestdm[i]]
    print(closestdm_dmmass)




if __name__=="__main__":
    """
    Routine if running as a script

    Arguments: 
    gofilename path to directory containing groupordered file + fof table
    # foffilename 
    snapnum (float)
    """
    script, gofilename, snapnum = argv
    baryon_centers, baryon_radii = set_up_baryon_fofs("SP-", str(snapnum), 'Sig2')
    groupMass, groupDMmass, groupPos= set_up_dm_fofs(str(snapnum), 'Sig2')
    compare_baryon_dm_fof(baryon_centers, groupPos, groupDMmass)
