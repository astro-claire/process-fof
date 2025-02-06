import numpy as np
import numpy as np
import pickle
from sys import argv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.concatenateclass import processedFOF
from modules.boundedness import get_allHalos, set_up_DM
from modules.fof_process import dx_wrap, dist2, set_snap_directories, open_hdf5, get_headerprops

def set_up_baryon_fofs(filename, snapnum,sv):
    """
    Grabs FOF data using processed FOF class
    
    Parameters: 
        fileame (str): path to FOF directory
        snapnum (str or int): 
    
    Returns:
        tuple: centers of bounded objs, radii (from boundedness) of bounded objects

    """
    fof = processedFOF(snapnum,filename,sv, path = "/u/home/c/clairewi/project-snaoz/FOF_project",bounded_path = "bounded3") #call processed fof class 
    fof.chopUnBounded() #remove any unbounded objects 
    if 'recalcRadii' in fof.properties.keys():
        print("using the recalculated radii")
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
        np.ndarray: array of total group mass, np.ndarray: array of dm group mass, array of group position [x,y,z], np.ndarray: array of group radii
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
    groupRadii = np.array(dmfof['Group/Group_R_Crit200'][mask])
    return groupMass, groupDMmass * massDMParticle, groupPos, groupRadii

def compare_baryon_dm_fof(baryon_centers, dm_centers, dmmass, dmradii, boxSize = 1775.): 
    """
    For every baryon FOF object, finds the nearest DM halos (closest one and within 5, 10 ckpc)
    
    Parameters: 
        baryon_centers (arr Nx3): centers of mass of the baryon primary objects
        dm_centers (arr, Mx3): centers of mass of the DM objects
        dmmmass (arr, M): dark matter mass of the DM objects
        boxSize (float): box size in code units

    Returns:
        arr, dtype = int: indices of closest DM object;
        arr, dtype = float : distances to closest DM object in code units; 
        arr, dtype = float: masses of closest DM object in code mass; 
        arr, dtype = bool: whether or not the object is within R200 of the closest DM; 
        arr, dtype = int: number of DM object within 10 code units;
        arr, dtype = int: number of DM object within 5 code units;
    """
    N= len(baryon_centers)
    masks10 = np.empty(N,dtype = np.ndarray)
    masks5 = np.empty(N,dtype = np.ndarray)
    closestdm = np.empty(N,dtype = np.ndarray)
    closestdm_dist = np.empty(N,dtype = float)
    closestdm_dmmass = np.empty(N,dtype= float)
    closestdm_inr200 = np.empty(N, dtype=bool)
    num_within10 =  np.empty(N,dtype = int)
    num_within5 =  np.empty(N,dtype = int)
    for i in range(N):
        com = baryon_centers[i]
        distances = dist2(com[0]-dm_centers[:,0], com[1]-dm_centers[:,1], com[2]-dm_centers[:,2], boxSize)
        mask = np.where(distances<100.), #find all halos within 10 kpc
        masks10[i]= mask
        num_within10[i]= len(mask[0][0])
        mask = np.where(distances<25.), # find all halos within 5 kpc
        masks5[i] = mask
        num_within5[i] = len(mask[0][0])
        closestdm[i] = np.argmin(distances) # find closest DM halo
        closestdm_dist[i] = np.sqrt(distances[closestdm[i]])
        closestdm_inr200[i] = np.greater(dmradii[closestdm[i]],closestdm_dist[i])
        closestdm_dmmass[i] = dmmass[closestdm[i]]
    return closestdm, closestdm_dist, closestdm_dmmass, closestdm_inr200, num_within10, num_within5

def compare_baryon_env(baryon_centers, boxSize = 1775.): 
    """
    For every baryon FOF object, finds the nearest other baryon objects (closest one and within 5, 10 ckpc)
    
    Parameters: 
        baryon_centers (arr Nx3): centers of mass of the baryon primary objects
        boxSize (float): box size in code units

    Returns:
        arr, dtype = int: indices of closest baryon object;
        arr, dtype = float : distances to closest baryon object in code units; 
        arr, dtype = int: number of baryon object within 10 code units;
        arr, dtype = int: number of baryon object within 5 code units;
    """
    N= len(baryon_centers)
    masks10 = np.empty(N,dtype = np.ndarray)
    masks5 = np.empty(N,dtype = np.ndarray)
    closest = np.empty(N,dtype = np.ndarray)
    closest_dist = np.empty(N,dtype = float)
    num_within10 =  np.empty(N,dtype = int)
    num_within5 =  np.empty(N,dtype = int)
    for i in range(N):
        com = baryon_centers[i]
        #remove the current object from the list of centers
        ex_centers = baryon_centers[np.isin(range(len(baryon_centers)),i,invert= True)] 
        distances = dist2(com[0]-ex_centers[:,0], com[1]-ex_centers[:,1], com[2]-ex_centers[:,2], boxSize)
        mask = np.where(distances<100.), #find all structures within 10 kpc
        masks10[i]= mask
        num_within10[i]= len(mask[0][0])
        mask = np.where(distances<25.), # find all structures within 5 kpc
        masks5[i] = mask
        num_within5[i] = len(mask[0][0])
        closest[i] = np.argmin(distances) # find closest structure
        closest_dist[i] = np.sqrt(distances[closest[i]])
    return closest, closest_dist, num_within10, num_within5

def dict_calculate(baryon_centers, dm_centers, dmmass, dmradii, boxSize = 1775.):
    """
    Creates dictionary of 
    
    Parameters: 
        baryon_centers (arr Nx3): centers of mass of the baryon primary objects
        dm_centers (arr, Mx3): centers of mass of the DM objects
        dmmmass (arr, M): dark matter mass of the DM objects
        boxSize (float): box size in code units
    
    Returns: 
        dict: dictionary containing all the output parameters of compare_baryon_env, compare_baryon_dm_fof
    """
    objs = {}
    print("Calculating baryon environment")
    objs['closestb'], objs['closestb_dist'], objs['num_within10b'], objs['num_within5b'] = compare_baryon_env(baryon_centers, boxSize = boxSize)
    print("Calculating DM environment")
    objs['closestdm'], objs['closestdm_dist'], objs['closestdm_dmmass'], objs['closestdm_inr200'], objs['num_within10dm'], objs['num_within5dm'] = compare_baryon_dm_fof(baryon_centers, dm_centers, dmmass, dmradii, boxSize = boxSize)
    return objs

def wrapper(directory, sv,snapnum, save = True, boxSize = 1775., path = '/u/home/c/clairewi/project-snaoz/FOF_project/'):   
    """
    Wrapper function. Currently there are some unused parameters and the 

    Parameters: 
        directory (str): name of directory where baryon output located (ie. "SP-", "SGP-")
        sv (str): Value of stream velocity - options "Sig0" or "Sig2" 
        snapnum (str or int): hdf5 snap output. 
        save (bool): whether or not to save the output at path + directory + sv

    """ 
    baryon_centers, _ = set_up_baryon_fofs(str(directory), str(snapnum), str(sv))
    _, groupDMmass, groupPos, groupRadii= set_up_dm_fofs(str(snapnum), str(sv))
    objs = dict_calculate(baryon_centers, groupPos, groupDMmass, groupRadii, boxSize = boxSize)
    if save ==True: 
        print("Saving output!")
        with open(str(path)+str(directory)+str(sv)+"/environment_"+str(snapnum)+"_V2.dat",'wb') as f:
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
    wrapper(directory, sv,snapnum)
    # baryon_centers, baryon_radii = set_up_baryon_fofs("SP-", str(snapnum), 'Sig2')
    # groupMass, groupDMmass, groupPos, groupRadii= set_up_dm_fofs(str(snapnum), 'Sig2')
    # compare_baryon_env(baryon_centers)
    # compare_baryon_dm_fof(baryon_centers, groupPos, groupDMmass, groupRadii)
