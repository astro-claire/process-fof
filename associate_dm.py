"""
For a Star or Gas primary object, associate dark matter particles with those particles. 

"""
import h5py 
import numpy as np
from sys import argv
import pickle
import fof_process
from fof_process import get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config,get_gasGroups, get_cosmo_props

UnitLength_in_cm = 3.085678e21 # code length unit in cm/h
UnitMass_in_g = 1.989e43       # code length unit in g/h


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
    Return Group COM
    """
    return cat.GroupPos[halo100_indices]

def get_GroupRadii(cat, halo100_indices):
    """
    Return Group COM
    """
    return cat.Group_R_Crit200[halo100_indices]

def get_DMIDs(f):
    """
    Get particle IDs (groupordered snap)
    """
    allDMIDs = f['PartType1/ParticleIDs']
    allDMPositions = f['PartType1/Coordinates']
    allDMVelocities = f['PartType1/Velocities']
    return allDMIDs, allDMPositions, allDMVelocities

def find_DM_shells(pDM,cm, massDMParticle,rgroup, rhodm, boxSize= 1775.):
    """
    This function will calculate the amount of DM inside spherical shells around a position x, y, z
    Parameters: 
        f (h5py): snapshot
        pDM (array): array of 3D positions of each DM particle in snapshot
        cm (array or list): 3 element array containing x, y, z position from which to calculate the shells.
        rgroup  (float): radius of group (will search within 20x this radius unless 0 is given)
    """
    
    #tempPosDM = dx_wrap(pDM-cm,boxSize)	
    tempAxis = 10* rgroup #search within the radius of the group
    if tempAxis ==0.:
        tempAxis = 10. #search within 10 kpc if no rgroup given
    distances = dist2(pDM[:,0]-cm[0],pDM[:,1]-cm[1],pDM[:,2]-cm[2],boxSize)
    nearidx = np.where(distances<=tempAxis**2)[0]
    shell_width = tempAxis/40. # break into 20 shells 
    if len(nearidx)==0: #if no DM 
        print("NoDM!")
        mDM_shells = np.zeros(40)
        shells = []
    else:
        mDM_shells = []
        shells = []
        shell = shell_width
        tempPosDM = distances[nearidx] #This was changed from shrinker
        while shell <= tempAxis: #calculate enclosed mass inside sphere 
            DM_encl = np.where(tempPosDM<=shell**2)[0]
            #The line below could eventually be used for an ellipsoidal search --note some things about tempPos DM have been changed. So would need to update
            #DM_encl = tempPosDM[:,0]**2/ratios[0]**2 + tempPosDM[:,1]**2/ratios[1]**2 + tempPosDM[:,2]**2 <= shell**2
            mask = np.ones(tempPosDM.shape, dtype='bool') #let's mask out all the particles that were in the inner shell 
            mask[DM_encl] = False #Remove the used particles
            tempPosDM = tempPosDM[mask] #next shell we'll only search the unused DM particles
            mDM_encl =  len(DM_encl)*massDMParticle  #number of DM particles times particle mass
            mDM_shells.append(mDM_encl)
            shells.append(shell)
            shell = shell+ shell_width
        edge_density = mDM_shells[-1]*UnitMass_in_g/(shells[-1]**(-3)*UnitLength_in_cm**3)
        if edge_density<200*rhodm:
            #ADD CODE TO ADD SHELLS
            print("Overdensity continues to further radii")
        else:
            pass
    return np.array(shells),np.array(mDM_shells)

def get_all_DM(allDMPositions,halo100_pos,massDMParticle, radii,rhodm, boxSize):
    """
    Calculates the dm shells for all the objects 
    Parameters: 
        allDMPositions (h5py.dataset): positions of all DM particles
        halo100_pos (numpy array): positions (COM) of all groups 
        massDMParticle (float): code mass of DM 
        radii (numpy array): radii of all groups to calculate 
    """
    all_shells = []
    mDMs = []
    allDMPositions = np.array(allDMPositions)
    for i in range(len(halo100_pos)):
        shells, mDM = find_DM_shells(allDMPositions,halo100_pos[i],massDMParticle, radii[i],rhodm,boxSize = boxSize)
        all_shells.append(shells)
        mDMs.append(mDM)
    return all_shells, mDMs

def files_and_groups(filename, snapnum, group="Stars"):
    print('opening files')
    gofilename = str(filename)
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, redshift, massDMParticle = get_headerprops(snap)
    cosmo = get_cosmo_props(snap)
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
    print("Now getting all DM particles and their 6D vectors")
    allDMIDs, allDMPositions, allDMVelocities =  get_DMIDs(snap)
    # TESTING MODE ONLY: Uncomment next line
    #halo100_indices = halo100_indices[0:2]
    print(str(len(halo100_indices))+' objects')
    print("Getting group COM!")
    halo100_pos = get_GroupPos(cat, halo100_indices)
    halo100_rad = get_GroupRadii(cat, halo100_indices)
    print("dividing into shells and finding the DM")
    #shells, mDM = find_DM_shells(allDMPositions,halo100_pos[1],massDMParticle, halo100_rad[1],boxSize = boxSize)
    #print(shells)
    #print(mDM)
    all_shells, mDMs = get_all_DM(allDMPositions,halo100_pos,massDMParticle, halo100_rad,cosmo['rhodm'], boxSize)
    objs['shells']=np.array(all_shells)
    objs['mDM_shells']=np.array(mDMs)
    objs['prim'] = prim
    objs['sec'] = sec
    print("done")
    #with open(gofilename+"/testdm.dat",'wb') as f:
    #    pickle.dump(objs, f)
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
    #     newstars = pickle.load(f,encoding = "latin1")
    #with open("/u/home/c/clairewi/project-snaoz/SF_MolSig2/newstars_Sig2_25Mpc.dat",'rb') as f:
    #    newstars = pickle.load(f,encoding = "latin1")
    objs = files_and_groups(gofilename, snapnum, group="Stars")
    with open(gofilename+"/dm_shells_"+str(snapnum)+"_V2.dat",'wb') as f:   
        pickle.dump(objs, f)
    print("Done!")
