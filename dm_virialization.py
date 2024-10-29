import numpy as np
import numpy as np
import pickle
from concatenateclass import processedFOF
from sys import argv
from boundedness import get_starIDs, get_DMIDs, calc_boundedness,chunked_calc_boundedness,check_virialized,dist2_indv,chunked_potential_energy_same_mass,chunked_potential_energy_between_groups
from fof_process import dx_wrap, dist2, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, get_Halos, get_DMIDgroups, get_starIDgroups

#Set units and parameters
UnitMass_in_g = 1.989e43     
UnitLength_in_cm = 3.085678e21 
hubbleparam = .71 #hubble constant
GRAVITY_cgs = 6.672e-8
UnitVelocity_in_cm_per_s = 1.0e5


# def set_up_DM_fofs(filename, snapnum,sv):
#     """
#     Grabs FOF data using processed FOF class
    
#     Parameters: 
#         fileame (str): path to FOF directory
#         snapnum (str or int): 
    
#     Returns:
#         tuple: centers of bounded objs, radii (from boundedness) of bounded objects

#     """
#     if "DMP-GS-" in filename:
#         directory = "DMP-GS-"
#     fof = processedFOF(snapnum,directory,sv, path = '/u/home/c/clairewi/project-snaoz/FOF_project/') #call processed fof class 
#     return fof.properties['DMIDs'], fof.properties['starIDs'], fof.properties['starMasses']

def chunked_calc_dm_boundedness(energyStars, starPos_inGroup, starMass_inGroup, groupVelocity,boxSize,boxSizeVel, pDM, vDM ,atime,massDMParticle):
    """
    Calculate the boundedness of a stellar group with dark matter using chunked processing to reduce memory usage.

    Parameters:
        energyStars (float): Total energy of the stellar group.
        starVel_inGroup (ndarray): Array of star velocities in the group.
        starPos_inGroup (ndarray): Array of star positions in the group.
        starMass_inGroup (ndarray): Array of star masses in the group.
        groupPos (ndarray): Position of the group.
        groupVelocity (ndarray): Velocity of the group.
        boxSize (float): Size of the simulation box for positions.
        boxSizeVel (float): Size of the simulation box for velocities.
        pDM (ndarray): Array of dark matter particle positions in the group.
        vDM (ndarray): Array of dark matter particle velocities in the group.
        groupRadius (float): Radius of the group.
        atime (float): Scale factor at the snapshot time.
        massDMParticle (float): Mass of a dark matter particle.

    Returns:
        tuple: Boundedness 
    """
    pDM=  np.array(pDM) *atime / hubbleparam
    vDM = np.array(vDM) * np.sqrt(atime) 
    tempvelDM = dx_wrap(vDM-groupVelocity, boxSizeVel)
    velMagDM = np.sqrt((tempvelDM*tempvelDM).sum(axis=1))
    lengroup = len(pDM)
    massDM = lengroup* massDMParticle
    #Kinetic energy component
    kineticEnergyDM = np.sum(massDMParticle/2 *velMagDM*velMagDM*UnitVelocity_in_cm_per_s*UnitVelocity_in_cm_per_s)
    potentialEnergyDM = 0
    potentialEnergyStarsDM = 0
    #DM self potential energy
    print("chunked DM potential calculation")
    potentialEnergyDM = chunked_potential_energy_same_mass(massDMParticle, pDM,boxSize,chunk_size = 10000)
    print("chunked stars and dm potential calculation")
    potentialEnergyStarsDM = chunked_potential_energy_between_groups(massDMParticle, pDM, starMass_inGroup,starPos_inGroup,boxSize,chunk_size = 10000)
    totEnergy = energyStars + kineticEnergyDM + potentialEnergyStarsDM + potentialEnergyDM
    if totEnergy<0:
         print("object is bound after DM!")
         boundedness =  1
    else:
         print("not bound even after DM!")
         boundedness = 0
    return boundedness, totEnergy, kineticEnergyDM, potentialEnergyStarsDM + potentialEnergyDM, massDM


def get_fof_particles(filename, snapnum, sv):
    """
    Gets the particles from the snapshot
    """
    gofilename = str(filename)
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, redshift, massDMParticle = get_headerprops(snap)
    atime = 1./(1.+redshift)
    print('redshift is '+str(redshift))
    cat = set_subfind_catalog(fof)
    print("used groups of 300 or more DM")
    halo100_indices=get_Halos(cat)
    print(len(halo100_indices))
    _, allStarMasses, allStarPositions, allStarVelocities = get_starIDs(snap)
    _, allDMPositions, allDMVelocities = get_DMIDs(snap)
    groupPos =  cat.GroupPos[halo100_indices]
    groupVel = cat.GroupVel[halo100_indices]
    startAllDM, endAllDM = get_DMIDgroups(cat,halo100_indices)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    objs={}
    #DMIDs_inGroups, starIDs_inGroups, starMasses_inGroups = set_up_DM_fofs(str(filename),snapnum, sv)
    bounded, virialized, usedDM = iterate_objs(halo100_indices,groupPos,groupVel,startAllDM, endAllDM, startAllStars, endAllStars,allStarPositions, allStarMasses,allStarVelocities, allDMPositions, allDMVelocities,boxSize,atime,massDMParticle)
    objs['bounded'] = bounded
    objs['virialized'] = virialized
    objs['usedDM'] = usedDM
    print("saving output at" + str(gofilename))
    with open(gofilename+"/boundedDM_"+str(snapnum)+"_V1.dat",'wb') as f:   
        pickle.dump(objs, f)
    return 

def iterate_objs(halo100_indices,groupPos,groupVel,startAllDM, endAllDM, startAllStars, endAllStars,allStarPositions,allStarMasses, allStarVelocities, allDMPositions, allDMVelocities,boxSize, atime,massDMParticle): 
    """
    Iterate over DM primary objects and return whether or not the objects are bounded and virialized, as well as whether or not they are virialized
    
    Parameters: 
        halo100_indices (Numpy.ndarray): Indices of halos containing 100 or more star particles.
        allStarMasses (Numpy.ndarray):  Array of stellar masses for all particles.
        allStarPositions (Numpy.ndarray): Array of stellar positions for all particles.
        allStarVelocities (Numpy.ndarray): Array of stellar velocities for all particles.
        startAllStars (Numpy.ndarray): Start indices of star particles for each galaxy.
        endAllStars (Numpy.ndarray): End indices of star particles for each galaxy.
        atime (float): Scale factor for the current snapshot.
        boxSize (float): Size of the simulation box in comoving units.
        startAllDM (Numpy.ndarray):  Start indices of dark matter particles for each galaxy.
        endAllDM (Numpy.ndarray):  End indices of dark matter particles for each galaxy.
        massDMParticle (float): Dark matter particle mass.


    Returns: 
        (int) whether or not the objects are bounded, (int) whether or not the objects are virialized, (int) whether or not DM was used to establish virialization
    """
    Omega0 = 0.27
    OmegaLambda = 0.73
    groupPos = groupPos *atime / hubbleparam 
    groupVel = groupVel /atime # convert to physical units
    #hubble flow correction
    boxSizeVel = boxSize * hubbleparam * .1 * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)
    boxSize = boxSize * atime/hubbleparam
    boundednessgroups = np.zeros(len(halo100_indices), dtype = float)
    virialgroups  = np.zeros(len(halo100_indices), dtype = float)
    usedDMgroups  = np.zeros(len(halo100_indices), dtype = float)
    for i,j in enumerate(halo100_indices):
        usedDM = 0
        starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
        starVel_inGroup = allStarVelocities[startAllStars[i]:endAllStars[i]]
        starMass_inGroup = allStarMasses[startAllStars[i]:endAllStars[i]]
        starMass_inGroup = np.array(starMass_inGroup) * UnitMass_in_g / hubbleparam #convert masses
        starVel_inGroup = np.array(starVel_inGroup) * np.sqrt(atime) #unit conversions on the particle coordinates 
        starPos_inGroup = np.array(starPos_inGroup) *atime / hubbleparam
        boundedness, energyStars, kineticEnergyStars, potentialEnergyStars, _=chunked_calc_boundedness(starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos[i],groupVel[i],boxSize,boxSizeVel)
        virialized, _ = check_virialized(kineticEnergyStars,potentialEnergyStars)
        if virialized ==0:
            usedDM = 1
            pDM = allDMPositions[startAllDM[i]:endAllDM[i]]
            vDM = allDMVelocities[startAllDM[i]:endAllDM[i]]
            boundednessDM,_, kineticEnergyDM, potentialEnergyDM, _ =  chunked_calc_dm_boundedness(energyStars, starPos_inGroup, starMass_inGroup, groupVel[i],boxSize,boxSizeVel, pDM, vDM ,atime,massDMParticle)
            kineticEnergy = kineticEnergyDM +kineticEnergyStars
            potentialEnergy = potentialEnergyDM +potentialEnergyStars
            virialized,_ = check_virialized(kineticEnergy, potentialEnergy)
        if (boundedness == 1) or (boundednessDM == 1):
            boundednessgroups[i] = 1
        virialgroups[i]=virialized
        usedDMgroups[i]= usedDM
    return boundednessgroups, virialgroups, usedDMgroups

if __name__ =="__main__":
    script, gofilename, snapnum, SV = argv
    get_fof_particles(gofilename, snapnum, SV)