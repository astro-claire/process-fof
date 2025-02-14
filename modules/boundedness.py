import h5py 
import numpy as np
from sys import argv
import pickle
import sys 
import os
import concurrent.futures
from functools import partial
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.fof_process import dx_wrap, dist2, get_DMIDgroups, get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config,get_gasGroups, get_cosmo_props,get_starIDgroups, get_headerprops

import config.configuration as config

#Set units and parameters
constants = config.load_constants()
UnitMass_in_g = constants['UnitMass_in_g']     
UnitLength_in_cm = constants['UnitLength_in_cm']
hubbleparam = constants['hubbleparam'] #hubble constant
GRAVITY_cgs = constants['GRAVITY_cgs']
UnitVelocity_in_cm_per_s = constants['UnitVelocity_in_cm_per_s']


def get_allHalos(cat):
    """
    Identify all halos containing more than 32 dark matter (DM) particles.

    Parameters:
        cat (halo catalog): The catalog containing halo information, including the number of particles per halo.

    Returns:
        np.ndarray: Indices of halos with more than 32 DM particles.
    """
    #print("Warning!! halos set to testing mode!")
    over300idx, = np.where(np.greater(cat.GroupLenType[:,1],32))

    return over300idx

def set_up_DM(SV, snapnum):   
    """
    Set up the dark matter (DM) halo catalog from the simulation snapshot.

    Parameters:
        SV (int or str): Simulation version identifier (used for directory/file naming).
        snapnum (int): Snapshot number to retrieve.

    Returns:
        tuple: A tuple containing the following:
            - halo100_indices (np.ndarray): Indices of halos with more than 32 DM particles.
            - halo_positions (np.ndarray): Positions of the halos.
            - startAllDM (np.ndarray): Starting indices for DM particles in each halo.
            - endAllDM (np.ndarray): Ending indices for DM particles in each halo.
            - dmsnap (h5py snapshot): The opened HDF5 snapshot file containing DM data.
    """
    inputdir, outputdir = config.set_directories()
    # gofilename = '/u/home/c/clairewi/project-snaoz/FOF_project/DMP-GS-' + str(SV) 
    gofilename = str(inputdir)+ '/DMP-GS-' + str(SV) 
    dmgofile, dmfoffile  = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename) )
    dmsnap, dmfof = open_hdf5(dmgofile, dmfoffile)
    cat = set_subfind_catalog(dmfof)
    halo100_indices=get_allHalos(cat) #this gets all the halos even those without the DM particles
    halo_positions = get_GroupPos(cat, halo100_indices)
    print(len(halo100_indices)) 
    startAllDM, endAllDM = get_DMIDgroups(cat, halo100_indices)
    return halo100_indices, halo_positions, startAllDM, endAllDM, dmsnap

def nearby_DM(com, halo_positions, startAllDM, endAllDM,dmsnap, boxSize,limit =10.):
    """
    Find all halos and their DM particles within a certain distance from a specified center of mass (com).

    Parameters:
        com (np.ndarray): The 3D coordinates of the center of mass (x, y, z).
        halo_positions (np.ndarray): The positions of the halos in the box.
        startAllDM (np.ndarray): Starting indices for DM particles in each halo.
        endAllDM (np.ndarray): Ending indices for DM particles in each halo.
        dmsnap (object): The opened HDF5 snapshot file containing DM data.
        boxSize (float): The size of the simulation box.
        limit (float, optional): The maximum distance (in code units) from the center of mass to search for nearby halos. Defaults to 10.0.

    Returns:
        tuple: A tuple containing the following:
            - allnearbyDMPos (np.ndarray): Positions of the DM particles in the nearby halos.
            - allnearbyDMVel (np.ndarray): Velocities of the DM particles in the nearby halos.
    """
    print("finding the nearby halos within "+ str(limit))
    distances = dist2(halo_positions[:,0]-com[0], halo_positions[:,1]-com[1],halo_positions[:,2]-com[2], boxSize)
    nearhalos = np.where(distances< limit**2)[0]
    if len(nearhalos) == 0:
        return np.array([]), np.array([])  # Return empty arrays if no halos are nearby
    _,allDMPositions, allDMVelocities= get_DMIDs(dmsnap)
    total_DM_size = sum(endAllDM[i] - startAllDM[i] for i in nearhalos)
    # Pre-allocate arrays for nearby DM positions and velocities
    allnearbyDMPos = np.empty((total_DM_size, 3))
    allnearbyDMVel = np.empty((total_DM_size, 3))
    len_counter = 0
    for i in nearhalos:
        start_idx = startAllDM[i]
        end_idx = endAllDM[i]
        num_dm_particles = end_idx - start_idx
        
        allnearbyDMPos[len_counter:len_counter + num_dm_particles] = allDMPositions[start_idx:end_idx]
        allnearbyDMVel[len_counter:len_counter + num_dm_particles] = allDMVelocities[start_idx:end_idx]
        
        len_counter += num_dm_particles
    print("DM pos length is"+str(len_counter))
    return allnearbyDMPos, allnearbyDMVel
    

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    Function from chat gpt lol

    Parameters: 
        lst (list): list or array to divide into chunks
        n (int): chunk size
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def dx_indv(dx, box):
    """
    Apply periodic boundary conditions to a coordinate difference.

    Parameters:
        dx (float): The difference in a single coordinate (x, y, or z).
        box (float): The size of the simulation box.

    Returns:
        float: The adjusted coordinate difference, taking into account periodic boundary conditions.
    """
    if dx > +box/2.0:
        dx -= box
    if dx < -box/2.0:
        dx += box 
    return dx

def dist2_indv(dx,dy,dz,box):
    """
    Calculate the squared distance between two points with periodic boundary conditions.

    Parameters:
        dx (float): The difference in the x-coordinate between the two points.
        dy (float): The difference in the y-coordinate between the two points.
        dz (float): The difference in the z-coordinate between the two points.
        box (float): The size of the simulation box.

    Returns:
        float: The squared distance between the two points, considering periodic boundary conditions.
    """
    return dx_indv(dx,box)**2 + dx_indv(dy,box)**2 + dx_indv(dz,box)**2

def get_GroupRadii(cat, halo100_indices):
    """
    Return Group COM
    """
    return cat.Group_R_Crit200[halo100_indices]

def get_GroupVel(cat, halo100_indices):
    """
    Return Group COM
    """
    return cat.GroupVel[halo100_indices]

def get_GroupPos(cat, halo100_indices):
    """
    Return Group COM
    """
    return cat.GroupPos[halo100_indices]

def get_starIDs(f):
    """
    Get particle IDs (groupordered snap)
    """
    allStarIDs = f['PartType4/ParticleIDs']
    allStarMasses = f['PartType4/Masses']
    allStarPositions = f['PartType4/Coordinates']
    allStarVelocities = f['PartType4/Velocities']
    return allStarIDs, allStarMasses, allStarPositions, allStarVelocities

def get_DMIDs(f):
	"""
	Get particle IDs (groupordered snap)
	"""
	allDMIDs = f['PartType1/ParticleIDs']
	allDMPositions = f['PartType1/Coordinates']
	allDMVelocities = f['PartType1/Velocities']
	
	return allDMIDs, allDMPositions, allDMVelocities

def check_virialized(kineticEnergy, potentialEnergy):
    """
    Check if an object is virialized based on its kinetic and potential energy.

    Parameters:
        kineticEnergy (float): The kinetic energy of the object.
        potentialEnergy (float): The potential energy of the object.

    Returns:
        tuple: A tuple containing:
            - virialized (int): 1 if the object is virialized (virial ratio between 1.5 and 2.5), 0 otherwise.
            - ratio (float): The virial ratio (|potentialEnergy / kineticEnergy|).
    """
    ratio = np.abs(potentialEnergy/kineticEnergy)
    virialized = 0
    if 1.5<ratio<2.5:
        print("object is virialized")
        virialized = 1
    return virialized, ratio

def calc_virial_radius(potentialEnergy, mass):
    """
    Calculate the virial radius given the mass and potential energy of an object (in cgs units).

    Parameters:
        potentialEnergy (float): The potential energy of the object.
        mass (float): The mass of the object.

    Returns:
        float: The virial radius of the object.
    """
    return - GRAVITY_cgs * mass * mass / potentialEnergy

def calc_max_radius(starPos_inGroup,groupPos,boxSize):
    """
    Calculate the maximum radial distance between a group of stars and the group's center.

    Parameters:
        starPos_inGroup (np.ndarray): The positions of stars in the group (shape: N x 3).
        groupPos (np.ndarray): The position of the group's center (shape: 3).
        boxSize (float): The size of the simulation box.

    Returns:
        float: The maximum distance between the stars and the group's center, in cm.
    """
    distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
    maxdist = max(distances)
    return UnitLength_in_cm* np.sqrt(maxdist)


def calc_boundedness(starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos,groupVelocity,boxSize,boxSizeVel):
    """
    Calculate whether a group of stars is gravitationally bound based on their velocities and positions.

    Parameters:
        starVel_inGroup (np.ndarray): The velocities of the stars in the group (shape: N x 3).
        starPos_inGroup (np.ndarray): The positions of the stars in the group (shape: N x 3).
        starMass_inGroup (np.ndarray): The masses of the stars in the group (shape: N).
        groupPos (np.ndarray): The position of the group's center (shape: 3).
        groupVelocity (np.ndarray): The velocity of the group's center (shape: 3).
        boxSize (float): The size of the simulation box.
        boxSizeVel (float): The simulation velocity to offset.

    Returns:
        tuple: A tuple containing:
            - boundedness (int): 1 if the group is gravitationally bound, 0 otherwise.
            - energyStars (float): The total energy (kinetic + potential) of the stars in the group.
            - kineticEnergyStars (float): The total kinetic energy of the stars in the group.
            - potentialEnergyStars (float): The total potential energy of the stars in the group.
            - massStars (float): The total mass of the stars in the group.
    """
    tempvelstars = dx_wrap(starVel_inGroup-groupVelocity, boxSizeVel)
    velMagStars = np.sqrt((tempvelstars*tempvelstars).sum(axis=1))
    kineticEnergyStars = np.sum(starMass_inGroup/2 *velMagStars*velMagStars*UnitVelocity_in_cm_per_s*UnitVelocity_in_cm_per_s)
    potentialEnergyStars = 0
    massStars = np.sum(starMass_inGroup)
    print("stellarmass is " + str(massStars))
    lengroup = len(starMass_inGroup)
    print("group len is "+str(lengroup))

    for i in range(lengroup):
         for j in range(i+1,lengroup):
            #r_ij = UnitLength_in_cm* np.linalg.norm(starPos_inGroup[i] - starPos_inGroup[j])  # Compute distance between mass i and mass j
            r_ij  = UnitLength_in_cm* np.sqrt(dist2_indv(starPos_inGroup[i,0]-starPos_inGroup[j,0],starPos_inGroup[i,1]-starPos_inGroup[j,1],starPos_inGroup[i,2]-starPos_inGroup[j,2],boxSize))
            if r_ij != 0:
                 potentialEnergyStars += -GRAVITY_cgs * starMass_inGroup[i] * starMass_inGroup[j] / r_ij              
    energyStars = kineticEnergyStars+ potentialEnergyStars
    print("total energy")
    print(energyStars)
    if energyStars<0:
         print("object is bound")
         boundedness =  1
    else:
         print("object not bound")
         boundedness = 0
    return boundedness, energyStars, kineticEnergyStars, potentialEnergyStars, massStars

#Functions for parallel tensor creation

def tensor_dist(i,j, starPos_inGroup = np.array([[0,0,0]]),boxSize=1776.):
    """
    Computes the distance between particles i and j
    
    Parameters: 
        i (int): index of particle i
        j (int): index of particle j
        starPos_inGroup (np.ndarray): The positions of the stars in the group (shape: N x 3).
        boxSize (float): The size of the simulation box
    Returns: 
        float: the distance between particle i and j in cm
    """
    return UnitLength_in_cm* np.sqrt(dist2_indv(starPos_inGroup[i,0]-starPos_inGroup[j,0],starPos_inGroup[i,1]-starPos_inGroup[j,1],starPos_inGroup[i,2]-starPos_inGroup[j,2],boxSize))

def tensor_dist_2type(i,j, starPos_inGroup = np.array([[0,0,0]]),pDM = np.array([[0,0,0]]),boxSize=1776.):
    """
    Computes the distance between particles i and j where i is a DM particle and j is a star particle
    
    Parameters: 
        i (int): index of particle i
        j (int): index of particle j
        starPos_inGroup (np.ndarray): The positions of the stars in the group (shape: N x 3).
        pDM (np.ndarray): The positions of the dark matter in the group (shape: M x 3).
        boxSize (float): The size of the simulation box.

    Returns: 
        float: the distance between particle i and j in cm
    """
    return UnitLength_in_cm* np.sqrt(dist2_indv(pDM[i,0]-starPos_inGroup[j,0],pDM[i,1]-starPos_inGroup[j,1],pDM[i,2]-starPos_inGroup[j,2],boxSize))

def compute_row(i_elem,J, starPos_inGroup,boxSize):
    """
    Compute the distance tensor row for a given particle in a group.

    Parameters:
        i_elem (int): Index of the particle for which the distances are computed.
        J (iterable): Iterable of indices of particles to compute distances with respect to.
        starPos_inGroup (ndarray): Array of particle positions within the group.
        boxSize (float): Size of the simulation box.

    Returns:
        List[float]: List of distances between the particle `i_elem` and particles in `J`.
    """
    return [tensor_dist(i_elem, j_elem,starPos_inGroup=starPos_inGroup,boxSize=boxSize) for j_elem in J]

def compute_row_2type(i_elem,J, starPos_inGroup,pDM,boxSize):
    """
    Compute the distance tensor row for a given particle in a group, with respect to particles of a different type (e.g., dark matter).

    Parameters:
        i_elem (int): Index of the particle for which the distances are computed.
        J (iterable): Iterable of indices of particles to compute distances with respect to.
        starPos_inGroup (ndarray): Array of particle positions within the group.
        pDM (ndarray): Array of positions for the second particle type (e.g., dark matter).
        boxSize (float): Size of the simulation box.

    Returns:
        List[float]: List of distances between the particle `i_elem` and particles in `J` from the second particle type.
    """
    return [tensor_dist_2type(i_elem, j_elem,starPos_inGroup=starPos_inGroup,pDM = pDM,boxSize=boxSize) for j_elem in J]


def parallel_calc_boundedness(starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos,groupVelocity,boxSize,boxSizeVel):
    """
    Calculate the boundedness of a stellar group using parallel processing.

    Parameters:
        starVel_inGroup (ndarray): Array of star velocities in the group.
        starPos_inGroup (ndarray): Array of star positions in the group.
        starMass_inGroup (ndarray): Array of star masses in the group.
        groupPos (ndarray): Position of the group.
        groupVelocity (ndarray): Velocity of the group.
        boxSize (float): Size of the simulation box for positions.
        boxSizeVel (float): Velocity for simualtion box hubble flow correction.

    Returns:
        tuple: Boundedness flag (1 if bound, 0 otherwise), total energy, kinetic energy, potential energy, and total stellar mass.
    """
    tempvelstars = dx_wrap(starVel_inGroup-groupVelocity, boxSizeVel)
    velMagStars = np.sqrt((tempvelstars*tempvelstars).sum(axis=1))
    kineticEnergyStars = np.sum(starMass_inGroup/2 *velMagStars*velMagStars*UnitVelocity_in_cm_per_s*UnitVelocity_in_cm_per_s)
    potentialEnergyStars = 0
    massStars = np.sum(starMass_inGroup)
    print("stellarmass is " + str(massStars))
    lengroup = len(starMass_inGroup)
    print("group len is "+str(lengroup))
    idxgroup = range(lengroup)
    # Use ThreadPoolExecutor for parallel processing
    print("creating the tensor")
    max_workers = 24
    try: 
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Using functools.partial to include additional arguments
            #result = list(executor.map(lambda i_elem: compute_row(i_elem,J= idxgroup, starPos_inGroup=starPos_inGroup, boxSize = boxSize), idxgroup))
            #result = list(executor.map(lambda i_elem: compute_row(i_elem,idxgroup, starPos_inGroup,boxSize), idxgroup))
            result = list(executor.map(partial(compute_row,J= idxgroup, starPos_inGroup=starPos_inGroup, boxSize = boxSize), idxgroup))
    except Exception:
        print('encountered error, trying again')
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Using functools.partial to include additional arguments
                #result = list(executor.map(lambda i_elem: compute_row(i_elem,J= idxgroup, starPos_inGroup=starPos_inGroup, boxSize = boxSize), idxgroup))
                #result = list(executor.map(lambda i_elem: compute_row(i_elem,idxgroup, starPos_inGroup,boxSize), idxgroup))
                result = list(executor.map(partial(compute_row,J= idxgroup, starPos_inGroup=starPos_inGroup, boxSize = boxSize), idxgroup))
        except Exception:
            print("again an error. ugh! doing the normal way then.")
            result = np.zeros((lengroup, lengroup))
            for i in range(lengroup):
                for j in range(i+1,lengroup):
                    #r_ij = UnitLength_in_cm* np.linalg.norm(starPos_inGroup[i] - starPos_inGroup[j])  # Compute distance between mass i and mass j
                    result[i][j]  = UnitLength_in_cm* np.sqrt(dist2_indv(starPos_inGroup[i,0]-starPos_inGroup[j,0],starPos_inGroup[i,1]-starPos_inGroup[j,1],starPos_inGroup[i,2]-starPos_inGroup[j,2],boxSize))
    # Convert result to tensor-like structure
    tensor_r_ij = result
    print("created the tensor")
    for i in range(lengroup):
         r_ij = 0
         for j in range(i+1,lengroup):
            #r_ij = UnitLength_in_cm* np.linalg.norm(starPos_inGroup[i] - starPos_inGroup[j])  # Compute distance between mass i and mass j
            #r_ij  = UnitLength_in_cm* np.sqrt(dist2_indv(starPos_inGroup[i,0]-starPos_inGroup[j,0],starPos_inGroup[i,1]-starPos_inGroup[j,1],starPos_inGroup[i,2]-starPos_inGroup[j,2],boxSize))
            r_ij = tensor_r_ij[i][j]
            if r_ij != 0:
                 potentialEnergyStars += -GRAVITY_cgs * starMass_inGroup[i] * starMass_inGroup[j] / r_ij              
    energyStars = kineticEnergyStars+ potentialEnergyStars
    print("total energy")
    print(energyStars)
    if energyStars<0:
         print("object is bound")
         boundedness =  1
    else:
         print("object not bound")
         boundedness = 0
    return boundedness, energyStars, kineticEnergyStars, potentialEnergyStars, massStars

def chunked_potential_energy(masses, positions, box_size, G=GRAVITY_cgs, chunk_size = 10000):
    """
    Calculate the total potential energy of a system of particles using pairwise interactions.
    
    Parameters:
        masses : np.array
            1D array of particle masses.
        positions : np.array
            2D array of particle positions (shape: N x 3).
        box_size : float
            Size of the periodic box.
        G : float
            Gravitational constant,
    Returns:
        total_potential : float
            Total potential energy of the system.
    """
    N = len(masses)
    total_potential = 0.0
    if N<=1: #no potential energy if no particles in the group
        total_potential =0
    else: 
        if chunk_size > (N/3):
            chunk_size = int(N/100)  # Use chunking to avoid excessive memory usage
            if chunk_size ==0 and N>20:
                chunk_size =int(N/20)
            elif chunk_size==0:
                chunk_size =1
        for i in range(0, N, chunk_size):
            #print(f"starting potential chunk {i}")
            for j in range(i + 1, N, chunk_size):
                # Select smaller chunks to handle
                masses_i = masses[i:i+chunk_size]
                masses_j = masses[j:j+chunk_size]
                pos_i = positions[i:i+chunk_size]
                pos_j = positions[j:j+chunk_size]
                
                # Calculate pairwise distances between chunks
                for k in range(len(masses_i)):
                    dx = pos_i[k, 0] - pos_j[:, 0]
                    dy = pos_i[k, 1] - pos_j[:, 1]
                    dz = pos_i[k, 2] - pos_j[:, 2]
                    
                    r2 = dist2(dx, dy, dz, box_size)
                    r = np.sqrt(r2)
                    
                    # Avoid division by zero by setting an effective minimum distance
                    r = np.where(r < 1e-10, 1e-10, r)
                    mask = r >= 2e-10
                    valid_r = r[mask]
                    valid_masses_j = masses_j[mask]

                    # Sum up potential energy contributions
                    total_potential += -G * np.sum(masses_i[k] * valid_masses_j / valid_r/UnitLength_in_cm)

    return total_potential

def chunked_potential_energy_same_mass(mass, positions, box_size, G=GRAVITY_cgs, chunk_size = 10000):
    """
    Calculate the total potential energy of a system of particles with the same mass using pairwise interactions.
    
    Parameters:
        mass : float
            Mass of each particle (same for all particles).
        positions : np.array
            2D array of particle positions (shape: N x 3).
        box_size : float
            Size of the periodic box.
        G : float
            Gravitational constant, defaults to cgs units.
    
    Returns:
        total_potential : float
            Total potential energy of the system.
    """
    N = len(positions)
    masses =np.ones(N)*mass
    total_potential = 0.0
    if N<=1:
        total_potential =0.0
    else: 
        if chunk_size >(N/3): #errors will occur if the chunks are too large
            chunk_size = int(N/100)  # Use chunking to avoid excessive memory usage
            if chunk_size ==0 and N>20:
                chunk_size =int(N/20)
            elif chunk_size==0:
                chunk_size =1
        for i in range(0, N, chunk_size):
            #print(f"starting potential chunk {i}")
            for j in range(i + 1, N, chunk_size):
                # Select smaller chunks to handle
                masses_i = masses[i:i+chunk_size]
                masses_j = masses[j:j+chunk_size]
                pos_i = positions[i:i+chunk_size]
                pos_j = positions[j:j+chunk_size]
                
                # Calculate pairwise distances between chunks
                for k in range(len(masses_i)):
                    dx = pos_i[k, 0] - pos_j[:, 0]
                    dy = pos_i[k, 1] - pos_j[:, 1]
                    dz = pos_i[k, 2] - pos_j[:, 2]
                    
                    r2 = dist2(dx, dy, dz, box_size)
                    r = np.sqrt(r2)
                    
                    # Avoid division by zero by setting an effective minimum distance
                    r = np.where(r < 1e-10, 1e-10, r)
                    mask = r >= 2e-10
                    valid_r = r[mask]
                    valid_masses_j = masses_j[mask]

                    # Sum up potential energy contributions
                    total_potential += -G * np.sum(masses_i[k] * valid_masses_j / valid_r/UnitLength_in_cm)

    return total_potential


# def chunked_potential_energy_between_sets(masses1, positions1, masses2, positions2, box_size, G=GRAVITY_cgs, chunk_size = 10000):
#     """
#     Calculate the total potential energy between two different sets of particles using pairwise interactions.
    
#     Parameters:
#         masses1 : np.array
#             1D array of particle masses for set 1.
#         positions1 : np.array
#             2D array of particle positions for set 1 (shape: N1 x 3).
#         masses2 : np.array
#             1D array of particle masses for set 2.
#         positions2 : np.array
#             2D array of particle positions for set 2 (shape: N2 x 3).
#         box_size : float
#             Size of the periodic box.
#         G : float
#             Gravitational constant, defaults to cgs units.
    
#     Returns:
#         total_potential : float
#             Total potential energy between the two sets of particles.
#     """
#     N1 = len(masses1)
#     N2 = len(masses2)
#     total_potential = 0.0
#     chunk_size = chunk_size  # Use chunking to avoid excessive memory usage

#     # Loop over the first set of particles in chunks
#     for i in range(0, N1, chunk_size):
#         # Loop over the second set of particles in chunks
#         for j in range(0, N2, chunk_size):
#             # Select chunks of particles
#             masses1_chunk = masses1[i:i+chunk_size]
#             positions1_chunk = positions1[i:i+chunk_size]
#             masses2_chunk = masses2[j:j+chunk_size]
#             positions2_chunk = positions2[j:j+chunk_size]
            
#             # Calculate pairwise distances between the two sets
#             for k in range(len(masses1_chunk)):
#                 dx = positions1_chunk[k, 0] - positions2_chunk[:, 0]
#                 dy = positions1_chunk[k, 1] - positions2_chunk[:, 1]
#                 dz = positions1_chunk[k, 2] - positions2_chunk[:, 2]
                
#                 r2 = dist2(dx, dy, dz, box_size)
#                 r = np.sqrt(r2)
                
#                 # Avoid division by zero by setting an effective minimum distance
#                 mask = r >= 1e-10
#                 valid_r = r[mask]
#                 valid_masses_j = masses2_chunk[mask]
                
#                 # Sum up potential energy contributions between particles in set 1 and set 2
#                 total_potential += -G * np.sum(masses1_chunk[k] * valid_masses_j /valid_r /UnitLength_in_cm)

#     return total_potential

def chunked_potential_energy_between_groups(mass1, positions1, masses2, positions2, box_size, G=GRAVITY_cgs, chunk_size = 10000):
    """
    Calculate the total potential energy between two groups of particles, 
    where the first group has the same mass for all particles, and the second group has an array of masses.
    
    Parameters:
        mass1 : float
            Mass of each particle in the first group (same for all particles in group 1).
        positions1 : np.array
            2D array of particle positions for group 1 (shape: N1 x 3).
        masses2 : np.array
            1D array of particle masses for group 2.
        positions2 : np.array
            2D array of particle positions for group 2 (shape: N2 x 3).
        box_size : float
            Size of the periodic box.
        G : float
            Gravitational constant, defaults to cgs units.
    
    Returns:
        total_potential : float
            Total potential energy between the two groups of particles.
    """
    N1 = len(positions1)
    N2 = len(masses2)
    total_potential = 0.0
    if N1==0 or N2 ==0: #edge case where one set of particles is empty
        total_potential = 0.0
    else: 
        if chunk_size >(N1/3): #errors will occur if the chunks are too large
            chunk_size = int(N1/100)  # Use chunking to avoid excessive memory usage
            if chunk_size ==0 and N1>20:
                chunk_size =int(N1/20)
            elif chunk_size==0:
                chunk_size =1
        if chunk_size >(N2/3): #errors will occur if the chunks are too large
            chunk_size = int(N2/100)  # Use chunking to avoid excessive memory usage
            if chunk_size ==0 and N2>20:
                chunk_size =int(N2/20)
            elif chunk_size==0:
                chunk_size =1
        # Loop over the first group of particles in chunks
        for i in range(0, N1, chunk_size):
            # Loop over the second group of particles in chunks
            for j in range(0, N2, chunk_size):
                # Select chunks of particles
                positions1_chunk = positions1[i:i+chunk_size]
                positions2_chunk = positions2[j:j+chunk_size]
                masses2_chunk = masses2[j:j+chunk_size]
                
                # Calculate pairwise distances between the two groups
                for k in range(len(positions1_chunk)):
                    dx = positions1_chunk[k, 0] - positions2_chunk[:, 0]
                    dy = positions1_chunk[k, 1] - positions2_chunk[:, 1]
                    dz = positions1_chunk[k, 2] - positions2_chunk[:, 2]
                    
                    r2 = dist2(dx, dy, dz, box_size)
                    r = np.sqrt(r2) 
                    
                    # Avoid division by zero by setting an effective minimum distance
                    r = np.where(r < 1e-10, 1e-10, r)
                    mask = r >= 2e-10
                    valid_r = r[mask]
                    valid_masses_j = masses2_chunk[mask]

                    # Sum up potential energy contributions
                    total_potential += -G * np.sum(mass1 * valid_masses_j / valid_r /UnitLength_in_cm)

    return total_potential

def chunked_calc_boundedness(starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos,groupVelocity,boxSize,boxSizeVel):
    """
    Calculate the boundedness of a stellar group using chunked processing to reduce memory usage.

    Parameters:
        starVel_inGroup (ndarray): Array of star velocities in the group.
        starPos_inGroup (ndarray): Array of star positions in the group.
        starMass_inGroup (ndarray): Array of star masses in the group.
        groupPos (ndarray): Position of the group.
        groupVelocity (ndarray): Velocity of the group.
        boxSize (float): Size of the simulation box for positions.
        boxSizeVel (float): Size of the simulation box for velocities.

    Returns:
        tuple: Boundedness flag (1 if bound, 0 otherwise), total energy, kinetic energy, potential energy, and total stellar mass.
    """
    tempvelstars = dx_wrap(starVel_inGroup-groupVelocity, boxSizeVel)
    velMagStars = np.sqrt((tempvelstars*tempvelstars).sum(axis=1))
    kineticEnergyStars = np.sum(starMass_inGroup/2 *velMagStars*velMagStars*UnitVelocity_in_cm_per_s*UnitVelocity_in_cm_per_s)
    if 'inf' in str(kineticEnergyStars):
        chunk_size = 500  # Process in smaller batches
        kineticEnergyStars = 0.0
        #doing this way to eliminate infinite error
        for i in range(0, len(velMagStars), chunk_size):
            chunk = velMagStars[i:i + chunk_size]
            chunkstarmass = starMass_inGroup[i:i + chunk_size]
            kineticEnergyStars += np.sum(chunkstarmass /UnitMass_in_g/ 2 * chunk**2 * UnitVelocity_in_cm_per_s**2)   
        kineticEnergyStars = kineticEnergyStars*UnitMass_in_g #add back in the units
    potentialEnergyStars = 0
    massStars = np.sum(starMass_inGroup)
    print("stellarmass is " + str(massStars))
    lengroup = len(starMass_inGroup)
    print("group len is "+str(lengroup))
    potentialEnergyStars = chunked_potential_energy(starMass_inGroup, starPos_inGroup,boxSize,chunk_size = 10000)
    energyStars = kineticEnergyStars+ potentialEnergyStars
    print("total energy")
    print(energyStars)
    print(kineticEnergyStars, potentialEnergyStars)
    if energyStars<0:
         print("object is bound")
         boundedness =  1
    else:
         print("object not bound")
         boundedness = 0
    return boundedness, energyStars, kineticEnergyStars, potentialEnergyStars, massStars


def parallel_calc_dm_boundedness(energyStars,starVel_inGroup, starPos_inGroup, starMass_inGroup,  groupPos, groupVelocity,boxSize,boxSizeVel, pDM, vDM,groupRadius ,atime,massDMParticle):
    """
    Calculate the boundedness of a stellar group with dark matter using parallel processing.

    Parameters:
        energyStars (float): Total energy of the stellar group.
        starVel_inGroup (ndarray): Array of star velocities in the group.
        starPos_inGroup (ndarray): Array of star positions in the group.
        starMass_inGroup (ndarray): Array of star masses in the group.
        groupPos (ndarray): Position of the group.
        groupVelocity (ndarray): Velocity of the group.
        boxSize (float): Size of the simulation box for positions.
        boxSizeVel (float): Size of the simulation box for velocities.
        pDM (ndarray): Array of dark matter particle positions.
        vDM (ndarray): Array of dark matter particle velocities.
        groupRadius (float): Radius of the group.
        atime (float): Scale factor at the snapshot time.
        massDMParticle (float): Mass of a dark matter particle.

    Returns:
        tuple: Boundedness flag (1 if bound, 0 otherwise), total energy, kinetic energy of dark matter, total potential energy between stars and dark matter, and dark matter mass.
    """
    if groupRadius <= 0:
        distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
        maxdist = max(distances)
    else:
        maxdist = (groupRadius *atime /hubbleparam) **2
    pDM=  np.array(pDM) *atime / hubbleparam
    vDM = np.array(vDM) * np.sqrt(atime) 
    distances = dist2(pDM[:,0]-groupPos[0],pDM[:,1]-groupPos[1],pDM[:,2]-groupPos[2],boxSize) #Note this is distance SQUARED
    inGroupDM = np.where(distances<maxdist)[0]
    tempvelDM = dx_wrap(vDM[inGroupDM]-groupVelocity, boxSizeVel)
    velMagDM = np.sqrt((tempvelDM*tempvelDM).sum(axis=1))
    #Kinetic energy component
    kineticEnergyDM = np.sum(massDMParticle/2 *velMagDM*velMagDM*UnitVelocity_in_cm_per_s*UnitVelocity_in_cm_per_s)
    potentialEnergyDM = 0
    potentialEnergyStarsDM = 0
    lengroup = len(inGroupDM)
    massDM = lengroup* massDMParticle
    #DM self potential energy
    idxgroup = range(lengroup)
    # Use ThreadPoolExecutor for parallel processing
    print("creating the tensor")
    max_workers = os.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Using functools.partial to include additional arguments
        result = list(executor.map(partial(compute_row,J= idxgroup, starPos_inGroup=pDM[inGroupDM], boxSize = boxSize), idxgroup))
    # Convert result to tensor-like structure
    tensor_r_ij = result
    print("created the tensor")
    for i in range(lengroup):
            r_ij = 0
            for j in range(i+1,lengroup):
                r_ij = tensor_r_ij[i][j]
            if r_ij != 0:
                potentialEnergyDM += -(GRAVITY_cgs * massDMParticle**2) / r_ij             
    lenstars= len(starMass_inGroup)
    idxstars = range(lenstars)
    print("creating the stars + DM tensor")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Using functools.partial to include additional arguments
        result = list(executor.map(partial(compute_row_2type,J= idxstars, starPos_inGroup=starPos_inGroup, pDM=pDM[inGroupDM],boxSize = boxSize), idxgroup))
    # Convert result to tensor-like structure
    tensor_r_ij_stardm = result
    print("finished making the tensor")
     #potential energy between stars and DM
    for i in range(lengroup):
         for j in range(lenstars):
            r_ij = tensor_r_ij_stardm[i][j]
            if r_ij != 0:
                 potentialEnergyStarsDM += -(GRAVITY_cgs * massDMParticle *starMass_inGroup[j] ) / r_ij               
    totEnergy = energyStars + kineticEnergyDM + potentialEnergyStarsDM + potentialEnergyDM
    if totEnergy<0:
         print("object is bound after DM!")
         boundedness =  1
    else:
         print("not bound even after DM!")
         boundedness = 0
    return boundedness, totEnergy, kineticEnergyDM, potentialEnergyStarsDM + potentialEnergyDM, massDM

def chunked_calc_dm_boundedness(energyStars,starVel_inGroup, starPos_inGroup, starMass_inGroup,  groupPos, groupVelocity,boxSize,boxSizeVel, pDM, vDM,groupRadius ,atime,massDMParticle):
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
        pDM (ndarray): Array of dark matter particle positions.
        vDM (ndarray): Array of dark matter particle velocities.
        groupRadius (float): Radius of the group.
        atime (float): Scale factor at the snapshot time.
        massDMParticle (float): Mass of a dark matter particle.

    Returns:
        tuple: Boundedness 
    """
    if groupRadius <= 0:
        distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
        maxdist = max(distances)
    else:
        maxdist = (groupRadius *atime /hubbleparam) **2
    pDM=  np.array(pDM) *atime / hubbleparam
    vDM = np.array(vDM) * np.sqrt(atime) 
    distances = dist2(pDM[:,0]-groupPos[0],pDM[:,1]-groupPos[1],pDM[:,2]-groupPos[2],boxSize) #Note this is distance SQUARED
    inGroupDM = np.where(distances<maxdist)[0]
    distances = 0. 
    tempvelDM = dx_wrap(vDM[inGroupDM]-groupVelocity, boxSizeVel)
    velMagDM = np.sqrt((tempvelDM*tempvelDM).sum(axis=1))
    lengroup = len(inGroupDM)
    massDM = lengroup* massDMParticle
    #Kinetic energy component
    kineticEnergyDM = np.sum(massDMParticle/2 *velMagDM*velMagDM*UnitVelocity_in_cm_per_s*UnitVelocity_in_cm_per_s)
    if 'inf' in str(kineticEnergyDM):
        chunk_size = 500  # Process in smaller batches
        kineticEnergyDM = 0.0
        #doing this way to elimatine infinite error
        for i in range(0, len(velMagDM), chunk_size):
            chunk = velMagDM[i:i + chunk_size]
            kineticEnergyDM += np.sum(massDMParticle /UnitMass_in_g/ 2 * chunk**2 * UnitVelocity_in_cm_per_s**2)   
        kineticEnergyDM = kineticEnergyDM*UnitMass_in_g #add back in the units
    potentialEnergyDM = 0
    potentialEnergyStarsDM = 0
    #DM self potential energy
    print("chunked DM potential calculation")
    potentialEnergyDM = chunked_potential_energy_same_mass(massDMParticle, pDM[inGroupDM],boxSize,chunk_size = 10000)
    print("chunked stars and dm potential calculation")
    potentialEnergyStarsDM = chunked_potential_energy_between_groups(massDMParticle, pDM[inGroupDM], starMass_inGroup,starPos_inGroup,boxSize,chunk_size = 10000)
    totEnergy = energyStars + kineticEnergyDM + potentialEnergyStarsDM + potentialEnergyDM
    if totEnergy<0:
         print("object is bound after DM!")
         boundedness =  1
    else:
         print("not bound even after DM!")
         boundedness = 0
    return boundedness, totEnergy, kineticEnergyDM, potentialEnergyStarsDM + potentialEnergyDM, massDM
    
def calc_dm_boundedness(energyStars,starVel_inGroup, starPos_inGroup, starMass_inGroup,  groupPos, groupVelocity,boxSize,boxSizeVel, pDM, vDM,groupRadius ,atime,massDMParticle):
     """
     Calculate the boundedness of a stellar group with dark matter without any tricks to reduce memory usage

     Parameters:
        energyStars (float): Total energy of the stellar group.
        starVel_inGroup (ndarray): Array of star velocities in the group.
        starPos_inGroup (ndarray): Array of star positions in the group.
        starMass_inGroup (ndarray): Array of star masses in the group.
        groupPos (ndarray): Position of the group.
        groupVelocity (ndarray): Velocity of the group.
        boxSize (float): Size of the simulation box for positions.
        boxSizeVel (float): Size of the simulation box for velocities.
        pDM (ndarray): Array of dark matter particle positions.
        vDM (ndarray): Array of dark matter particle velocities.
        groupRadius (float): Radius of the group.
        atime (float): Scale factor at the snapshot time.
        massDMParticle (float): Mass of a dark matter particle.

     Returns:
        tuple: Boundedness 
     """
     if groupRadius <= 0:
        distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
        maxdist = max(distances)
     else:
        maxdist = (groupRadius *atime /hubbleparam) **2
     pDM=  np.array(pDM) *atime / hubbleparam
     vDM = np.array(vDM) * np.sqrt(atime) 
     distances = dist2(pDM[:,0]-groupPos[0],pDM[:,1]-groupPos[1],pDM[:,2]-groupPos[2],boxSize) #Note this is distance SQUARED
     inGroupDM = np.where(distances<maxdist)[0]
     tempvelDM = dx_wrap(vDM[inGroupDM]-groupVelocity, boxSizeVel)
     velMagDM = np.sqrt((tempvelDM*tempvelDM).sum(axis=1))
     #Kinetic energy component
     kineticEnergyDM = np.sum(massDMParticle/2 *velMagDM*velMagDM*UnitVelocity_in_cm_per_s*UnitVelocity_in_cm_per_s)
     potentialEnergyDM = 0
     potentialEnergyStarsDM = 0
     lengroup = len(inGroupDM)
     massDM = lengroup* massDMParticle
     #DM self potential energy
     for i in range(lengroup):
         for j in range(i+1,lengroup):
            #r_ij = UnitLength_in_cm* np.linalg.norm(pDM[inGroupDM][i] - pDM[inGroupDM][j])  # Compute distance between mass i and mass j
            r_ij  = UnitLength_in_cm*np.sqrt(dist2_indv(pDM[inGroupDM][i,0]-pDM[inGroupDM][j,0],pDM[inGroupDM][i,1]-pDM[inGroupDM][j,1],pDM[inGroupDM][i,2]-pDM[inGroupDM][j,2],boxSize))
            if r_ij != 0:
                 potentialEnergyDM += -(GRAVITY_cgs * (massDMParticle)**2) / r_ij      
     lenstars= len(starMass_inGroup)
     #potential energy between stars and DM
     for i in range(lengroup):
         for j in range(lenstars):
            #r_ij = UnitLength_in_cm* np.linalg.norm(pDM[inGroupDM][i] - starPos_inGroup[j])  # Compute distance between mass i and mass j
            r_ij  = UnitLength_in_cm* np.sqrt(dist2_indv(pDM[inGroupDM][i,0]-starPos_inGroup[j,0],pDM[inGroupDM][i,1]-starPos_inGroup[j,1],pDM[inGroupDM][i,2]-starPos_inGroup[j,2],boxSize))
            if r_ij != 0:
                 potentialEnergyStarsDM += -(GRAVITY_cgs * massDMParticle *starMass_inGroup[j] ) / r_ij               
     totEnergy = energyStars + kineticEnergyDM + potentialEnergyStarsDM + potentialEnergyDM
     if totEnergy<0:
         print("object is bound after DM!")
         boundedness =  1
     else:
         print("not bound even after DM!")
         boundedness = 0
     return boundedness, totEnergy, kineticEnergyDM, potentialEnergyStarsDM + potentialEnergyDM, massDM

def calc_DM_mass(starPos_inGroup,  groupPos,boxSize, pDM,groupRadius ,atime,massDMParticle):
    """
    Calculate the mass of dark matter (DM) within a group.

    Parameters:
        starPos_inGroup (ndarray): Positions of stars in the group.
        groupPos (ndarray): Position of the center of the group.
        boxSize (float): Size of the simulation box for periodic boundary conditions.
        pDM (ndarray): Positions of dark matter particles.
        groupRadius (float): Radius of the group (physical or comoving).
        atime (float): Scale factor for the simulation.
        massDMParticle (float): Mass of each dark matter particle.
    
    Returns:
        massDM (float): Total mass of the dark matter particles within the group.
    """
    if len(pDM)==0:
        massDM = 0.
    else:
        if groupRadius <= 0:
            distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
            maxdist = max(distances)
        else:
            maxdist = (groupRadius  )**2
        pDM=  np.array(pDM) *atime / hubbleparam
        distances = dist2(pDM[:,0]-groupPos[0],pDM[:,1]-groupPos[1],pDM[:,2]-groupPos[2],boxSize) #Note this is distance SQUARED
        inGroupDM = np.where(distances<maxdist)[0]
        #Kinetic energy component
        lengroup = len(inGroupDM)
        massDM = lengroup* massDMParticle
    return massDM


def check_if_exists(filepath, idx,snapnum): 
    """
    Check if a file corresponding to a particular snapshot and chunk exists in the specified directory.

    Parameters:
        filepath (str): Path to the directory.
        idx (int): Chunk index.
        snapnum (int): Snapshot number.

    Returns:
        (bool): True if the file exists, False otherwise.
    """
    fof_process_name = "bounded_portion_"+str(snapnum)+"_chunk"+str(idx)+"_"
    exists = False
    for filename in os.listdir(filepath):
        # Check if the file exists in that directory
        if fof_process_name in filename:
            print("File has already been created!")
            print(filename)
            exists = True
    return exists

def check_if_exists_indv(filepath, idx,obj,snapnum): 
    """
    Check if an individual file corresponding to a particular snapshot, chunk, and object exists in the specified directory.

    Parameters:
        filepath (str): Path to the directory.
        idx (int): Chunk index.
        obj (int): Object identifier.
        snapnum (int): Snapshot number.

    Returns:
        exists (bool): True if the file exists, False otherwise.
    """
    fof_process_name = "indv_bounded_portion_"+str(snapnum)+"_chunk"+str(idx)+"_object"+str(obj)+"_"
    exists = False
    for filename in os.listdir(filepath):
        # Check if the file exists in that directory
        if fof_process_name in filename:
            print("File has already been created!")
            print(filename)
            exists = True
    return exists



def iterate_galaxies_chunked_resub_N_saveindv(N, gofilename,snapnum,atime, boxSize, halo100_indices, allStarMasses, allStarPositions,allStarVelocities, startAllStars,endAllStars, groupRadii,groupPos, groupVelocities,massDMParticle,halo_positions, startAllDM, endAllDM,dmsnap):
    """
    Iterate over galaxies in chunks, checking for boundedness and virialization, and calculating dark matter mass for each galaxy.
    Saves results for individual objects.

    Parameters:
        N (int): Index to start iterating over chunks.
        gofilename (str): Path for saving results.
        snapnum (int or str) Snapshot number for the simulation.
        atime (float): Scale factor for the current snapshot.
        boxSize (float): Size of the simulation box in comoving units.
        halo100_indices (Numpy.ndarray): Indices of halos containing 100 or more star particles.
        allStarMasses (Numpy.ndarray):  Array of stellar masses for all particles.
        allStarPositions (Numpy.ndarray): Array of stellar positions for all particles.
        allStarVelocities (Numpy.ndarray): Array of stellar velocities for all particles.
        startAllStars (Numpy.ndarray): Start indices of star particles for each galaxy.
        endAllStars (Numpy.ndarray): End indices of star particles for each galaxy.
        groupRadii (Numpy.ndarray):  Radii of the halos in the group catalog.
        groupPos (Numpy.ndarray): Positions of halos in the group catalog.
        groupVelocities (Numpy.ndarray): Velocities of halos in the group catalog.
        massDMParticle (float): Dark matter particle mass.
        halo_positions (Numpy.ndarray): Positions of dark matter particles.
        startAllDM (Numpy.ndarray):  Start indices of dark matter particles for each galaxy.
        endAllDM (Numpy.ndarray):  End indices of dark matter particles for each galaxy.
        dmsnap (int or str): Snapshot number dark matter primary

    Returns:
        objs (dict): Dictionary containing boundedness, virialization status, dark matter mass, star mass, recalculated radii, and virial ratios for each galaxy.
    """
    objs = {}
    #FIX THESE SO THEY AREN'T HARD CODED
    Omega0 = 0.27
    OmegaLambda = 0.73
    groupPos = groupPos *atime / hubbleparam 
    groupVelocities = groupVelocities /atime # convert to physical units
    #hubble flow correction
    boxSizeVel = boxSize * hubbleparam * .1 * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)
    boxSize = boxSize * atime/hubbleparam

    #break into chunks and go through objects in reverse order, saving as we go. 
    print("We'll need to do " +str(len(halo100_indices)/50.)+" chunks.")
    chunked_indices = list(chunks(halo100_indices, 50))
    chunked_radii = list(chunks(groupRadii,50))
    chunked_groupPos = list(chunks(groupPos,50))
    chunked_startAllStars = list(chunks(startAllStars,50))
    chunked_endAllStars = list(chunks(endAllStars,50))
    chunked_groupVelocities = list(chunks(groupVelocities,50))
    #Reversing because the earlier ones will take less time
    chunked_indices.reverse()
    chunked_radii.reverse()
    chunked_groupPos.reverse()
    chunked_startAllStars.reverse()
    chunked_endAllStars.reverse()
    chunked_groupVelocities.reverse()
    filepath = str(gofilename)+"/bounded3"
    print("checking in the following file")
    print(filepath)
    for chunkidx, chunk in enumerate(chunked_indices):
        objs = {}
        if int(chunkidx) >int(N) and int(chunkidx) <= int(int(N)+10) and check_if_exists(filepath, chunkidx,snapnum)==False: 
            print("file has not been created. continuing with the calculations! for index "+ str(chunkidx))
            print(f"There are {len(chunk)} objects in the chunk.")
            bounded = []
            virialized = []
            recalcRadii = []
            massesDM = []
            massStars= []
            virialRatios = []
            usedDMs = []
            massStar = []
            for i, j in enumerate(chunk):
                indv_filepath = filepath +"/indv_objs"
                print(f"Processing index {j} in chunk starting with {chunk[0]}")
                if check_if_exists_indv(indv_filepath, chunkidx, j,snapnum)==False:
                    groupRadii = chunked_radii[chunkidx]
                    groupPos = chunked_groupPos[chunkidx]
                    groupVelocities= chunked_groupVelocities[chunkidx]
                    startAllStars = chunked_startAllStars[chunkidx]
                    endAllStars = chunked_endAllStars[chunkidx]
                    r200group = groupRadii[i]
                    if r200group <=0:
                        print("no r200")
                    #print(i) 
                    boundedness = 0
                    virialization = 0
                    mass= 0
                    massDM =0 
                    usedDM = 0
                    virial_radius = -1
                    starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
                    starVel_inGroup = allStarVelocities[startAllStars[i]:endAllStars[i]]
                    starMass_inGroup = allStarMasses[startAllStars[i]:endAllStars[i]]
                    starMass_inGroup = np.array(starMass_inGroup) * UnitMass_in_g / hubbleparam #convert masses
                    starVel_inGroup = np.array(starVel_inGroup) * np.sqrt(atime) #unit conversions on the particle coordinates 
                    starPos_inGroup = np.array(starPos_inGroup) *atime / hubbleparam
                    #First, check for boundedness and virialization with just the stellar component
                    boundedness, energyStars, kineticEnergy, potentialEnergy, mass= chunked_calc_boundedness(starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos[i],groupVelocities[i],boxSize,boxSizeVel)
                    massStar = mass
                    virialization, virial_ratio = check_virialized(kineticEnergy, potentialEnergy)
                    pDM, vDM = nearby_DM(groupPos[i]/atime *hubbleparam, halo_positions, startAllDM, endAllDM,dmsnap, boxSize,limit =10. )
                    if boundedness ==0: 
                        #If that failed, try again when all DM within maximum star's distance is included   
                        print("doing the dm calculation")       
                        if len(pDM)==0:
                            #edge case where there's no DM
                            massDM = 0.
                            boundedness =0.
                            totEnergy = energyStars
                            kineticEnergyDM=0.
                            potentialEnergyDM = 0.
                        else:
                            boundedness,totEnergy, kineticEnergyDM, potentialEnergyDM, massDM = chunked_calc_dm_boundedness(energyStars,starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos[i],groupVelocities[i],boxSize,boxSizeVel,pDM, vDM,groupRadii[i],atime,massDMParticle) 
                        kineticEnergy += kineticEnergyDM
                        potentialEnergy += potentialEnergyDM
                        mass += massDM
                        #Check again for boundedness and virialization if the boundedness status has changed with this calculation
                        if boundedness ==1:
                            usedDM = 1 #this just tracks whether or not we used the DM to find virialization/boundedness
                            if virialization ==0: 
                                print("bounded, checking for virialized")
                                virialization,virial_ratio = check_virialized(kineticEnergy, potentialEnergy)
                    if virialization ==1:
                        #If it was found to be virialized by either means, calculate virial radius from R = - G M^2/U
                        virial_radius = calc_virial_radius(potentialEnergy,mass) #returns virial radius in cm 
                        print("virial radius is " + str(virial_radius/UnitLength_in_cm) +" kpc")
                        if massDM <=0:
                            #if we hadn't already calculated the DM mass, let's get it now using the virial radius
                            massDM = calc_DM_mass(starPos_inGroup,groupPos[i],boxSize, pDM,virial_radius/UnitLength_in_cm,atime,massDMParticle)
                    else: 
                        #otherwise, the radius will be Rmax
                        virial_radius = calc_max_radius(starPos_inGroup,groupPos[i],boxSize)
                        if massDM <= 0:
                            #if we hadn't calculated DM mass already, find DM within the maximum radius
                            massDM = calc_DM_mass(starPos_inGroup,groupPos[i],boxSize, pDM,virial_radius/UnitLength_in_cm,atime,massDMParticle)
                    print("DMmass is " + str(massDM))
                    bounded.append(boundedness)
                    massesDM.append(massDM)
                    massStars.append(massStar)
                    virialRatios.append(virial_ratio)
                    recalcRadii.append(virial_radius)
                    virialized.append(virialization)
                    usedDMs.append(usedDM)
                    indv_objs = {}
                    indv_objs['boundedness']= boundedness
                    indv_objs['massDM']= massDM
                    indv_objs['massStar']= massStar
                    indv_objs['virial_radius']= virial_radius
                    indv_objs['virialization']= virialization
                    indv_objs['virial_ratio']= virial_ratio
                    indv_objs['usedDM']= usedDM
                    print(f"Finished processing obj {j} in chunk starting with {chunk[0]}, saving progress...")
                    with open(gofilename+"/bounded3/indv_objs/indv_bounded_portion_"+str(snapnum)+"_chunk"+str(chunkidx)+"_object"+str(j)+"_startidx"+str(chunk[0])+"_V1.dat",'wb') as f:   
                        pickle.dump(indv_objs, f)
            objs['bounded'] = np.array(bounded)
            objs['virialized'] = np.array(virialized)
            objs['massDM'] = np.array(massesDM)
            objs['massStars'] = np.array(massStars)
            objs['recalcRadii'] = np.array(recalcRadii)
            objs['virialRatios'] = np.array(virialRatios)
            objs['usedDM']  = np.array(usedDMs)
            objs['r200'] = np.array(groupRadii) *atime /hubbleparam *UnitLength_in_cm #just for comparison, let's include the original group radius as calculated by the halo finder
            # print(f"Finished processing chunk starting with {chunk[0]}, saving progress...")
            # with open(gofilename+"/bounded2/bounded_portion_"+str(snapnum)+"_chunk"+str(chunkidx)+"_startidx"+str(chunk[0])+"_V1.dat",'wb') as f:   
            #     pickle.dump(objs, f)
    return objs

def iterate_galaxies_chunked_resub_N(N, gofilename,snapnum,atime, boxSize, halo100_indices, allStarMasses, allStarPositions,allStarVelocities, startAllStars,endAllStars, groupRadii,groupPos, groupVelocities,massDMParticle,halo_positions, startAllDM, endAllDM,dmsnap):
    """
    Iterate over galaxies in chunks, checking for boundedness and virialization, and calculating dark matter mass for each galaxy.
    Saves results for chunks of  objects.

    Parameters:
        N (int): Index to start iterating over chunks.
        gofilename (str): Path for saving results.
        snapnum (int or str) Snapshot number for the simulation.
        atime (float): Scale factor for the current snapshot.
        boxSize (float): Size of the simulation box in comoving units.
        halo100_indices (Numpy.ndarray): Indices of halos containing 100 or more star particles.
        allStarMasses (Numpy.ndarray):  Array of stellar masses for all particles.
        allStarPositions (Numpy.ndarray): Array of stellar positions for all particles.
        allStarVelocities (Numpy.ndarray): Array of stellar velocities for all particles.
        startAllStars (Numpy.ndarray): Start indices of star particles for each galaxy.
        endAllStars (Numpy.ndarray): End indices of star particles for each galaxy.
        groupRadii (Numpy.ndarray):  Radii of the halos in the group catalog.
        groupPos (Numpy.ndarray): Positions of halos in the group catalog.
        groupVelocities (Numpy.ndarray): Velocities of halos in the group catalog.
        massDMParticle (float): Dark matter particle mass.
        halo_positions (Numpy.ndarray): Positions of dark matter particles.
        startAllDM (Numpy.ndarray):  Start indices of dark matter particles for each galaxy.
        endAllDM (Numpy.ndarray):  End indices of dark matter particles for each galaxy.
        dmsnap (int or str): Snapshot number dark matter primary

    Returns:
        objs (dict): Dictionary containing boundedness, virialization status, dark matter mass, star mass, recalculated radii, and virial ratios for each galaxy.
    """
    objs = {}
    #FIX THESE SO THEY AREN'T HARD CODED
    Omega0 = 0.27
    OmegaLambda = 0.73
    groupPos = groupPos *atime / hubbleparam 
    groupVelocities = groupVelocities /atime # convert to physical units
    #hubble flow correction
    boxSizeVel = boxSize * hubbleparam * .1 * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)
    boxSize = boxSize * atime/hubbleparam

    #break into chunks and go through objects in reverse order, saving as we go. 
    print("We'll need to do " +str(len(halo100_indices)/50.)+" chunks.")
    chunked_indices = list(chunks(halo100_indices, 50))
    chunked_radii = list(chunks(groupRadii,50))
    chunked_groupPos = list(chunks(groupPos,50))
    chunked_startAllStars = list(chunks(startAllStars,50))
    chunked_endAllStars = list(chunks(endAllStars,50))
    chunked_groupVelocities = list(chunks(groupVelocities,50))
    #Reversing because the earlier ones will take less time
    chunked_indices.reverse()
    chunked_radii.reverse()
    chunked_groupPos.reverse()
    chunked_startAllStars.reverse()
    chunked_endAllStars.reverse()
    chunked_groupVelocities.reverse()
    filepath = str(gofilename)+"/bounded3"
    print("checking in the following file")
    print(filepath)
    for chunkidx, chunk in enumerate(chunked_indices):
        objs = {}
        if int(chunkidx) >int(N) and int(chunkidx) <= int(int(N)+10) and check_if_exists(filepath, chunkidx,snapnum)==False: 
            print("file has not been created. continuing with the calculations! for index "+ str(chunkidx))
            print(f"There are {len(chunk)} objects in the chunk.")
            bounded = []
            virialized = []
            recalcRadii = []
            massesDM = []
            massStars= []
            virialRatios = []
            usedDMs = []
            massStar = []
            for i, j in enumerate(chunk):
                print(f"Processing index {j} in chunk starting with {chunk[0]}")
                groupRadii = chunked_radii[chunkidx]
                groupPos = chunked_groupPos[chunkidx]
                groupVelocities= chunked_groupVelocities[chunkidx]
                startAllStars = chunked_startAllStars[chunkidx]
                endAllStars = chunked_endAllStars[chunkidx]
                r200group = groupRadii[i]
                if r200group <=0:
                    print("no r200")
                #print(i) 
                boundedness = 0
                virialization = 0
                mass= 0
                massDM =0 
                usedDM = 0
                virial_radius = -1
                starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
                starVel_inGroup = allStarVelocities[startAllStars[i]:endAllStars[i]]
                starMass_inGroup = allStarMasses[startAllStars[i]:endAllStars[i]]
                starMass_inGroup = np.array(starMass_inGroup) * UnitMass_in_g / hubbleparam #convert masses
                starVel_inGroup = np.array(starVel_inGroup) * np.sqrt(atime) #unit conversions on the particle coordinates 
                starPos_inGroup = np.array(starPos_inGroup) *atime / hubbleparam
                #First, check for boundedness and virialization with just the stellar component
                boundedness, energyStars, kineticEnergy, potentialEnergy, mass= chunked_calc_boundedness(starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos[i],groupVelocities[i],boxSize,boxSizeVel)
                massStar = mass
                virialization, virial_ratio = check_virialized(kineticEnergy, potentialEnergy)
                pDM, vDM = nearby_DM(groupPos[i]/atime *hubbleparam, halo_positions, startAllDM, endAllDM,dmsnap, boxSize,limit =10. )
                if boundedness ==0: 
                    #If that failed, try again when all DM within maximum star's distance is included   
                    print("doing the dm calculation")       
                    if len(pDM)==0:
                        #edge case where there's no DM
                        massDM = 0.
                        boundedness =0.
                        totEnergy = energyStars
                        kineticEnergyDM=0.
                        potentialEnergyDM = 0.
                    else:
                        boundedness,totEnergy, kineticEnergyDM, potentialEnergyDM, massDM = chunked_calc_dm_boundedness(energyStars,starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos[i],groupVelocities[i],boxSize,boxSizeVel,pDM, vDM,groupRadii[i],atime,massDMParticle) 
                    kineticEnergy += kineticEnergyDM
                    potentialEnergy += potentialEnergyDM
                    mass += massDM
                    #Check again for boundedness and virialization if the boundedness status has changed with this calculation
                    if boundedness ==1:
                        usedDM = 1 #this just tracks whether or not we used the DM to find virialization/boundedness
                        if virialization ==0: 
                            print("bounded, checking for virialized")
                            virialization,virial_ratio = check_virialized(kineticEnergy, potentialEnergy)
                if virialization ==1:
                    #If it was found to be virialized by either means, calculate virial radius from R = - G M^2/U
                    virial_radius = calc_virial_radius(potentialEnergy,mass) #returns virial radius in cm 
                    print("virial radius is " + str(virial_radius/UnitLength_in_cm) +" kpc")
                    if massDM <=0:
                        #if we hadn't already calculated the DM mass, let's get it now using the virial radius
                        massDM = calc_DM_mass(starPos_inGroup,groupPos[i],boxSize, pDM,virial_radius/UnitLength_in_cm,atime,massDMParticle)
                else: 
                    #otherwise, the radius will be Rmax
                    virial_radius = calc_max_radius(starPos_inGroup,groupPos[i],boxSize)
                    if massDM <= 0:
                        #if we hadn't calculated DM mass already, find DM within the maximum radius
                        massDM = calc_DM_mass(starPos_inGroup,groupPos[i],boxSize, pDM,virial_radius/UnitLength_in_cm,atime,massDMParticle)
                print("DMmass is " + str(massDM))
                bounded.append(boundedness)
                massesDM.append(massDM)
                massStars.append(massStar)
                virialRatios.append(virial_ratio)
                recalcRadii.append(virial_radius)
                virialized.append(virialization)
                usedDMs.append(usedDM)
            objs['bounded'] = np.array(bounded)
            objs['virialized'] = np.array(virialized)
            objs['massDM'] = np.array(massesDM)
            objs['massStars'] = np.array(massStars)
            objs['recalcRadii'] = np.array(recalcRadii)
            objs['virialRatios'] = np.array(virialRatios)
            objs['usedDM']  = np.array(usedDMs)
            objs['r200'] = np.array(groupRadii) *atime /hubbleparam *UnitLength_in_cm #just for comparison, let's include the original group radius as calculated by the halo finder
            print(f"Finished processing chunk starting with {chunk[0]}, saving progress...")
            with open(gofilename+"/bounded3/bounded_portion_"+str(snapnum)+"_chunk"+str(chunkidx)+"_startidx"+str(chunk[0])+"_V1.dat",'wb') as f:   
                pickle.dump(objs, f)
    return objs




def add_bounded_calculation_N(filename,N, snapnum, group = "Stars"):
    """
    Wrapper function to calculate boundedness and virialization for galaxy groups.
    
    Parameters:
        filename (str): Path to the input files.
        N (int): Index to start iterating over chunks.
        snapnum (int or str): Snapshot number for the simulation.
        group (str): Group type to calculate boundedness for ('Stars', 'Gas', or 'DM').
        
    Returns:
        dict: Dictionary containing the boundedness, virialization status, dark matter mass, star mass, recalculated radii, and virial ratios for each galaxy group.
    """
    print('opening files')
    gofilename = str(filename)
    SV = gofilename[-4:]
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, redshift, massDMParticle = get_headerprops(snap)
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
        print("Not supported!")
    print("Loading star particles")
    _,allStarMasses, allStarPositions, allStarVelocities= get_starIDs(snap)
    #_,allDMPositions, allDMVelocities= get_DMIDs(snap)
    print("Setting up halo catalogue for faster DM association")
    DMhalo_indices, halo_positions, startAllDM, endAllDM, dmsnap = set_up_DM(SV, snapnum)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    halo100_rad = get_GroupRadii(cat, halo100_indices)
    halo100_vel = get_GroupVel(cat,halo100_indices)
    atime = 1./(1.+redshift)
    print("calculating boundedness for all objects")
    objs = iterate_galaxies_chunked_resub_N(N, str(filename),snapnum, atime, boxSize, halo100_indices,allStarMasses, allStarPositions,allStarVelocities, startAllStars,endAllStars, halo100_rad,halo100_pos, halo100_vel,massDMParticle* UnitMass_in_g / hubbleparam,halo_positions, startAllDM, endAllDM, dmsnap)
    return objs

def add_bounded_calculation_N_indv(filename,N, snapnum, group = "Stars"):
    """
    Wrapper function to calculate boundedness and virialization and save individual galaxy groups.
    
    Parameters:
        filename (str): Path to the input files.
        N (int): Index to start iterating over chunks.
        snapnum (int or str): Snapshot number for the simulation.
        group (str): Group type to calculate boundedness for ('Stars', 'Gas', or 'DM').
    
    Returns:
        dict: Dictionary containing the boundedness, virialization status, dark matter mass, star mass, recalculated radii, and virial ratios for each galaxy group.
    """
    print('opening files')
    gofilename = str(filename)
    SV = gofilename[-4:]
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, redshift, massDMParticle = get_headerprops(snap)
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
        print("Not supported!")
    print("Loading star particles")
    _,allStarMasses, allStarPositions, allStarVelocities= get_starIDs(snap)
    #_,allDMPositions, allDMVelocities= get_DMIDs(snap)
    print("Setting up halo catalogue for faster DM association")
    DMhalo_indices, halo_positions, startAllDM, endAllDM, dmsnap = set_up_DM(SV, snapnum)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    halo100_rad = get_GroupRadii(cat, halo100_indices)
    halo100_vel = get_GroupVel(cat,halo100_indices)
    atime = 1./(1.+redshift)
    print("calculating boundedness for all objects")
    objs = iterate_galaxies_chunked_resub_N_saveindv(N, str(filename),snapnum, atime, boxSize, halo100_indices,allStarMasses, allStarPositions,allStarVelocities, startAllStars,endAllStars, halo100_rad,halo100_pos, halo100_vel,massDMParticle* UnitMass_in_g / hubbleparam,halo_positions, startAllDM, endAllDM, dmsnap)
    return objs


def boundedness_mode(filename,N, snapnum, mode = "group", group = "Stars"):
    """
    Starts full boundedness calculation for either save as group or individual

    Parameters: 
        filename (str): path to FOF files
        N (int): chunk at which to start the calculation
        snapnum (int or str): hdf5 snap number
        mode (str): mode for saving output
            Options: 
                Default: "group" - saves the group in chunks
                "indv": saves each individual object. Requires a directory named "indv_objs" in bounded directory. 
    
    Returns :
        dict: all object properties. 
    
    """
    if mode =="group":
        print("group mode")
        objs= add_bounded_calculation_N(filename,N, snapnum, group = group)
    elif mode =="indv":
        #run in individual mode
        print("individual mode, saving individual objects")
        objs = add_bounded_calculation_N_indv(filename,N, snapnum, group = group)
    else: 
        print(f"ModeError: mode {mode} is unsupported, sorry.")
    return objs

if __name__=="__main__":
    """
    Routine if running as a script

    Parameters: 
        gofilename path to directory containing groupordered file + fof table
        # foffilename 
        snapnum (float)
    """
    script, gofilename, snapnum = argv
    # with open("/home/x-cwilliams/FOF_calculations/newstars_Sig2_25Mpc.dat",'rb') as f:
    # 	newstars = pickle.load(f,encoding = "latin1")
    #Testing
    N=85
    add_bounded_calculation_N_indv( gofilename, N, snapnum)
    # add_bounded_calculation_N( gofilename, N, snapnum)
    # objs = add_bounded_calculation(gofilename, snapnum)

    # with open(gofilename+"/stellar_rotation_"+str(snapnum)+"_V1.dat",'wb') as f:   
    #     pickle.dump(objs, f)
    # with open("/u/scratch/c/clairewi/test_stellar_rotation_"+str(snapnum)+"_V2.dat",'wb') as f:   
    #     pickle.dump(objs, f)
    # print("SAVED OUTPUT!")
