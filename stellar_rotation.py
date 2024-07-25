import h5py 
import numpy as np
from sys import argv
import pickle
import sys 
sys.path.append('/home/x-cwilliams/FOF_calculations/process-fof')
from fof_process import get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config,get_gasGroups, get_cosmo_props,get_starIDgroups

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
    allStarPositions = f['PartType4/Coordinates']
    allStarVelocities = f['PartType4/Velocities']
    return allStarIDs, allStarPositions, allStarVelocities

def get_gasIDs(f):
	"""
	Get particle IDs (groupordered snap)
	"""
	allGasIDs = f['PartType0/ParticleIDs']
	allGasVelocities = f['PartType0/Velocities']
	allGasPositions = f['PartType0/Coordinates']

	return allGasIDs, allGasPositions, allGasVelocities

def calc_stellar_rotation(starVel_inGroup,starPos_inGroup, groupPos,groupVelocity,boxSize,boxSizeVel):
    """
    Calculate rotation curve of stellar component - 25 steps
    """
    tempvelstars = dx_wrap(starVel_inGroup-groupVelocity, boxSizeVel)
    velMagStars = np.sqrt((tempvelstars*tempvelstars).sum(axis=1))
    distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
    #Calculate velocity dispersion of galaxy
    velDispStars = np.sqrt(np.sum((velMagStars - np.mean(velMagStars))**2))/np.size(velMagStars) #velocity dispersion of magnitudes (not projected along a LOS)
    inner_rad = min(distances)
    outer_rad = max(distances)
    step = (outer_rad-inner_rad)/25
    radius = inner_rad
    rotation_curve = []
    radii = []
    while radius < outer_rad:
         shell_idx = np.where(distances<radius)[0]
         if len(shell_idx)>0: #only use shells containing star particles
            vel_inShell = velMagStars[shell_idx]
            mask = np.ones(distances.shape, dtype='bool')
            mask[shell_idx] = False #Remove the used particles
            distances = distances[mask] #next shell we'll only search the unused DM particles
            velMagStars = velMagStars[mask]
            velocity = sum(vel_inShell)/len(vel_inShell) #average velocity in shell
            rotation_curve.append(velocity)
            radii.append(radius)
         else: 
              #Case with empty shell
              velocity = 0.
         radius = radius + step
    return rotation_curve, radii, velDispStars
     

def iterate_galaxies(atime, boxSize, halo100_indices, allStarPositions,allStarVelocities, startAllStars,endAllStars, groupRadii,groupPos, groupVelocities):
    """
    iterate all the galaxies in the FOF and find their rotation curves
    """
    objs = {}
    hubbleparam= 0.71 #FIX THESE SO THEY AREN'T HARD CODED
    Omega0 = 0.27
    OmegaLambda = 0.71
    groupPos = groupPos *atime / hubbleparam 
    groupVelocities = groupVelocities /atime # convert to physical units
    rotation = []
    radii = []
    dispersions = []
    #hubble flow correction
    boxSizeVel = boxSize * hubbleparam * .1 * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)
    boxSize = boxSize * atime/hubbleparam
    for i,j in enumerate(halo100_indices):
        print(i) 
        starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
        starVel_inGroup = allStarVelocities[startAllStars[i]:endAllStars[i]]
        starVel_inGroup = np.array(starVel_inGroup) * np.sqrt(atime) #unit conversions on the particle coordinates 
        starPos_inGroup = np.array(starPos_inGroup) *atime / hubbleparam
        stellar_rotation_curve, rotation_radii, dispersion = calc_stellar_rotation(starVel_inGroup,starPos_inGroup, groupPos[i],groupVelocities[i],boxSize,boxSizeVel)
        rotation.append(stellar_rotation_curve)
        radii.append(rotation_radii)
        dispersions.append(dispersion)
    objs['rot_curves'] = np.array(rotation,dtype=object)
    objs['rot_radii'] =np.array(radii,dtype=object)
    objs['vel_dispersion'] = np.array(dispersions)
    return objs

def add_rotation_curves(filename, snapnum, group = "Stars"):
    """
    wrapper function
    """
    print('opening files')
    gofilename = str(filename)
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, redshift, _ = get_headerprops(snap)
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
    # TESTING MODE: UNCOMMENT below!!
    #halo100_indices = halo100_indices[-20:-1]
    _,allStarPositions, allStarVelocities= get_starIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    halo100_rad = get_GroupRadii(cat, halo100_indices)
    halo100_vel = get_GroupVel(cat,halo100_indices)
    atime = 1./(1.+redshift)
    print("calculating rotation curves for all objects")
    objs = iterate_galaxies(atime, boxSize, halo100_indices, allStarPositions,allStarVelocities, startAllStars,endAllStars, halo100_rad,halo100_pos, halo100_vel)
    return objs


def add_rotation_curves_gas(filename, snapnum, group = "Stars"):
    """
    wrapper function - gas verison!!!!!!!! calculates for gas!!
    """
    print('opening files')
    gofilename = str(filename)
    gofilename, foffilename = set_snap_directories(gofilename, snapnum, foffilename = str(gofilename))
    snap, fof = open_hdf5(gofilename, foffilename)
    boxSize, redshift, _ = get_headerprops(snap)
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
    # TESTING MODE: UNCOMMENT below!!
    #halo100_indices = halo100_indices[-20:-1]
    _,allGasPositions, allGasVelocities= get_gasIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    halo100_rad = get_GroupRadii(cat, halo100_indices)
    halo100_vel = get_GroupVel(cat,halo100_indices)
    atime = 1./(1.+redshift)
    print("calculating rotation curves for all objects")
    objs = iterate_galaxies(atime, boxSize, halo100_indices, allGasPositions,allGasVelocities, startAllStars,endAllStars, halo100_rad,halo100_pos, halo100_vel)
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
    # 	newstars = pickle.load(f,encoding = "latin1")
    objs = add_rotation_curves_gas(gofilename, snapnum)
    print(objs['vel_dispersion'])
    print(objs['rot_curves'])
    print(objs['rot_radii'])
    with open(gofilename+"/test_stellar_rotation_"+str(snapnum)+"_v2.dat",'wb') as f:   
        pickle.dump(objs, f)
    # with open("/anvil/projects/x-ast180056/FOF_project/test_stellar_rotation_"+str(snapnum)+"_v2.dat",'wb') as f:   
    #     pickle.dump(objs, f)
    print("SAVED OUTPUT!")
