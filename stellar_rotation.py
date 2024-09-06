import h5py 
import numpy as np
from sys import argv
import pickle
import sys 
sys.path.append('/home/x-cwilliams/FOF_calculations/process-fof')
from fof_process import get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config,get_gasGroups, get_cosmo_props,get_starIDgroups, get_Halos

UnitMass_in_g = 1.989e43     
UnitLength_in_cm = 3.085678e21 
hubbleparam = .71 #hubble constant
GRAVITY_cgs = 6.672e-8
UnitVelocity_in_cm_per_s = 1.0e5

def dx_wrap(dx,box):
	#wraps to account for period boundary conditions. This mutates the original entry
	idx = dx > +box/2.0
	dx[idx] -= box
	idx = dx < -box/2.0
	dx[idx] += box 
	return dx

# def dx_vec(dx,dy,dz,box):
#      return np.stack((dx_wrap(dx,box),dx_wrap(dy,box),dx_wrap(dz,box)), axis =1)

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
    allStarMasses = f['PartType4/Masses']
    allStarPositions = f['PartType4/Coordinates']
    allStarVelocities = f['PartType4/Velocities']
    return allStarIDs, allStarMasses, allStarPositions, allStarVelocities


def get_gasIDs(f):
	"""
	Get particle IDs (groupordered snap)
	"""
	allGasIDs = f['PartType0/ParticleIDs']
	allGasMasses = f['PartType0/Masses']
	allGasVelocities = f['PartType0/Velocities']
	allGasPositions = f['PartType0/Coordinates']

	return allGasIDs,allGasMasses, allGasPositions, allGasVelocities

#functions to calculate various bulk rotational properties (they are weird for memory reasons)
def calc_vrms(mtot,velmagstars, starMass_inGroup ):
     velmag2stars = np.array([x**2 for x in velmagstars])
     vrms_argument = np.array([velmag2stars[i]*starMass_inGroup[i] for i in range(len(velmag2stars))])
     return np.sqrt(np.sum(vrms_argument)/mtot)

def calc_vrad(mtot,v_radscalari, starMass_inGroup ):
    vrad_argument = np.array([v_radscalari[i]*starMass_inGroup[i] for i in range(len(v_radscalari))])
    return np.sum(vrad_argument)/mtot

def calc_vrot(mtot,v_rotmagi, starMass_inGroup ):
     vrot_argument = np.array([v_rotmagi[i] * v_rotmagi[i] * starMass_inGroup[i] for i in range(len(v_rotmagi))])
     return np.sqrt(np.sum(vrot_argument)/mtot)

def calc_vturb(mtot, v_turbmagi,starMass_inGroup):
     vturb_argument = np.array([v_turbmagi[i] * v_turbmagi[i] * starMass_inGroup[i] for i in range(len(v_turbmagi))])
     return np.sqrt(np.sum(vturb_argument)/mtot)

def calc_iscalar(tempradstars,lunitvec,starMass_inGroup):
     iscalar_cross = np.cross(tempradstars,lunitvec)
     iscalar_argument = np.array([np.linalg.norm(iscalar_cross[i])**2 * starMass_inGroup[i] for i in range(len(iscalar_cross))])
     return np.sum(iscalar_argument)

def calc_stellar_rotation(starMass_inGroup,starVel_inGroup,starPos_inGroup, groupPos,groupVelocity,boxSize,boxSizeVel):
    """
    Calculate rotation curve of stellar component - 25 steps
    """
    #setup the vectors
    tempvelstars = dx_wrap(starVel_inGroup-groupVelocity, boxSizeVel) #velocity wrt group velocity
    velmagstars =  np.linalg.norm(tempvelstars, axis=1)[:,np.newaxis] #magnitude of velocity
    tempradstars = dx_wrap(starPos_inGroup-groupPos,boxSize) #radius wrt group position
    distances=  np.linalg.norm(tempradstars, axis=1)
    rmagstars = distances[:,np.newaxis] #magnitude of radius (distance)
    epsilon = 1e-2 #introduce an error of 1e-2 cm to avoid runtime warnings 
    rmagstars[rmagstars == 0] = epsilon
    runitstars = tempradstars / rmagstars #unit vector in direction of radius

    #Overall group quantities
    lvec = np.sum((np.cross(tempradstars,tempvelstars)*starMass_inGroup[:,np.newaxis]), axis = 0) #angular momentum
    lmag = np.linalg.norm(lvec) #magnitude of angular momentum
    lunitvec = lvec/ lmag #unit vector in direction of L
    #iscalar = np.sum(np.linalg.norm(np.cross(tempradstars,lunitvec),axis = 1)**2*starMass_inGroup[:,np.newaxis]) #moment of inertia
    iscalar = calc_iscalar(tempradstars, lunitvec,starMass_inGroup)
    omegavec = lvec/iscalar

    #calculate more individual vectors 
    v_rotveci = np.cross(tempradstars, omegavec) #rotational velocity
    v_rotmagi = np.linalg.norm(v_rotveci, axis=1)[:,np.newaxis] #rotational velocity magnitude
    v_radveci = runitstars * np.sum(runitstars * tempvelstars, axis =1)[:,np.newaxis] #radial velocity
    v_radscalari = np.sum(runitstars * tempvelstars, axis =1)[:,np.newaxis] #radial velocity magnitude
    v_turbveci = tempvelstars - v_rotveci - v_radveci #vector turbulent velocity
    v_turbmagi = np.linalg.norm(v_turbveci,axis = 1)[:,np.newaxis] #magnitude turbulent velocity

    #bulk velocities for the halo - doing this weirdly to get around memory limit problems with array multiplication
    mtot = np.sum(starMass_inGroup)
    #total rms velocity
    v_rmstot = calc_vrms(mtot,velmagstars, starMass_inGroup)
    #total radial velocity
    v_radtot = calc_vrad(mtot,v_radscalari, starMass_inGroup )
    #total rotational velocity
    # v_rottot = np.sqrt(np.sum(starMass_inGroup * (np.linalg.norm(v_rotveci, axis=1)[:,np.newaxis])**2)/mtot)
    v_rottot = calc_vrot(mtot,v_rotmagi,starMass_inGroup)
    #turbulent velocity 
    v_turbtot = calc_vturb(mtot, v_turbmagi, starMass_inGroup)

    #Calculate velocity dispersion of galaxy
    #velDispStars = np.sqrt(np.sum((velMagStars - np.mean(velMagStars))**2))/np.size(velMagStars) #velocity dispersion of magnitudes (not projected along a LOS)
    inner_rad = min(rmagstars)
    outer_rad = max(rmagstars)
    step = (outer_rad-inner_rad)/25
    radius = inner_rad
    rotation_curve_rms = []
    rotation_curve_rad = []
    rotation_curve_rot = []
    rotation_curve_turb = []
    radii = []
    while radius < outer_rad:
         shell_idx = np.where(distances<radius)[0]
         if len(shell_idx)>0: #only use shells containing star particles
            vel_inShell = velmagstars[shell_idx]
            vel_rad_inShell = v_radscalari[shell_idx]
            vel_rot_inShell = v_rotmagi[shell_idx]
            vel_turb_inShell = v_turbmagi[shell_idx]
            mass_inShell = starMass_inGroup[shell_idx]
            mshell = np.sum(mass_inShell)
            mask = np.ones(distances.shape, dtype='bool')
            mask[shell_idx] = False #Remove the used particles
            distances = distances[mask] #next shell we'll only search the unused DM particles
            velmagstars = velmagstars[mask]
            v_radscalari = v_radscalari[mask]
            v_rotmagi=v_rotmagi[mask]
            v_turbmagi=v_turbmagi[mask]
            starMass_inGroup = starMass_inGroup[mask]
            #velocity = sum(vel_inShell)/len(vel_inShell) #average velocity in shell
            velocity = calc_vrms(mshell,vel_inShell,mass_inShell)
            velocity_rad = calc_vrad(mshell,vel_rad_inShell, mass_inShell )
            velocity_rot = calc_vrot(mshell, vel_rot_inShell,mass_inShell)
            velocity_turb = calc_vturb(mshell, vel_turb_inShell, mass_inShell)
            rotation_curve_rms.append(velocity)
            rotation_curve_rad.append(velocity_rad)
            rotation_curve_rot.append(velocity_rot)
            rotation_curve_turb.append(velocity_turb)
            radii.append(radius)
         else: 
              #Case with empty shell
              velocity = 0.
         radius = radius + step
    return rotation_curve_rms,rotation_curve_rad,rotation_curve_rot, rotation_curve_turb,radii,v_rmstot,v_radtot,v_rottot, v_turbtot
     

def iterate_galaxies(atime, boxSize, halo100_indices, allStarMasses, allStarPositions,allStarVelocities, startAllStars,endAllStars, groupRadii,groupPos, groupVelocities):
    """
    iterate all the galaxies in the FOF and find their rotation curves.
    Perform all the unit conversions.
    """
    objs = {}
    hubbleparam= 0.71 #FIX THESE SO THEY AREN'T HARD CODED
    Omega0 = 0.27
    OmegaLambda = 0.71
    groupPos = groupPos *atime / hubbleparam 
    groupVelocities = groupVelocities  /atime # convert to physical units
    # groupPos = groupPos * UnitLength_in_cm*atime / hubbleparam 
    # groupVelocities = groupVelocities * UnitVelocity_in_cm_per_s /atime # convert to physical units
    rotation_rms = []
    rotation_rad = []
    rotation_rot = []
    rotation_turb = []
    v_rms = []
    v_rot = []
    v_rad = []
    v_turb = []
    radii = []
    dispersions = []
    #hubble flow correction
    boxSizeVel = boxSize * hubbleparam * .1 * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)
    # boxSizeVel = boxSize * UnitVelocity_in_cm_per_s* hubbleparam * .1 * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)
    # boxSize = boxSize * UnitLength_in_cm* atime/hubbleparam
    boxSize = boxSize * atime/hubbleparam
    for i,j in enumerate(halo100_indices):
        print(i) 
        starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
        starVel_inGroup = allStarVelocities[startAllStars[i]:endAllStars[i]]
        starMass_inGroup = allStarMasses[startAllStars[i]:endAllStars[i]]
        starMass_inGroup = np.array(starMass_inGroup)   / hubbleparam #convert masses
        starVel_inGroup = np.array(starVel_inGroup) *  np.sqrt(atime) #unit conversions on the particle coordinates 
        starPos_inGroup = np.array(starPos_inGroup) * atime / hubbleparam
        # starMass_inGroup = np.array(starMass_inGroup)  * UnitMass_in_g / hubbleparam #convert masses
        # starVel_inGroup = np.array(starVel_inGroup) * UnitVelocity_in_cm_per_s * np.sqrt(atime) #unit conversions on the particle coordinates 
        # starPos_inGroup = np.array(starPos_inGroup) *UnitLength_in_cm *atime / hubbleparam
        rotation_curve_rms,rotation_curve_rad,rotation_curve_rot, rotation_curve_turb,rotation_radii,v_rmstot,v_radtot,v_rottot, v_turbtot  = calc_stellar_rotation(starMass_inGroup,starVel_inGroup,starPos_inGroup, groupPos[i],groupVelocities[i],boxSize,boxSizeVel)
        rotation_rms.append(rotation_curve_rms)
        rotation_rad.append(rotation_curve_rad)
        rotation_rot.append(rotation_curve_rot)
        rotation_turb.append(rotation_curve_turb)
        radii.append(rotation_radii)
        v_rms.append(v_rmstot)
        v_rad.append(v_radtot)
        v_rot.append(v_rottot)
        v_turb.append(v_turbtot)
    # objs['rot_curves'] = np.array(rotation,dtype=object)
    objs['rot_radii'] =np.array(radii,dtype=object)
    # objs['vel_dispersion'] = np.array(dispersions)
    objs['rotation_curve_rms'] = rotation_rms
    objs['rotation_curve_rad'] = rotation_rad
    objs['rotation_curve_rot'] = rotation_rot
    objs['rotation_curve_turb'] = rotation_turb
    objs['v_rms'] = v_rms
    objs['v_rad'] = v_rad
    objs['v_rot'] = v_rot
    objs['v_turb'] = v_turb
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
        print("used groups of 300 or more DM!")
        halo100_indices = get_Halos(cat)
    print("Loading star particles")
    # TESTING MODE: UNCOMMENT below!!
    #halo100_indices = halo100_indices[-20:-1]
    print(len(halo100_indices))
    _, allStarMasses, allStarPositions, allStarVelocities= get_starIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    halo100_rad = get_GroupRadii(cat, halo100_indices)
    halo100_vel = get_GroupVel(cat,halo100_indices)
    atime = 1./(1.+redshift)
    print("calculating rotation curves for all objects")
    objs = iterate_galaxies(atime, boxSize, halo100_indices, allStarMasses, allStarPositions,allStarVelocities, startAllStars,endAllStars, halo100_rad,halo100_pos, halo100_vel)
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
        print("used groups of 300 or more DM!")
        halo100_indices = get_Halos(cat)
    print("Loading star particles")
    # TESTING MODE: UNCOMMENT below!!
    #halo100_indices = halo100_indices[-20:-1]
    _,allGasMasses, allGasPositions, allGasVelocities= get_gasIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    halo100_rad = get_GroupRadii(cat, halo100_indices)
    halo100_vel = get_GroupVel(cat,halo100_indices)
    atime = 1./(1.+redshift)
    print("calculating rotation curves for all objects")
    objs = iterate_galaxies(atime, boxSize, halo100_indices, allGasMasses,allGasPositions,allGasVelocities, startAllStars,endAllStars, halo100_rad,halo100_pos, halo100_vel)
    print(objs)
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
    # with open(gofilename+"/test_stellar_rotation_"+str(snapnum)+"_v2.dat",'wb') as f:   
    #     pickle.dump(objs, f)
    # with open("/anvil/projects/x-ast180056/FOF_project/test_stellar_rotation_"+str(snapnum)+"_v2.dat",'wb') as f:   
    #     pickle.dump(objs, f)
    #print("SAVED OUTPUT!")




    # velMagStars = np.sqrt((tempvelstars*tempvelstars).sum(axis=1))
    # print(velMagStars)
    # distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
    # #Calculate velocity dispersion of galaxy
    # velDispStars = np.sqrt(np.sum((velMagStars - np.mean(velMagStars))**2))/np.size(velMagStars) #velocity dispersion of magnitudes (not projected along a LOS)
    # inner_rad = min(distances)
    # outer_rad = max(distances)
    # step = (outer_rad-inner_rad)/25
    # radius = inner_rad
    # rotation_curve = []
    # radii = []
    # while radius < outer_rad:
    #      shell_idx = np.where(distances<radius)[0]
    #      if len(shell_idx)>0: #only use shells containing star particles
    #         vel_inShell = velMagStars[shell_idx]
    #         mask = np.ones(distances.shape, dtype='bool')
    #         mask[shell_idx] = False #Remove the used particles
    #         distances = distances[mask] #next shell we'll only search the unused DM particles
    #         velMagStars = velMagStars[mask]
    #         velocity = sum(vel_inShell)/len(vel_inShell) #average velocity in shell
    #         rotation_curve.append(velocity)
    #         radii.append(radius)
    #      else: 
    #           #Case with empty shell
    #           velocity = 0.
    #      radius = radius + step
    # return rotation_curve, radii, velDispStars