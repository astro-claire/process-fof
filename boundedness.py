import h5py 
import numpy as np
from sys import argv
import pickle
import sys 
sys.path.append('/u/home/c/clairewi/project-snaoz/FOF_Testing/process-fof')
from fof_process import get_starGroups, set_snap_directories, open_hdf5, get_headerprops, set_subfind_catalog, set_config,get_gasGroups, get_cosmo_props,get_starIDgroups, get_headerprops

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

# def dx_indv(delta, box):
#     new_delta = []
#     for di in delta:
#         if di > +box/2.0:
#             di -= box
#         if di < -box/2.0:
#             di += box
#     new_delta.append(di)
#     return np.array(new_delta)

def dx_indv(dx, box):
    if dx > +box/2.0:
        dx -= box
    if dx < -box/2.0:
        dx += box 
    return dx

def dist2(dx,dy,dz,box):
	#Calculates distance taking into account periodic boundary conditions
	return dx_wrap(dx,box)**2 + dx_wrap(dy,box)**2 + dx_wrap(dz,box)**2

def dist2_indv(dx,dy,dz,box):
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
    ratio = np.abs(potentialEnergy/kineticEnergy)
    virialized = 0
    if 1.5<ratio<2.5:
        virialized = 1
    return virialized, ratio

def calc_virial_radius(potentialEnergy, mass):
    return - GRAVITY_cgs * mass * mass / potentialEnergy

def calc_max_radius(starPos_inGroup,groupPos,boxSize):
    distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
    maxdist = max(distances)
    return UnitLength_in_cm* np.sqrt(maxdist)


def calc_boundedness(starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos,groupVelocity,boxSize,boxSizeVel):
    """
    Calculate boundedness
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

    
def calc_dm_boundedness(energyStars,starVel_inGroup, starPos_inGroup, starMass_inGroup,  groupPos, groupVelocity,boxSize,boxSizeVel, pDM, vDM,groupRadius ,atime,massDMParticle):
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
     print("DMmass is " + str(massDM))
     #DM self potential energy
     for i in range(lengroup):
         for j in range(i+1,lengroup):
            #r_ij = UnitLength_in_cm* np.linalg.norm(pDM[inGroupDM][i] - pDM[inGroupDM][j])  # Compute distance between mass i and mass j
            r_ij  = UnitLength_in_cm*np.sqrt(dist2_indv(pDM[inGroupDM][i,0]-pDM[inGroupDM][j,0],pDM[inGroupDM][i,1]-pDM[inGroupDM][j,1],pDM[inGroupDM][i,2]-pDM[inGroupDM][j,2],boxSize))
            if r_ij != 0:
                 potentialEnergyDM += -(GRAVITY_cgs * massDMParticle**2) / r_ij      
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
         print("object is bound")
         boundedness =  1
     else:
         print("not bound even after DM!")
         boundedness = 0
     return boundedness, totEnergy, kineticEnergyDM, potentialEnergyStarsDM + potentialEnergyDM, massDM

def calc_DM_mass(starPos_inGroup,  groupPos,boxSize, pDM,groupRadius ,atime,massDMParticle):
    if groupRadius <= 0:
        distances = dist2(starPos_inGroup[:,0]-groupPos[0],starPos_inGroup[:,1]-groupPos[1],starPos_inGroup[:,2]-groupPos[2],boxSize)
        maxdist = max(distances)
    else:
        maxdist = (groupRadius *atime /hubbleparam )**2
    pDM=  np.array(pDM) *atime / hubbleparam
    distances = dist2(pDM[:,0]-groupPos[0],pDM[:,1]-groupPos[1],pDM[:,2]-groupPos[2],boxSize) #Note this is distance SQUARED
    inGroupDM = np.where(distances<maxdist)[0]
    #Kinetic energy component
    lengroup = len(inGroupDM)
    massDM = lengroup* massDMParticle
    return massDM

          
def iterate_galaxies(atime, boxSize, halo100_indices, allStarMasses, allStarPositions,allStarVelocities, allDMPositions, allDMVelocities, startAllStars,endAllStars, groupRadii,groupPos, groupVelocities,massDMParticle):
    """
    iterate all the galaxies in the FOF and find their rotation curves
    """
    objs = {}
    #FIX THESE SO THEY AREN'T HARD CODED
    Omega0 = 0.27
    OmegaLambda = 0.71
    groupPos = groupPos *atime / hubbleparam 
    groupVelocities = groupVelocities /atime # convert to physical units
    #hubble flow correction
    boxSizeVel = boxSize * hubbleparam * .1 * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)
    boxSize = boxSize * atime/hubbleparam
    bounded = []
    virialized = []
    recalcRadii = []
    massesDM = []
    virialRatios = []
    for i,j in enumerate(halo100_indices):
        r200group = groupRadii[i]
        if r200group <=0:
             print("no r200")
        #print(i) 
        boundedness = 0
        virialization = 0
        mass= 0
        massDM =0 
        virial_radius = -1
        starPos_inGroup = allStarPositions[startAllStars[i]:endAllStars[i]]
        starVel_inGroup = allStarVelocities[startAllStars[i]:endAllStars[i]]
        starMass_inGroup = allStarMasses[startAllStars[i]:endAllStars[i]]
        starMass_inGroup = np.array(starMass_inGroup) * UnitMass_in_g / hubbleparam #convert masses
        starVel_inGroup = np.array(starVel_inGroup) * np.sqrt(atime) #unit conversions on the particle coordinates 
        starPos_inGroup = np.array(starPos_inGroup) *atime / hubbleparam
        print(i)
        boundedness, energyStars, kineticEnergy, potentialEnergy, mass= calc_boundedness(starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos[i],groupVelocities[i],boxSize,boxSizeVel)
        virialization, virial_ratio = check_virialized(kineticEnergy, potentialEnergy)
        if boundedness ==0:    
             print("doing the dm calculation")       
             boundedness,totEnergy, kineticEnergyDM, potentialEnergyDM, massDM = calc_dm_boundedness(energyStars,starVel_inGroup,starPos_inGroup,starMass_inGroup, groupPos[i],groupVelocities[i],boxSize,boxSizeVel,allDMPositions, allDMVelocities,groupRadii[i],atime,massDMParticle) 
             kineticEnergy += kineticEnergyDM
             potentialEnergy += potentialEnergyDM
             mass += massDM
             if boundedness ==1: 
                print("bounded, checking for virialized")
                virialization,virial_ratio = check_virialized(kineticEnergy, potentialEnergy)
        if virialization ==1:
            virial_radius = calc_virial_radius(potentialEnergy,mass) #returns virial radius in cm 
            print("virial radius is " + str(virial_radius/UnitLength_in_cm) +" kpc")
        else: 
            virial_radius = calc_max_radius(starPos_inGroup,groupPos[i],boxSize)
        bounded.append(boundedness)
        massesDM.append(massDM)
        virialRatios.append(virial_ratio)
        recalcRadii.append(virial_radius)
        virialized.append(virialization)
    # objs['rot_curves'] = np.array(rotation,dtype=object)
    # objs['rot_radii'] =np.array(radii,dtype=object)
    objs['bounded'] = np.array(bounded)
    objs['virialized'] = np.array(virialized)
    objs['massDM'] = np.array(massesDM)
    objs['recalcRadii'] = np.array(recalcRadii)
    objs['virialRatios'] = np.array(virialRatios)
    objs['r200'] = groupRadii[halo100_indices]
    return objs

def add_bounded_calculation(filename, snapnum, group = "Stars"):
    """
    wrapper function
    """
    print('opening files')
    gofilename = str(filename)
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
    # TESTING MODE: UNCOMMENT below!!
    halo100_indices = halo100_indices[-150:-1]
    _,allStarMasses, allStarPositions, allStarVelocities= get_starIDs(snap)
    _,allDMPositions, allDMVelocities= get_DMIDs(snap)
    startAllStars, endAllStars = get_starIDgroups(cat, halo100_indices)
    halo100_pos = get_GroupPos(cat, halo100_indices)
    halo100_rad = get_GroupRadii(cat, halo100_indices)
    halo100_vel = get_GroupVel(cat,halo100_indices)
    atime = 1./(1.+redshift)
    print("calculating boundedness for all objects")
    objs = iterate_galaxies(atime, boxSize, halo100_indices,allStarMasses, allStarPositions,allStarVelocities, allDMPositions, allDMVelocities, startAllStars,endAllStars, halo100_rad,halo100_pos, halo100_vel,massDMParticle* UnitMass_in_g / hubbleparam)
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
    objs = add_bounded_calculation(gofilename, snapnum)

    # with open(gofilename+"/stellar_rotation_"+str(snapnum)+"_V1.dat",'wb') as f:   
    #     pickle.dump(objs, f)
    # with open("/u/scratch/c/clairewi/test_stellar_rotation_"+str(snapnum)+"_V2.dat",'wb') as f:   
    #     pickle.dump(objs, f)
    # print("SAVED OUTPUT!")
