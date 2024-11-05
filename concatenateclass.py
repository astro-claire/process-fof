import numpy as np
import h5py 
import numpy as np
import pickle
import os
from astropy import units as un
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM


class processedFOF(): 
    def __init__(self,snapnum, directory, sv, path = '/u/home/c/clairewi/project-snaoz/FOF_project', verbose = True, maxidx = 300, **kwargs): 
        """
        Initializes fof post processing  concatenator
        
        Parameters: 
            snapnum (int): snapshot number
            directory (str): directory where files are located inside path (minus the stream velocity part)
            sv (str): stream velocity value for the run
            path (str): location of all post processing files
            verbose (bool): whether or not you want output to be printed 
        
        """
        self.snapnum = snapnum 
        self.sv = sv
        self.maxidx = maxidx
        self.directory = directory
        self.path = path + "/"+  str(directory) + str(sv)
        if 'extra_path' in kwargs.keys():
            self.path = self.path + "/"+str(kwargs['extra_path'])
        self.bounded_path = "bounded2"
        if 'bounded_path' in kwargs.keys():
            self.bounded_path = str(kwargs['bounded_path'])
        self.verbose = verbose
        self.group = self._setGroup()
        self.properties = self._findFOF()
        self._findRotation() 
        self.properties['centers'], self.properties['fofradii'] = self._getFOFData()
        self._properties_notBounded = list(self.properties.keys())
        self.boundedidx=0.
        self.properties['bounded'] =[]
        if 'DM' in self.properties['prim']:  
            self._findDMBounded(path = self.bounded_path)
        else:
            self._findBounded(path = self.bounded_path)
        self._allKeys = list(self.properties.keys())
        if 'prim' in self._allKeys: #remove non array properties for the unfinished/unbounded remover to work
            self._allKeys.remove('prim')
        if 'sec' in self._allKeys:
            self._allKeys.remove('sec')
        if 'bounded' in self._allKeys:
            self._allKeys.remove('bounded')
        self.goodidx = []
        #self.chopUnfinished()
        #self.chopUnBounded()
        # self.setupCosmo()
        
    def _findFOF(self): 
        """
        Opens fof_postprocess file and load in the objects

        Returns: 
            dict: properties of the objects for the fof script located at self.path for snapnum self.snapnum.
        """
        filepath = self.path 
        fof_process_name = "fof_postprocess_"+ str(self.snapnum)+"_"+ str(self.sv) + "_" + self.directory
        for filename in os.listdir(filepath):
            if fof_process_name in filename:
                        fof_process_path = filepath+"/"+filename      
        if 'fof_process_path' in locals():
                if self.verbose ==True: 
                    print(fof_process_path)
                with open(str(fof_process_path),'rb') as f: 
                    objs = pickle.load(f) 
                properties = objs
        else: 
            properties = {}
            properties['prim']='DM' #set default to DM primary FOF 
        return properties
    
    def _setGroup(self):
        """
        Sets which type of particle was used for groups in the fof postprocessing

        Returns: 
            str: "stars" if star particles, "DM" if DM particles, or "gas" if gas particles
        """
        if self.directory == "SGP-" or self.directory == "SP-":
            return "stars"
        elif self.directory =="DMP-GS-" or self.directory == "SGDMP-":
            return "DM"
        elif self.directory =="GP-SS-" or self.directory == "GP-":
            return "gas"
        
    def _getFOFData(self):
        """
        Gets the centers of mass and R_crit_200 derived by fof 

        Attributes: 
            groupnum (int): threshhold number of particles required to return a group
                300 if DM 
                100 if stars or gas

        Returns: 
            Numpy.ndarray  (centers of mass), Numpy.ndarray (radii)
        
        Sets: 
            properties['maxradii']: if file exists, adds array of max radii to properties dictionary 
        """
        filepath = str(self.path) + "/fof_subhalo_tab_" + str(self.snapnum).zfill(3)+".hdf5"
        f = h5py.File(filepath)
        centers = []
        fofradii = []
        groupnum  =100
        if self.group == "gas":
            grouptype = 0
        elif self.group== "stars":
            grouptype = 4
        elif self.group == "DM":
            grouptype = 1
            groupnum = 300
        else:
            if self.verbose ==True: 
                print("defaulting to DM grouptype")
            grouptype = 1
            groupnum = 300
        # Convert relevant data to NumPy arrays if they aren't already
        group_pos = np.array(f['Group']['GroupPos'])
        group_len_type = np.array(f['Group']['GroupLenType'])
        group_r_crit200 = np.array(f['Group']['Group_R_Crit200'])
        self.atime = f['Header'].attrs['Time']
        self.omegaLambda = f['Header'].attrs['OmegaLambda']
        self.H0 = f['Header'].attrs['HubbleParam']
        self.omegaMatter0 = f['Header'].attrs['Omega0']
        # Apply a mask to filter groups based on 'grouptype' and 'groupnum'
        mask = group_len_type[:, grouptype] > groupnum
        # Use the mask to select the centers and radii directly
        centers = group_pos[mask]
        fofradii = group_r_crit200[mask]
        maxradname = "maxradii_"+str(self.snapnum)+"_V1.dat"
        filepath = self.path 
        for filename in os.listdir(filepath):
            # Check if the file exists in that directory
            if maxradname in filename:
                max_rad_path = filepath+"/"+filename
        if 'max_rad_path' in locals():
            if self.verbose ==True: 
                print(max_rad_path)
            with open(str(max_rad_path),'rb') as f: 
                maxrad = pickle.load(f) 
                self.properties['maxradii']= maxrad['maxradii']
        return centers, fofradii
    
    def _findRotation(self): 
        """
        Opens the stellar rotation output file and adds rotational properties to self.properties dictionary

        Returns: 
            None
        """
        filepath = self.path 
        star_rot_name = "stellar_rotation_"+ str(self.snapnum)+"_"+ str(self.sv) + "_" + self.directory +"_v5"  
        for filename in os.listdir(filepath):
            # Check if the file exists in that directory
            if star_rot_name in filename:
                star_rot_path = filepath+"/"+filename
        #Now, also get the gas rotation file if it exists: 
        #NOTE WE ARE SPECIFYING VERSION NUMBER HERE
        gas_rot_name = "gas_rotation_"+ str(self.snapnum)+"_"+ str(self.sv) + "_" + self.directory +"_v5"  
        for filename in os.listdir(filepath):
            # Check if the file exists in that directory
            if gas_rot_name in filename:
                gas_rot_path = filepath+"/"+filename
        if 'star_rot_path' in locals():
            if self.verbose ==True: 
                print(star_rot_path)
            with open(str(star_rot_path),'rb') as f: 
                starrot = pickle.load(f) 
            self.properties['rotation_curve_rms']= starrot['rotation_curve_rms']
            self.properties['rotation_curve_turb']= starrot['rotation_curve_turb']
            self.properties['rotation_curve_rot']= starrot['rotation_curve_rot']
            self.properties['rotation_curve_rad']= starrot['rotation_curve_rad']   
            self.properties['v_rms']= starrot['v_rms']   
            self.properties['v_rad']= starrot['v_rad']   
            self.properties['v_rot']= starrot['v_rot']   
            self.properties['v_turb']= starrot['v_turb']
            self.properties['avg_stellarMass'] = starrot['avg_mass']
            self.properties['nstarsgroup'] = starrot['ngroup']
            self.properties['star_rot_radii']= starrot['rot_radii']                                              
        if 'gas_rot_path' in locals():
            if self.verbose ==True:
                print(gas_rot_path)
            with open(str(gas_rot_path),'rb') as f: 
                gasrot = pickle.load(f) 
            self.properties['gas_rotation_curve_rms']= gasrot['rotation_curve_rms']
            self.properties['gas_rotation_curve_turb']= gasrot['rotation_curve_turb']
            self.properties['gas_rotation_curve_rot']= gasrot['rotation_curve_rot']
            self.properties['gas_rotation_curve_rad']= gasrot['rotation_curve_rad']   
            self.properties['gas_v_rms']= gasrot['v_rms']   
            self.properties['gas_v_rad']= gasrot['v_rad']   
            self.properties['gas_v_rot']= gasrot['v_rot']   
            self.properties['gas_v_turb']= gasrot['v_turb']   
            self.properties['gas_rot_radii']= gasrot['rot_radii']  
            self.properties['avg_gasMass'] = gasrot['avg_mass']
            self.properties['ngasgroup'] = gasrot['ngroup']
        #Can check the print statements to make sure the expected files have loaded. 
        
    def _findDMBounded(self,path = "bounded2"): 
        """
        Opens bounded chunk files, concatenates, and adds their properties to the properties dict. 

        Returns: 
            None
        """
        filepath = self.path+"/" + str(path)
        print(filepath)
        if os.path.exists(filepath):
            indices = list(range(700))
            indices.reverse()
            self.properties['bounded'] = []
            self.properties['virialized']= []
            self.properties['usedDM']= []
            indexexists = 0
            for j in indices:
                fof_process_name = "bounded_portion_"+str(self.snapnum)+"_chunk"+str(j)+"_"
                for filename in os.listdir(filepath):
                    # Check if the file exists in that directory
                    if fof_process_name in filename:
                        if indexexists == 0: 
                            biggestchunkfile = filename
                            indexexists = 1
                        fof_process_path = filepath+"/"+filename
                    #if 'fof_process_path' in locals():
                        with open(fof_process_path,'rb') as f: 
                            chunk = pickle.load(f) 
                        self.properties['bounded'] = np.concatenate((self.properties['bounded'], chunk['bounded']), axis = None)
                        self.properties['virialized'] = np.concatenate((self.properties['virialized'], chunk['virialized']), axis = None)
                        self.properties['usedDM'] = np.concatenate((self.properties['usedDM'], chunk['usedDM']), axis = None)
            try:
                self.boundedidx  = int(biggestchunkfile[-10:-7])
            except ValueError:
                try:
                    self.boundedidx = int(biggestchunkfile[-9:-7])
                except ValueError:
                    self.boundedidx = int(biggestchunkfile[-8:-7])
            except NameError:
                self.boundedidx = 0
            if self.verbose == True: 
                print("highest bounded calculated is " + str(self.boundedidx))
        else:
            print("no bounded exists for this snap file")

    def _findBounded(self,path = "bounded2"): 
        """
        Opens bounded chunk files, concatenates, and adds their properties to the properties dict. 

        Returns: 
            None
        """
        filepath = self.path+"/" + str(path)
        if os.path.exists(filepath):
            indices = list(range(self.maxidx))
            indices.reverse()
            self.properties['massDM'] = []
            self.properties['massStars']=[]
            self.properties['bounded'] = []
            self.properties['virialized']= []
            self.properties['recalcRadii']= []
            self.properties['usedDM']= []
            indexexists = 0
            for j in indices:
                fof_process_name = "bounded_portion_"+str(self.snapnum)+"_chunk"+str(j)+"_"
                for filename in os.listdir(filepath):
                    # Check if the file exists in that directory
                    if fof_process_name in filename:
                        if indexexists == 0: 
                            biggestchunkfile = filename
                            indexexists = 1
                        fof_process_path = filepath+"/"+filename
                    #if 'fof_process_path' in locals():
                        with open(fof_process_path,'rb') as f: 
                            chunk = pickle.load(f) 
                        self.properties['massDM'] = np.concatenate((self.properties['massDM'], chunk['massDM']), axis = None)
                        self.properties['massStars'] = np.concatenate((self.properties['massStars'], chunk['massStars']), axis = None)
                        self.properties['bounded'] = np.concatenate((self.properties['bounded'], chunk['bounded']), axis = None)
                        self.properties['virialized'] = np.concatenate((self.properties['virialized'], chunk['virialized']), axis = None)
                        self.properties['recalcRadii'] = np.concatenate((self.properties['recalcRadii'], chunk['recalcRadii']), axis = None)
                        self.properties['usedDM'] = np.concatenate((self.properties['usedDM'], chunk['usedDM']), axis = None)
            try:
                self.boundedidx  = int(biggestchunkfile[-10:-7])
            except ValueError:
                try:
                    self.boundedidx = int(biggestchunkfile[-9:-7])
                except ValueError:
                    self.boundedidx = int(biggestchunkfile[-8:-7])
            except NameError:
                self.boundedidx = 0
            if self.verbose == True: 
                print("highest bounded calculated is " + str(self.boundedidx))
        else:
            self.boundedidx = 0
            print("no bounded exists for this snap file")
        
    def accessBoundedComplete(self,arr):
        """
        This function can be used when the chunking hasn't finished running to ensure that only systems with a known bounded calculation are being used
        
        Arguments: 
            arr (Numpy.ndarray): array to crop
        
        Returns:
            Numpy.ndarray: array with only the bounded completed indices
        """
        return arr[self.boundedidx:]
    
    def accessBounded(self, arr):
        """
        returns the array but only of systems which are bounded

        Arguments: 
            arr (Numpy.ndarray): array to crop
        
        Returns:
            Numpy.ndarray: array with only the bounded indices
        """
        return arr[np.array(self.properties['bounded'], dtype = bool)]

    def chopUnfinished(self): 
        """
        For all the parameters not calculated in the bounded script, removes the indices of the objects which do not have a bounded calculation. 
        
        Returns:
            None
        """
        for key in self._properties_notBounded: 
            self.properties[key] = self.accessBoundedComplete(self.properties[key])

    def chopUnBounded(self): 
        """
        For all the parameters in self.properties, removes all unbounded objects

        Returns: 
            None
        """
        boundedidx = np.array(self.properties['bounded'].astype(bool))
        for key in self._allKeys:
            try: 
                self.properties[key] = np.array(self.properties[key])[boundedidx]
            except ValueError: 
                self.properties[key] = np.array(self.properties[key],dtype = np.ndarray)[boundedidx]
            except IndexError:
                print(str(key)+ " is not the right length")
    
    def chopResolutionHalos(self, cutoff = 300):
        """
        if the list contains DM halos beyond the resolution limit, chop them off
        """
        #TODO: Claire: rerun the rotations with correct DM group and then remove 
        self._nonRotationKeys = self._allKeys
        if 'SFR' in self.properties.keys():
            self._nonRotationKeys.append('SFR')
        for key in ['centers','fofradii','rotation_curve_rms','rotation_curve_turb','rotation_curve_rot', 'rotation_curve_rad', 'v_rms','v_rad','v_rot','v_turb','star_rot_radii','gas_rot_radii','gas_rotation_curve_rms','gas_rotation_curve_turb','gas_rotation_curve_rot', 'gas_rotation_curve_rad', 'gas_v_rms','gas_v_rad','gas_v_rot','gas_v_turb']:
            if key in self._nonRotationKeys:
                self._nonRotationKeys.remove(key)
        if 'DM' in self.properties['prim']:
            filepath = str(self.path) + "/snap-groupordered_" + str(self.snapnum)+".hdf5"
            f = h5py.File(filepath)
            resolution = f['Header'].attrs['MassTable'][1]*cutoff
            mask = self.properties['DMMass'] >resolution
            mask = np.array(mask.astype(bool))
            for key in self._nonRotationKeys:
                if len(self.properties[key])>1.:
                    try: 
                        self.properties[key]= np.array(self.properties[key])[mask]
                    except ValueError: 
                        self.properties[key] = np.array(self.properties[key],dtype = object)[mask]
                    except IndexError:
                        print(str(key)+" is not the right length")


    def addEnvironment(self):
        """
        Search for environment output and add to properties. Must be run post bounded! Only available for baryonic primaries. 
        """
        filepath = self.path 
        env_name = "environment_"+ str(self.snapnum)+"_V1"  
        for filename in os.listdir(filepath):
            # Check if the file exists in that directory
            if env_name in filename:
               env_path = filepath+"/"+filename
        #Now, also get the gas rotation file if it exists: 
        #NOTE WE ARE SPECIFYING VERSION NUMBER HERE
        if 'env_path' in locals():
            if self.verbose ==True: 
                print(env_path)
            with open(str(env_path),'rb') as f: 
                envdict = pickle.load(f) 
            if len(envdict['closestb'])== len(self.properties['virialized']):
                self.properties['closestb']= envdict['closestb']
                self.properties['closestb_dist']= envdict['closestb_dist']
                self.properties['num_within10b']= envdict['num_within10b']
                self.properties['num_within5b']= envdict['num_within5b']
                self.properties['closestdm']= envdict['closestdm']
                self.properties['closestdm_dist']= envdict['closestdm_dist']
                self.properties['closestdm_dmmass']= envdict['closestdm_dmmass']
                self.properties['closestdm_inr200']= envdict['closestdm_inr200']
                self.properties['num_within10dm']= envdict['num_within10dm']
                self.properties['num_within5dm']= envdict['num_within5dm']
            else: 
                print("ERROR: number of objects in environment directory doesn't match number of objects in FOF bounded. ")

    def setupCosmo(self): 
        """
        Calculates relevant cosmology properties. ASSUMES SNAPS ARE SEPARATED BY 1 redshift!!!!!!
        
        Sets: 
            self.deltat (float): elapsed time since last snapshot in years
        """
        self.redshift = 1./self.atime -1.
        if self.verbose ==True: print(f"creating flat LCDM cosmology with hubble param = {self.H0}  and omega_m0 = {self.omegaMatter0}." )
        cosmo = FlatLambdaCDM(H0=self.H0 * 100, Om0=self.omegaMatter0, Ob0=0.044)
        t1= cosmo.age(self.redshift).to('Myr')
        t2= cosmo.age(self.redshift+1).to('Myr')
        self.deltat = ((t1-t2).to('yr')).value  #
    
    def calcMUV(self): 
        """
        Calculates absolute uv magnitude
        
        Sets: 
            self.parameters['SFR'] (arr): star formation rates
            self.M_uv
        """
        self.setupCosmo()
        K_uv = 1.15*10.**(-28.) # kappa uv literature value
        if 'new_mStar' in self.properties.keys():
            self.properties['SFR'] = self.properties['new_mStar']*1e10/self.H0/self.deltat # star formation rate in solar masses per eyar
            L_uv = self.properties['SFR']/K_uv
            nonzero_Luv =L_uv[np.nonzero(L_uv)]
            self.M_uv = -2.5*np.log10(nonzero_Luv)+51.6  
            if 'DM' in self.properties['prim']: #Need a DM mass indicator if you're going to correct for number densities
                self.DMMass_Muv = self.properties['DMMass'][np.nonzero(L_uv)]*1e10 /self.H0
            elif 'closestdm_dmmass' in self.properties.keys():
                self.DMMass_Muv = self.properties['closestdm_dmmass'][np.nonzero(L_uv)]*1e10 /self.H0
            else: 
                print("No suitable DM key for Muv DM masses. Skipping it.")
            if 'stellarMass' in self.properties.keys():
                self.stellarMass_Muv = self.properties['stellarMass'][np.nonzero(L_uv)]*1e10 /self.H0
            # note this is a different length than most arrays because we've removed nonzero luminosity. 

    def calcTRelax(self, radiuskey = 'maxradii', radiusfactor = 1.):
        UnitVelocity_in_cm_per_s = 1.0e5 * un.cm / un.s
        UnitLength_in_cm = 3.085678e21 *un.cm
        GRAVITY_cgs = c.G.cgs
        UnitMass_in_g = 1.989e43 * un.g
        if 'nstarsgroup' in self.properties.keys():
            print("relaxation timescale for stars (per coulomb logarithm)")
            vel = self.properties['v_rms'] * UnitVelocity_in_cm_per_s
            mass = self.properties['avg_stellarMass'] * UnitMass_in_g
            N =self.properties['nstarsgroup']
            r = self.properties[radiuskey] * UnitLength_in_cm * radiusfactor
            return (vel**3 / (GRAVITY_cgs**2 * mass**2 * (N/(4./3. *np.pi * r**3 )))).to('Myr')
    
    def calcTRelaxGas(self, radiuskey = 'maxradii', radiusfactor = 1.):
        UnitVelocity_in_cm_per_s = 1.0e5 * un.cm / un.s
        UnitLength_in_cm = 3.085678e21 *un.cm
        GRAVITY_cgs = c.G.cgs
        UnitMass_in_g = 1.989e43 * un.g
        if 'ngasgroup' in self.properties.keys():
            print("relaxation timescale for gas (per coulomb logarithm)")
            vel = self.properties['gas_v_rms'] * UnitVelocity_in_cm_per_s
            mass = self.properties['avg_gasMass'] * UnitMass_in_g
            N =self.properties['ngasgroup']
            r = self.properties[radiuskey] * UnitLength_in_cm * radiusfactor
            return (vel**3 / (GRAVITY_cgs**2 * mass**2 * (N/(4./3. *np.pi * r**3 )))).to('Myr')

    def calcTSFR(self): 
        UnitMass_in_g = 1.989e43 * un.g
        if 'SFR' in self.properties.keys():
            self.properties['SFRpermass'] = self.properties['SFR']*un.Msun/un.yr/ (self.properties['stellarMass'] * UnitMass_in_g/self.H0)
            return (self.properties['SFRpermass'] **(-1.)).to('Myr')