import numpy as np
import h5py 
import numpy as np
import pickle
import os

class processedFOF(): 
    def __init__(self,snapnum, directory, sv, path = '/u/home/c/clairewi/project-snaoz/FOF_project', verbose = True, maxidx = 300): 
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
        self.verbose = verbose
        self.group = self._setGroup()
        self.properties = self._findFOF()
        self._findRotation() 
        self.properties['centers'], self.properties['fofradii'] = self._getFOFData()
        self._properties_notBounded = list(self.properties.keys())
        self._findBounded()
        self._allKeys = list(self.properties.keys())
        self._allKeys.remove('prim')
        self._allKeys.remove('sec')
        self._allKeys.remove('bounded')
        self.goodidx = []
        #self.chopUnfinished()
        #self.chopUnBounded()
        
    def _findFOF(self): 
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
        return properties
    
    def _setGroup(self):
        if self.directory == "SGP-" or self.directory == "SP-":
            return "stars"
        elif self.directory =="DMP-GS-" or self.directory == "SGDMP-":
            return "DM"
        elif self.directory =="GP-SS-" or self.directory == "GP-":
            return "gas"
        
    def _getFOFData(self):
        filepath = str(self.path) + "/fof_subhalo_tab_" + str(self.snapnum)+".hdf5"
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
        # Convert relevant data to NumPy arrays if they aren't already
        group_pos = np.array(f['Group']['GroupPos'])
        group_len_type = np.array(f['Group']['GroupLenType'])
        group_r_crit200 = np.array(f['Group']['Group_R_Crit200'])

        # Apply a mask to filter groups based on 'grouptype' and 'groupnum'
        mask = group_len_type[:, grouptype] > groupnum
        # Use the mask to select the centers and radii directly
        centers = group_pos[mask]
        fofradii = group_r_crit200[mask]
        # for idx in range(len(f['Group']['GroupPos'])):
        #     if f['Group']['GroupLenType'][idx][grouptype] >groupnum:
        #         center = [f['Group']['GroupPos'][idx][0], f['Group']['GroupPos'][idx][1], f['Group']['GroupPos'][idx][2]]
        #         fofradii.append(f['Group']['Group_R_Crit200'][idx])
        #         centers.append(center)
        return centers, fofradii
    
    def _findRotation(self): 
        filepath = self.path 
        star_rot_name = "stellar_rotation_"+ str(self.snapnum)+"_"+ str(self.sv) + "_" + self.directory +"_v4"  
        for filename in os.listdir(filepath):
            # Check if the file exists in that directory
            if star_rot_name in filename:
                star_rot_path = filepath+"/"+filename
        #Now, also get the gas rotation file if it exists: 
        #NOTE WE ARE SPECIFYING VERSION NUMBER HERE
        gas_rot_name = "gas_rotation_"+ str(self.snapnum)+"_"+ str(self.sv) + "_" + self.directory +"_v4"  
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
            self.properties['star_rot_radii']= starrot['rot_radii']                                              
        if 'gas_rot_path' in locals():
            if self.verbose ==True:
                print(gas_rot_path)
            with open(str(gas_rot_path),'rb') as f: 
                gasrot = pickle.load(f) 
            self.properties['gas_rotation_curve_rms']= starrot['rotation_curve_rms']
            self.properties['gas_rotation_curve_turb']= starrot['rotation_curve_turb']
            self.properties['gas_rotation_curve_rot']= starrot['rotation_curve_rot']
            self.properties['gas_rotation_curve_rad']= starrot['rotation_curve_rad']   
            self.properties['gas_v_rms']= starrot['v_rms']   
            self.properties['gas_v_rad']= starrot['v_rad']   
            self.properties['gas_v_rot']= starrot['v_rot']   
            self.properties['gas_v_turb']= starrot['v_turb']   
            self.properties['gas_rot_radii']= starrot['rot_radii']  
        #Can check the print statements to make sure the expected files have loaded. 
        
    def _findBounded(self): 
        filepath = self.path + "/bounded2"
        indices = list(range(self.maxidx))
        indices.reverse()
        self.properties['massDM'] = []
        self.properties['massStars']=[]
        self.properties['bounded'] = []
        self.properties['virialized']= []
        self.properties['recalcRadii']= []
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
                    with open(fof_process_path,'rb') as f: 
                        chunk = pickle.load(f) 
                    self.properties['massDM'] = np.concatenate((self.properties['massDM'], chunk['massDM']), axis = None)
                    self.properties['massStars'] = np.concatenate((self.properties['massStars'], chunk['massStars']), axis = None)
                    self.properties['bounded'] = np.concatenate((self.properties['bounded'], chunk['bounded']), axis = None)
                    self.properties['virialized'] = np.concatenate((self.properties['virialized'], chunk['virialized']), axis = None)
                    self.properties['recalcRadii'] = np.concatenate((self.properties['recalcRadii'], chunk['recalcRadii']), axis = None)
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
        
    def accessBoundedComplete(self,arr):
        """
        This function can be used when the chunking hasn't finished running to ensure that only systems with a known bounded calculation are being used
        
        """
        return arr[self.boundedidx:]
    
    def accessBounded(self, arr):
        """
        returns the array but only of systems which are bounded
        """
        return arr[np.array(self.properties['bounded'], dtype = bool)]

    def chopUnfinished(self): 
        for key in self._properties_notBounded: 
            self.properties[key] = self.accessBoundedComplete(self.properties[key])

    def chopUnBounded(self): 
        boundedidx = np.array(self.properties['bounded'].astype(bool))
        for key in self._allKeys:
            try: 
                self.properties[key] = np.array(self.properties[key])[boundedidx]
            except IndexError:
                print(str(key)+ " is not the right length")