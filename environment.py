import numpy as np
import h5py 
import numpy as np
import pickle
import os
from concatenateclass import processedFOF
from sys import argv

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
    return fof.properties['centers'], fof.properties['recalcradii']


if __name__=="__main__":
    """
    Routine if running as a script

    Arguments: 
    gofilename path to directory containing groupordered file + fof table
    # foffilename 
    snapnum (float)
    """
    script, gofilename, snapnum = argv