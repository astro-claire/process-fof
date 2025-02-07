import os
import yaml

def load_config(config_file='config.yaml'):
    """
    Load configuration from YAML file.
    """
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_constants(config_file='constants.yaml'):
    """
    Load constants from a YAML file.
    """
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    #ensure strings are removed!
    for key in config['constants'].keys():
        config['constants'][key]= float(config['constants'][key])
    return config['constants']

def set_directories(config_file='config.yaml'):
    """
    Returns the input and output directories specified by the user in file config.yaml.
    """
    config = load_config(config_file=config_file)
    input_dir = config['directories']['input_dir']
    output_dir= config['directories']['output_dir']
    return input_dir, output_dir

def get_code_dir(config_file='config.yaml'):
    """
    Returns the code directory
    """
    config = load_config(config_file=config_file)
    code_dir= config['directories']['code_dir']
    return code_dir

def get_newstarsfile(config_file='config.yaml'):
    """
    Returns the newstars file
    """
    config = load_config(config_file=config_file)
    newstars_sig0 = config['files']['newstars_sig0']
    newstars_sig2 = config['files']['newstars_sig2']
    return newstars_sig0, newstars_sig2

def get_snaps(config_file='config.yaml'):
    """
    Return the list of simulation snapshots from the configuration file.
    """
    config = load_config(config_file=config_file)
    snapshots = config['simulation_parameters']['snapshots']
    return snapshots

def get_box_size(config_file='config.yaml'):
    """
    Return the box size from the configuration file.
    """
    config = load_config(config_file=config_file)
    box_size = config['simulation_parameters']['boxsize']
    return box_size

# Example usage
if __name__ == '__main__':
    # config = load_config()
    config = load_constants()
    print(config)
