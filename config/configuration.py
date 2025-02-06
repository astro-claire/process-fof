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

def set_directories():
    """
    Returns the input and output directories specified by the user in file config.yaml.
    """
    config = load_config()
    input_dir = config['directories']['input_dir']
    output_dir= config['directories']['output_dir']
    return input_dir, output_dir

def get_code_dir():
    """
    Returns the code directory
    """
    config = load_config()
    code_dir= config['directories']['code_dir']
    return code_dir

# Example usage
if __name__ == '__main__':
    # config = load_config()
    config = load_constants()
    print(config)
