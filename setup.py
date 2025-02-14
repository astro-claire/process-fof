import os
import yaml

def create_directories(base_dir, fof_name, sigma):
    dir_name = f"{fof_name}-Sig{sigma}"
    path = os.path.join(base_dir, dir_name)
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, "bounded3", "indv_objs"), exist_ok=False)
        print(f"Created directories for {dir_name}")
    else:
        print(f"Directories for {dir_name} already exist")

def main():
    # Load configuration
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    input_dir = config['directories']['input_dir']
    fof_algorithms = config['fof_algorithms']
    for fof_name, fof_data in fof_algorithms.items():
        if fof_data['algorithm'] == 1:
            print(f"{fof_name} is on, creating directories")
            for sigma, use in fof_data['fof_params'].items():
                if use == 1:
                    if sigma == "0Sigma":
                        sigma_name = 0
                    elif sigma == "2Sigma":
                        sigma_name = 2
                    else:
                        sigma_name = sigma[0]
                        print("WARNING: SV value not default option of 0 or 2, may not be supported for some scripts.")
                    create_directories(input_dir, fof_name, str(sigma_name))
                elif use == 0:
                    continue
    print("If new directories have been created, please add hdf5 FOF output files to the appropriate locations.")

if __name__ == "__main__":
    main()