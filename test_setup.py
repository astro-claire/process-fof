import os
import yaml

def check_files(directory):
    has_snap_groupordered = False
    has_fof_subhalo_tab = False
    for file_name in os.listdir(directory):
        print(file_name)
        if file_name.startswith("snap-groupordered"):
            has_snap_groupordered = True
            break
    for file_name in os.listdir(directory):
        if file_name.startswith("fof_subhalo_tab"):
            has_fof_subhalo_tab = True
            break
    return has_snap_groupordered and has_fof_subhalo_tab

def main():
    # Load configuration
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    input_dir = config['directories']['input_dir']
    fof_algorithms = config['fof_algorithms']
    for fof_name, fof_data in fof_algorithms.items():
        if fof_data['algorithm'] == 1:
            print(f"{fof_name} is on, checking directories")
            for sigma, use in fof_data['fof_params'].items():
                if use == 1:
                    if sigma == "0Sigma":
                        sigma_name = 0
                    elif sigma == "2Sigma":
                        sigma_name = 2
                    else:
                        sigma_name = sigma[0]
                    dir_name = f"{fof_name}-Sig{sigma_name}"
                    path = os.path.join(input_dir, dir_name)
                    if os.path.isdir(path):
                        if check_files(path):
                            print(f"{dir_name} setup status: PASSED. All required files found in {path}")
                        else:
                            print(f"{dir_name} setup status: FAILED. Missing files in {path}")
                elif use == 0:
                    continue

if __name__ == "__main__":
    main()