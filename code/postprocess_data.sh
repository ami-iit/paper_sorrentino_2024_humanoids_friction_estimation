#!/bin/bash

# Usage: ./run_matlab.sh -f dataset_folder -j moving_joint -a "associate_joint1 associate_joint2 ..."

# Initialize variables
dataset_folder=""
moving_joint=""
associated_joints=""

# Parse command-line arguments using getopts
while getopts "f:j:a:" opt; do
    case $opt in
        f) dataset_folder="$OPTARG" ;;
        j) moving_joint="$OPTARG" ;;
        a) associated_joints="$OPTARG" ;;
        \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    esac
done

# Check if the required arguments are provided
if [ -z "$dataset_folder" ] || [ -z "$moving_joint" ] || [ -z "$associated_joints" ]; then
    echo "Usage: $0 -f dataset_folder -j moving_joint -a \"associate_joint1 associate_joint2 ...\""
    exit 1
fi

# Convert the associated joints string to a MATLAB-compatible cell array format
IFS=' ' read -r -a joints_array <<< "$associated_joints"
associated_joints_str="{"
for joint in "${joints_array[@]}"; do
    associated_joints_str+="'$joint',"
done
associated_joints_str="${associated_joints_str%,}}"

# Run MATLAB script with the parameters
# matlab -batch "reduce_and_resample_dataset('$dataset_folder', '$moving_joint', $associated_joints_str)"

# # Alternatively, you can use the -nodisplay and -nosplash options to avoid opening the GUI:
matlab -nodisplay -nosplash -r "reduce_and_resample_dataset('$dataset_folder', '$moving_joint', $associated_joints_str); exit;"

# call the python command
python3 parse_robot_data.py --folder_datasets $dataset_folder --urdf_uri package://ergoCub/robots/ergoCubSN001/model.urdf
