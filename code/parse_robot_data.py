import numpy as np
import os
import fnmatch
import h5py
import resolve_robotics_uri_py
import pickle
import argparse

from RobotDatasetLoader import RobotDatasetLoader


def main():
    parser = argparse.ArgumentParser(description="Parse robot data")
    parser.add_argument(
        "--folder_datasets",
        type=str,
        nargs="+",
        help="Folder containing the datasets to parse",
    )
    parser.add_argument(
        "--urdf_uri", type=str, help="URI of the robot URDF file"
    )

    args = parser.parse_args()

    config_dataset = {}
    # config_dataset["dataset_type"] = "pole"
    config_dataset["dataset_type"] = "current_control"
    folder_datasets = args.folder_datasets
    # urdf_path = str(resolve_robotics_uri_py.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN001/model.urdf"))
    urdf_path = str(resolve_robotics_uri_py.resolve_robotics_uri(args.urdf_uri))

    imu_name = "waist_imu_0"

    config_dataset["set_base_from_imu"] = True

    config_robot = {}

    config_robot["base_link"] = "torso_1"
    config_robot["contact_link"] = "root_link"

    for i, folder in enumerate(folder_datasets):
        config_dataset["imu_as_base"] = imu_name

        config_robot["model_path"] = urdf_path
        data_loader = RobotDatasetLoader(config_robot, config_dataset)

        # Take all .mat files in folder
        file_mat = []
        for filename in os.listdir(folder):
            if fnmatch.fnmatch(filename, "*.mat"):
                file_mat.append(os.path.join(folder, filename))

        for dataset in file_mat:

            # load the data only if it the dataset file name constains the world "_resampled"
            if "_resampled" not in dataset:
                continue

            print("Dataset: ", dataset)
            data_parsed = data_loader.load_and_parse_dataset(dataset)
            print("Done")
            print("")

            # Save the parsed dataset in a .mat file
            # Define output folder path
            output_folder = os.path.join(os.path.dirname(dataset), "parsed")

            # Check if output folder exists, if not create it
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Define output file path
            output_file = os.path.join(
                output_folder,
                os.path.splitext(os.path.basename(dataset))[0] + "_parsed.pickle",
            )

            # Save the parsed dataset in a .mat file
            # Save the data to an HDF5 file using h5py
            # Create an h5py file
            with open(output_file, "wb") as f:
                pickle.dump(data_parsed, f)


if __name__ == "__main__":
    main()