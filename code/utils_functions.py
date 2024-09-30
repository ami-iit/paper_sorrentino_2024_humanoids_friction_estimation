import numpy as np
import torch.nn as nn
from typing import List
import os
import torch
import pickle
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_rmse(model, x_test, y_test):
    model.eval()
    y_hat = model(x_test)
    pred = model.y_scaler.inverse_transform(y_hat.mean)
    return torch.sqrt(torch.mean((pred - y_test) ** 2))

def compute_train_test_split(x, y, test_size, device="cpu"):
    x_train, x_test, y_train, y_test, train_ix, test_ix = train_test_split(
        x,
        y,
        list(range(len(x))),
        test_size=test_size,
    )
    return (
        torch.from_numpy(x_train).to(torch.float32).to(device),
        torch.from_numpy(x_test).to(torch.float32).to(device),
        torch.from_numpy(y_train).to(torch.float32).to(device),
        torch.from_numpy(y_test).to(torch.float32).to(device),
        train_ix,
        test_ix,
    )

def sample_batch_indices(x, y, batch_size, rs=None):
    if rs is None:
        rs = np.random.RandomState()

    train_ix = np.arange(len(x))
    rs.shuffle(train_ix)

    n_batches = int(np.ceil(len(x) / batch_size))

    batch_indices = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_indices.append(train_ix[start:end].tolist())

    return batch_indices


def create_dataset(data: List, config, input_type, training=True):

    num_of_datasets = len(data)

    input = []
    output = []
    physics = []
    joint_vel = []

    for i in range(num_of_datasets):
        s_temp = np.array(data[i]["s"]).flatten()
        s_temp_motor_side = config["gear_ratio"] * np.array(data[i]["s"]).flatten()
        ds_temp = np.array(data[i]["ds"]).flatten()
        dds_temp = np.array(data[i]["dds"]).flatten()
        im_temp = np.array(data[i]["im_filtered"]).flatten()
        theta_temp = np.array(data[i]["theta"]).flatten()
        omega_temp = np.array(data[i]["omega"]).flatten()
        omega_dot_temp = np.array(data[i]["omega_dot"]).flatten()
        delta = theta_temp[0] - s_temp_motor_side[0]
        theta_temp = theta_temp - delta
        taum_temp = config["ktau"] * config["gear_ratio"] * im_temp
        tauj_temp = np.array(data[i]["tauj"]).flatten()
        tauf_temp = taum_temp - tauj_temp
        history_size = config["history_size"]

        chunks = np.array(data[i]["chunks"]).flatten()

        # I want to discard part of the dataset like the 60%
        # but from samples with abs(ds_temp) < 0.03. 
        # The idea is to find indices where abs(ds_temp) < 0.03
        # and extract randomly the 60% of them
        # Then, I will use the indices to discard the samples
        # from the dataset in the for loop

        # # Find indices where abs(ds_temp) < 0.03
        # indices = np.where(np.abs(ds_temp) < 0.02)[0]
        # print("Number of samples with abs(ds_temp) < 0.02: ")
        # print(len(indices))
        # # Calculate the number of samples to discard
        # discard_size = int(0.85 * len(indices))
        # print("Number of samples to discard: ")
        # print(discard_size)
        # # Randomly select the indices to discard
        # discard_indices = np.random.choice(indices, discard_size, replace=False)

        # # Plot the distribution of ds_temp without the discarded indices
        # plt.figure()
        # plt.hist(ds_temp[~np.isin(np.arange(len(ds_temp)), discard_indices)], bins=100)
        # plt.title("Distribution of ds_temp without discarded indices")
        # plt.savefig('ds_distr_without.png')

        # # Plot the distribution of ds_temp with the discarded indices
        # plt.figure()
        # plt.hist(ds_temp, bins=100)
        # plt.title("Distribution of ds_temp with discarded indices")
        # plt.savefig('ds_distr_with.png')
        
        # # Plot ds_temp without the discarded indices VS tauf_temp without the discarded indices
        # # Remove samples from plot where dds > threshold
        # indices = np.where(np.abs(dds_temp) > 0.1)[0]
        # discard_indices = np.concatenate((discard_indices, indices))
        # plt.figure()
        # plt.scatter(ds_temp[~np.isin(np.arange(len(ds_temp)), discard_indices)], tauf_temp[~np.isin(np.arange(len(ds_temp)), discard_indices)], s=1)
        # plt.title("ds_temp without discarded indices VS tauf_temp without discarded indices")
        # plt.savefig('ds_vs_tauf_without.png')

        Fc = config["Fc"]
        Fs = config["Fs"]
        Fv = config["Fv"]
        Vs = config["Vs"]
        alpha = 1e-10

        # Process the data, respecting chunk boundaries
        start_index = 0
        for i in range(1, len(chunks)):
            # If the chunk changes, reset the start_index
            if chunks[i] != chunks[start_index]:
                start_index = i
                continue  # No history can be formed yet, skip this iteration

            tk = i

            # Ensure there's enough history within the same chunk
            if tk - start_index < history_size - 1:
                continue  # Not enough history, skip

            if (
                np.abs(dds_temp[tk]) < config["joint_acc_limit"]
                and np.abs(im_temp[tk]) < config["im_limit"]
                and np.abs(tauf_temp[tk]) < config["tauf_limit"]
                and np.abs(ds_temp[tk]) < config["joint_vel_limit"]
                # and tk not in discard_indices
            ):
                if input_type == 1: # input is [rs - theta, ds]
                    s_history = s_temp_motor_side[tk - history_size + 1: tk + 1].flatten()
                    ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()
                    theta_history = theta_temp[tk - history_size + 1: tk + 1].flatten()

                    input.append(np.concatenate((s_history-theta_history, ds_history), axis=0))
                
                elif input_type == 2: # input is [omega, ds]
                    omega_hist = omega_temp[tk - history_size + 1: tk + 1].flatten()
                    ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()

                    input.append(np.concatenate((omega_hist, ds_history), axis=0))
                
                elif input_type == 3: # input is [(sr-theta), omega, ds]
                    s_history = s_temp_motor_side[tk - history_size + 1: tk + 1].flatten()
                    theta_history = theta_temp[tk - history_size + 1: tk + 1].flatten()
                    omega_hist = omega_temp[tk - history_size + 1: tk + 1].flatten()
                    ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()

                    input.append(np.concatenate((s_history-theta_history, omega_hist, ds_history), axis=0))

                elif input_type == 4: # input is [(sr-theta), s, ds]
                    s_history = s_temp_motor_side[tk - history_size + 1: tk + 1].flatten()
                    theta_history = theta_temp[tk - history_size + 1: tk + 1].flatten()
                    s_hist_joint_side = s_temp[tk - history_size + 1: tk + 1].flatten()
                    ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()

                    input.append(np.concatenate((s_history-theta_history, s_hist_joint_side, ds_history), axis=0))
                
                elif input_type == 5: # input is [theta, omega, s, ds]
                    theta_history = theta_temp[tk - history_size + 1: tk + 1].flatten()
                    omega_hist = omega_temp[tk - history_size + 1: tk + 1].flatten()
                    s_hist_joint_side = s_temp[tk - history_size + 1: tk + 1].flatten()
                    ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()

                    input.append(np.concatenate((theta_history, omega_hist, s_hist_joint_side, ds_history), axis=0))

                if training:
                    ds = ds_temp[tk].flatten()

                    stribeck_term = (Fs - Fc) * np.exp(-(ds / Vs) ** 2)

                    tanh_term = np.tanh(ds / alpha)

                    tauF = (Fc + stribeck_term) * tanh_term + Fv * ds

                    physics.append(tauF)

                # Prepare output
                tauF_k = tauf_temp[tk].flatten()
                output.append(tauF_k)

    return (
        np.array(input, dtype=np.float32),
        np.array(output, dtype=np.float32),
        np.array(physics, dtype=np.float32)
    )

def split_and_save_data(input_data: np.ndarray, output_data: np.ndarray, physics_data: np.ndarray, split_percentage_validation: float, output_dir: str):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate the number of samples to move to the validation set
    total_samples = input_data.shape[0]
    val_size = int(split_percentage_validation * total_samples)

    # Generate random indices for validation set
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Split the data into training and validation sets
    train_input, val_input = input_data[train_indices], input_data[val_indices]
    train_output, val_output = output_data[train_indices], output_data[val_indices]
    train_physics, val_physics = physics_data[train_indices], physics_data[val_indices]

    # Save training data
    train_data = {
        'input': train_input,
        'output': train_output,
        'physics': train_physics
    }
    with open(os.path.join(output_dir, 'train_data.pickle'), 'wb') as f:
        pickle.dump(train_data, f)

    # Save validation data
    val_data = {
        'input': val_input,
        'output': val_output,
        'physics': val_physics
    }
    with open(os.path.join(output_dir, 'val_data.pickle'), 'wb') as f:
        pickle.dump(val_data, f)

    print(f"Training and validation data saved in {output_dir}")

    return train_input, train_output, train_physics

def load_dataset(folders: List[str]):
    data = []
    # Access all the datasets and concatenate data
    for folder in folders:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        loaded_data = pickle.load(f)
                        data.append(loaded_data)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

    return data


