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


def create_dataset(data: List, config, input_type, training=True, discard_percentage=0.0):

    num_of_datasets = len(data)

    input = []
    output = []
    physics = []

    # Extract constants from config
    Fc = config["Fc"]
    Fs = config["Fs"]
    Fv = config["Fv"]
    Vs = config["Vs"]
    alpha = 1e-10
    gear_ratio = config["gear_ratio"]
    ktau = config["ktau"]
    joint_acc_limit = config["joint_acc_limit"]
    im_limit = config["im_limit"]
    tauf_limit = config["tauf_limit"]
    joint_vel_limit = config["joint_vel_limit"]
    history_size = config["history_size"]

    # Precompute chunk-based discards
    all_chunks = []
    for j in range(num_of_datasets):
        chunks = np.array(data[j]["chunks"]).flatten()
        all_chunks.extend(list(np.unique(chunks)))

    # Determine number of chunks to discard
    unique_chunks = np.unique(all_chunks)
    num_chunks_to_discard = int(discard_percentage * len(unique_chunks))

    if num_chunks_to_discard > 0:
        # Randomly select chunks to discard
        discard_chunks = np.random.choice(unique_chunks, num_chunks_to_discard, replace=False)
    else:
        discard_chunks = []

    for j in range(num_of_datasets):
        dataset = data[j]

        # Flatten necessary arrays only once
        s_temp = np.array(dataset["s"]).flatten()
        s_temp_motor_side = gear_ratio * s_temp
        ds_temp = np.array(dataset["ds"]).flatten()
        dds_temp = np.array(dataset["dds"]).flatten()
        im_temp = np.array(dataset["im_filtered"]).flatten()
        theta_temp = np.array(dataset["theta"]).flatten()
        omega_temp = np.array(dataset["omega"]).flatten()
        delta = theta_temp[0] - s_temp_motor_side[0]
        theta_temp -= delta  # Inline subtraction for delta
        taum_temp = ktau * gear_ratio * im_temp
        tauj_temp = np.array(dataset["tauj"]).flatten()
        tauf_temp = taum_temp - tauj_temp
        chunks = np.array(dataset["chunks"]).flatten()

        # Iterate over chunk boundaries and history
        start_index = 0
        for i in range(1, len(chunks)):
            if chunks[i] != chunks[start_index]:
                start_index = i
                continue

            # Skip chunks that are in discard_chunks
            if chunks[i] in discard_chunks:
                continue

            tk = i
            # Precompute condition checks once
            has_enough_history = tk - start_index >= history_size - 1
            within_limits = (
                np.abs(dds_temp[tk]) < joint_acc_limit
                and np.abs(im_temp[tk]) < im_limit
                and np.abs(tauf_temp[tk]) < tauf_limit
                and np.abs(ds_temp[tk]) < joint_vel_limit
            )

            if not (has_enough_history and within_limits):
                continue

            # Use slices instead of multiple np.concatenate
            slice_range = slice(tk - history_size + 1, tk + 1)

            if input_type == 1:  # input is [rs - theta, ds]
                input.append(np.concatenate((s_temp_motor_side[slice_range] - theta_temp[slice_range], ds_temp[slice_range])))
            elif input_type == 2:  # input is [omega, ds]
                input.append(np.concatenate((omega_temp[slice_range], ds_temp[slice_range])))
            elif input_type == 3:  # input is [(sr-theta), omega, ds]
                input.append(np.concatenate((s_temp_motor_side[slice_range] - theta_temp[slice_range], omega_temp[slice_range], ds_temp[slice_range])))
            elif input_type == 4:  # input is [(sr-theta), s, ds]
                input.append(np.concatenate((s_temp_motor_side[slice_range] - theta_temp[slice_range], s_temp[slice_range], ds_temp[slice_range])))
            elif input_type == 5:  # input is [theta, omega, s, ds]
                input.append(np.concatenate((theta_temp[slice_range], omega_temp[slice_range], s_temp[slice_range], ds_temp[slice_range])))

            if training:
                ds = ds_temp[tk]
                stribeck_term = (Fs - Fc) * np.exp(-(ds / Vs) ** 2)
                tauF = (Fc + stribeck_term) * np.tanh(ds / alpha) + Fv * ds
                physics.append(tauF)

            # Prepare output
            output.append(tauf_temp[tk])

    # Convert lists to numpy arrays
    input_array = np.array(input, dtype=np.float32)
    output_array = np.array(output, dtype=np.float32)
    physics_array = np.array(physics, dtype=np.float32) if training else None

    return (
        input_array,
        output_array,
        physics_array
    )

# def create_dataset(data: List, config, input_type, training=True):

#     num_of_datasets = len(data)

#     input = []
#     output = []
#     physics = []
#     joint_vel = []

#     Fc = config["Fc"]
#     Fs = config["Fs"]
#     Fv = config["Fv"]
#     Vs = config["Vs"]
#     alpha = 1e-10

#     print("Input type")
#     print(input_type)

#     for j in range(num_of_datasets):
#         s_temp = np.array(data[j]["s"]).flatten()
#         s_temp_motor_side = config["gear_ratio"] * np.array(data[j]["s"]).flatten()
#         ds_temp = np.array(data[j]["ds"]).flatten()
#         dds_temp = np.array(data[j]["dds"]).flatten()
#         im_temp = np.array(data[j]["im_filtered"]).flatten()
#         theta_temp = np.array(data[j]["theta"]).flatten()
#         omega_temp = np.array(data[j]["omega"]).flatten()
#         omega_dot_temp = np.array(data[j]["omega_dot"]).flatten()
#         delta = theta_temp[0] - s_temp_motor_side[0]
#         theta_temp = theta_temp - delta
#         taum_temp = config["ktau"] * config["gear_ratio"] * im_temp
#         tauj_temp = np.array(data[j]["tauj"]).flatten()
#         tauf_temp = taum_temp - tauj_temp
#         history_size = config["history_size"]

#         chunks = np.array(data[j]["chunks"]).flatten()

#         # Process the data, respecting chunk boundaries
#         start_index = 0
#         for i in range(1, len(chunks)):
#             # If the chunk changes, reset the start_index
#             if chunks[i] != chunks[start_index]:
#                 start_index = i
#                 continue  # No history can be formed yet, skip this iteration

#             tk = i

#             # Ensure there's enough history within the same chunk
#             if tk - start_index < history_size - 1:
#                 continue  # Not enough history, skip

#             if (
#                 np.abs(dds_temp[tk]) < config["joint_acc_limit"]
#                 and np.abs(im_temp[tk]) < config["im_limit"]
#                 and np.abs(tauf_temp[tk]) < config["tauf_limit"]
#                 and np.abs(ds_temp[tk]) < config["joint_vel_limit"]
#                 # and tk not in discard_indices
#             ):

#                 if input_type == 1: # input is [rs - theta, ds]
#                     s_history = s_temp_motor_side[tk - history_size + 1: tk + 1].flatten()
#                     ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()
#                     theta_history = theta_temp[tk - history_size + 1: tk + 1].flatten()

#                     input.append(np.concatenate((s_history-theta_history, ds_history), axis=0))
                
#                 elif input_type == 2: # input is [omega, ds]
#                     omega_hist = omega_temp[tk - history_size + 1: tk + 1].flatten()
#                     ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()

#                     input.append(np.concatenate((omega_hist, ds_history), axis=0))
                
#                 elif input_type == 3: # input is [(sr-theta), omega, ds]
#                     s_history = s_temp_motor_side[tk - history_size + 1: tk + 1].flatten()
#                     theta_history = theta_temp[tk - history_size + 1: tk + 1].flatten()
#                     omega_hist = omega_temp[tk - history_size + 1: tk + 1].flatten()
#                     ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()

#                     input.append(np.concatenate((s_history-theta_history, omega_hist, ds_history), axis=0))

#                 elif input_type == 4: # input is [(sr-theta), s, ds]
#                     s_history = s_temp_motor_side[tk - history_size + 1: tk + 1].flatten()
#                     theta_history = theta_temp[tk - history_size + 1: tk + 1].flatten()
#                     s_hist_joint_side = s_temp[tk - history_size + 1: tk + 1].flatten()
#                     ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()

#                     input.append(np.concatenate((s_history-theta_history, s_hist_joint_side, ds_history), axis=0))
                
#                 elif input_type == 5: # input is [theta, omega, s, ds]
#                     theta_history = theta_temp[tk - history_size + 1: tk + 1].flatten()
#                     omega_hist = omega_temp[tk - history_size + 1: tk + 1].flatten()
#                     s_hist_joint_side = s_temp[tk - history_size + 1: tk + 1].flatten()
#                     ds_history = ds_temp[tk - history_size + 1: tk + 1].flatten()

#                     input.append(np.concatenate((theta_history, omega_hist, s_hist_joint_side, ds_history), axis=0))

#                 if training:
#                     ds = ds_temp[tk].flatten()

#                     stribeck_term = (Fs - Fc) * np.exp(-(ds / Vs) ** 2)

#                     tanh_term = np.tanh(ds / alpha)

#                     tauF = (Fc + stribeck_term) * tanh_term + Fv * ds

#                     physics.append(tauF)

#                 # Prepare output
#                 tauF_k = tauf_temp[tk].flatten()
#                 output.append(tauF_k)

#     return (
#         np.array(input, dtype=np.float32),
#         np.array(output, dtype=np.float32),
#         np.array(physics, dtype=np.float32)
#     )

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


