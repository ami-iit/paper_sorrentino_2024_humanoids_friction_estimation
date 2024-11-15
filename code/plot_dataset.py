import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":

    joint = "r_ankle_pitch"

    folders = [
        # "/home/isorrentino/dev/dataset/friction/l_hip_roll/ramp/parsed",
        # "/home/isorrentino/dev/dataset/friction/l_hip_roll/sinusoid/parsed"
        "/home/isorrentino/dev/dataset/friction/r_ankle_pitch/parsed"
        ]

    joints = [
        "l_hip_pitch",
        "l_hip_roll",
        "l_hip_yaw",
        "l_knee",
        "l_ankle_pitch",
        "l_ankle_roll",
        "r_hip_pitch",
        "r_hip_roll",
        "r_hip_yaw",
        "r_knee",
        "r_ankle_pitch",
        "r_ankle_roll",
    ]

    gear_ratio = [-100.0, 160.0, -100.0, -100.0, -100.0, -160.0,
                  100.0, -160.0, 100.0, 100.0, 100.0, 160.0]

    ktau = [0.111, 0.047, 0.047, 0.111, 0.111, 0.047,
            0.111, 0.047, 0.047, 0.111, 0.111, 0.047] # Datasheet
    
    jnt_lim_inf = [-0.78168556, -0.61055556, -1.42433889, -1.83567889, -0.81332978, -0.44157122,
                   -0.78168556, -0.61055556, -1.41436067, -1.83567889, -0.80139778, -0.44982244]
    
    jnt_lim_sup = [1.92150556, 1.94976556, 1.42433889, 0.13519444, 0.81332978, 0.44157122,
                   1.92150556, 1.94441011, 1.41436067, 0.13519444, 0.80139778, 0.44982244]

    jnt_idx_here = joints.index(joint)

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

    # Concatenate all data
    tauF = []
    tauF_with_inertia = []
    tauF_from_filtered = []
    tauF_from_desired = []
    im = []
    im_filtered = []
    im_desired = []
    s = []
    ds = []
    dds = []
    omega = []
    omega_dot = []
    motor_temperature = []
    Mnudot = []
    h = []
    tauj = []
    theta = []

    keys = ["s", "ds", "dds", "im", "im_filtered", "im_desired", "tauj", "theta", "omega", "omega_dot"]
    # keys = ["s", "ds", "dds", "im", "im_filtered", "im_desired", "tauj", "theta", "omega", "omega_dot", "r_front_contact", "r_rear_contact", "r_leg_contact"]

    i = 0

    for d in data:
        # Discard samples where the joint position is outside the limits
        # Save the indeces and discard them
        # Combine all conditions into a single mask
        mask = (
            (np.array(d["s"]) >= (jnt_lim_inf[jnt_idx_here] + 0.0)) & 
            (np.array(d["s"]) <= (jnt_lim_sup[jnt_idx_here] - 0.0)) &
            (np.abs(np.array(d["ds"])) < 10.0) &
            # (np.array(d["ds"]) > 0.2) &
            # (np.array(d["ds"]) > 0.23) &
            (np.abs(np.array(d["dds"])) < 10.0) &
            (np.abs(np.array(d["im_filtered"])) < 4.0) &
            (np.abs(np.array(d["tauj"])) < 50)
        )
        # mask = (
        #     (np.array(d["s"]) >= (jnt_lim_inf[jnt_idx_here] + 0.0)) & 
        #     (np.array(d["s"]) <= (jnt_lim_sup[jnt_idx_here] - 0.0)) &
        #     (np.abs(np.array(d["ds"])) < 10.0) &
        #     # (np.array(d["ds"]) > 0.2) &
        #     # (np.array(d["ds"]) > 0.23) &
        #     (np.abs(np.array(d["dds"])) < 10.0) &
        #     (np.abs(np.array(d["im_filtered"])) < 50) &
        #     (np.abs(np.array(d["tauj"])) < 50) &
        #     (np.linalg.norm(np.array(d["r_front_contact"][:, :2]), axis=1) +
        #     (np.linalg.norm(np.array(d["r_rear_contact"][:, :2]), axis=1)) < 1.3)
        # )

        # Apply the combined mask to filter data
        d3 = {key: np.array(d[key])[mask] for key in keys}

        # Check if d3["ds"] has samples
        if len(d3["ds"]) == 0:
            continue

        tauF.extend(
            ktau[jnt_idx_here]
            * gear_ratio[jnt_idx_here]
            * np.array(d3["im"]).flatten()
            - np.array(d3["tauj"]).flatten()
        )

        # Compute motor torque
        motor_torque = ktau[jnt_idx_here] * gear_ratio[jnt_idx_here] * np.array(d3["im_filtered"]).flatten()

        theta_temp = np.array(d3["theta"]).flatten()
        theta_temp = theta_temp - theta_temp[0]
        s_temp = np.array(d3["s"]).flatten()
        s_temp = s_temp - s_temp[0]
        tauj_temp = np.array(d3["tauj"]).flatten()
        abs_tauj_temp = np.abs(tauj_temp)
        tau_spring = np.zeros_like(tauj_temp)
        error = np.zeros_like(tauj_temp)
        error[1:] = np.diff(theta_temp) / gear_ratio[jnt_idx_here]

        tauf_temp = motor_torque - tauj_temp
        
        tauF_from_filtered.extend(tauf_temp)

        tauF_from_desired.extend(
            ktau[jnt_idx_here]
            * gear_ratio[jnt_idx_here]
            * np.array(d3["im_desired"]).flatten()
            - np.array(d3["tauj"]).flatten()
        )
        im.extend(np.array(d3["im"]).flatten())
        im_filtered.extend(np.array(d3["im_filtered"]).flatten())
        im_desired.extend(np.array(d3["im_desired"]).flatten())
        s.extend(np.array(d3["s"]).flatten())
        ds.extend(np.array(d3["ds"]).flatten())
        dds.extend(np.array(d3["dds"]).flatten())
        tauj.extend(np.array(d3["tauj"]).flatten())
        delta = np.array(d3["theta"])[0].flatten() - gear_ratio[jnt_idx_here]*np.array(d3["s"])[0].flatten()
        theta.extend(np.array(d3["theta"]).flatten() - delta)

        i = i + 1

    end_index = 77000
    end_index = len(ds)

    plt.figure(figsize=(8, 6))
    plt.scatter(ds[:end_index], tauF_from_filtered[:end_index], c=np.abs(im_filtered[:end_index]), cmap="YlOrRd", alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label("Motor Current (A)")
    plt.xlabel("ds (rad/sec)")
    plt.ylabel(r"$\tau_F$ (Nm)")
    plt.title(joints[jnt_idx_here])
    plt.grid(True)

    # Plot samples vs joint position with scatter
    # plt.figure(figsize=(8, 6))
    # plt.scatter(np.linspace(0, len(s), len(s)), s)
    # plt.ylabel("Joint Position (rad)")
    # plt.title(joints[jnt_idx_here])

    # # Plot a 3D plot with the motor current on the x axis, ds on the y axis and tauj on the z axis
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(im_filtered[::10], ds[::10], tauj[0::10])
    # ax.set_xlabel("Motor Current (A)")
    # ax.set_ylabel("ds (rad/sec)")
    # ax.set_zlabel(r"$\tau_j$ (Nm)")
    # ax.set_title(joints[jnt_idx_here])
    # # set x axis between -1 and 1
    # ax.set_ylim(0.0, 0.7)

    # idx1 = 116000
    # idx2 = 345000
    # idx3 = 574000
    # idx4 = len(ds)

    # Plot 4 subplots dividing the dataset in 4 parts considering idx1, idx2, idx3 and idx4
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # axs[0, 0].scatter(ds[:idx1], tauF_from_filtered[:idx1], c=np.abs(im_filtered[:idx1]), cmap="YlOrRd", alpha=0.7)
    # axs[0, 0].set_title("First quarter")
    # axs[0, 0].set_xlabel("ds (rad/sec)")
    # axs[0, 0].set_ylabel(r"$\tau_F$ (Nm)")
    # axs[0, 0].grid(True)
    # axs[0, 1].scatter(ds[idx1:idx2], tauF_from_filtered[idx1:idx2], c=np.abs(im_filtered[idx1:idx2]), cmap="YlOrRd", alpha=0.7)
    # axs[0, 1].set_title("Second quarter")
    # axs[0, 1].set_xlabel("ds (rad/sec)")
    # axs[0, 1].set_ylabel(r"$\tau_F$ (Nm)")
    # axs[0, 1].grid(True)
    # axs[1, 0].scatter(ds[idx2:idx3], tauF_from_filtered[idx2:idx3], c=np.abs(im_filtered[idx2:idx3]), cmap="YlOrRd", alpha=0.7)
    # axs[1, 0].set_title("Third quarter")
    # axs[1, 0].set_xlabel("ds (rad/sec)")
    # axs[1, 0].set_ylabel(r"$\tau_F$ (Nm)")
    # axs[1, 0].grid(True)
    # axs[1, 1].scatter(ds[idx3:idx4], tauF_from_filtered[idx3:idx4], c=np.abs(im_filtered[idx3:idx4]), cmap="YlOrRd", alpha=0.7)
    # axs[1, 1].set_title("Fourth quarter")
    # axs[1, 1].set_xlabel("ds (rad/sec)")
    # axs[1, 1].set_ylabel(r"$\tau_F$ (Nm)")
    # axs[1, 1].grid(True)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(ds[:end_index], tauF_from_filtered[:end_index], c=np.abs(im_filtered[:end_index]), cmap="YlOrRd", alpha=0.7)
    # cbar = plt.colorbar()
    # cbar.set_label("Motor Current (A)")
    # plt.xlabel("ds (rad/sec)")
    # plt.ylabel(r"$\tau_F$ (Nm)")
    # plt.title(joints[jnt_idx_here])
    # plt.grid(True)

    # Plot time vs joint position
    # plt.figure(figsize=(8, 6))
    # plt.plot(s)
    # # plt.xlabel("Time (s)")
    # plt.ylabel("Joint Position (rad)")

    plt.show()
