import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":

    joint = "r_ankle_pitch"

    folders = [
        "/home/isorrentino/dev/dataset/friction/r_ankle_pitch/ramp/parsed",
        "/home/isorrentino/dev/dataset/friction/r_ankle_pitch/sinusoid/parsed"
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

    gear_ratio = [
        -100.0,
        160.0,
        -100.0,
        -100.0,
        -100.0,
        -160.0,
        100.0,
        -160.0,
        100.0,
        100.0,
        100.0,
        160.0,
    ]
    ktau = [
        0.156977705,
        0.066468037,
        0.066468037,
        0.156977705,
        0.156977705,
        0.066468037,
        0.156977705,
        0.066468037,
        0.066468037,
        0.156977705,
        0.156977705,
        0.066468037,
    ] # Datasheet
    jnt_lim_inf = [
        -0.78168556,
        -0.61055556,
        -1.42433889,
        -1.83567889,
        -0.81332978,
        -0.44157122,
        -0.78168556,
        -0.61055556,
        -1.41436067,
        -1.83567889,
        -0.80139778,
        -0.44982244,
    ]
    jnt_lim_sup = [
        1.92150556,
        1.94976556,
        1.42433889,
        0.13519444,
        0.81332978,
        0.44157122,
        1.92150556,
        1.94441011,
        1.41436067,
        0.13519444,
        0.80139778,
        0.44982244,
    ]
    Jm = [
        0.2348,
        0.0827,
        0.0827,
        0.2348,
        0.2348,
        0.0827,
        0.2348,
        0.0827,
        0.0827,
        0.2348,
        0.2348,
        0.0827,
    ]
    Jm = np.array(Jm) * 1e-4 # Convert to kg*m^2

    Jh = [
        0.054,
        0.054,
        0.054,
        0.054,
        0.054,
        0.054,
        0.054,
        0.054,
        0.054,
        0.054,
        0.054,
        0.054
    ]
    Jh = np.array(Jh) * 1e-4 # Convert to kg*m^2

    T1 = [
        3.9,
        3.9,
        3.9,
        3.9,
        3.9,
        3.9,
        3.9,
        3.9,
        3.9,
        3.9,
        3.9,
        3.9
    ]

    T2 = [
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        12.0
    ]

    Ke1 = [
        0.84,
        0.84,
        0.84,
        0.84,
        0.84,
        0.84,
        0.84,
        0.84,
        0.84,
        0.84,
        0.84,
        0.84
    ]
    Ke1 = np.array(Ke1) * 1e4 # Convert to Nm/rad

    Ke2 = [
        0.94,
        0.94,
        0.94,
        0.94,
        0.94,
        0.94,
        0.94,
        0.94,
        0.94,
        0.94,
        0.94,
        0.94
    ]
    Ke2 = np.array(Ke2) * 1e4 # Convert to Nm/rad

    Ke3 = [
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3
    ]
    Ke3 = np.array(Ke3) * 1e4 # Convert to Nm/rad

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

    i = 0

    for d in data:
        # Discard samples where the joint position is outside the limits
        # Save the indeces and discard them
        # Combine all conditions into a single mask
        mask = (
            (np.array(d["s"]) >= (jnt_lim_inf[jnt_idx_here] + 0.0)) & 
            (np.array(d["s"]) <= (jnt_lim_sup[jnt_idx_here] - 0.0)) &
            (np.abs(np.array(d["ds"])) < 10.0) &
            (np.abs(np.array(d["dds"])) < 10.0) &
            (np.abs(np.array(d["im_filtered"])) < 10.0)
        )

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

        # Inertia contribution
        inertia_contribution = pow(gear_ratio[jnt_idx_here], 2) * (Jm[jnt_idx_here] + Jh[jnt_idx_here]) * np.array(d3["dds"]).flatten()

        # Compute torque due to torsional spring as
        # tau_spring = Ke1 * (theta/r - s) if tauj > 0 && tauj < T1
        # tau_spring = Ke2 * (theta/r - s) if tauj > T1 && tauj < T2
        # tau_spring = Ke3 * (theta/r - s) if tauj > T2
        # where r is the gear ratio
        theta_temp = np.array(d3["theta"]).flatten()
        theta_temp = theta_temp - theta_temp[0]
        s_temp = np.array(d3["s"]).flatten()
        s_temp = s_temp - s_temp[0]
        tauj_temp = np.array(d3["tauj"]).flatten()
        abs_tauj_temp = np.abs(tauj_temp)
        tau_spring = np.zeros_like(tauj_temp)
        error = np.zeros_like(tauj_temp)
        error[1:] = np.diff(theta_temp) / gear_ratio[jnt_idx_here]

        for idx in range(len(tauj_temp)):
            if abs_tauj_temp[idx] > 0 and abs_tauj_temp[idx] < T1[jnt_idx_here]:
                tau_spring[idx] = Ke1[jnt_idx_here] * (theta_temp[idx]/gear_ratio[jnt_idx_here] - s_temp[idx])
            elif abs_tauj_temp[idx] > T1[jnt_idx_here] and abs_tauj_temp[idx] < T2[jnt_idx_here]:
                tau_spring[idx] = Ke2[jnt_idx_here] * (theta_temp[idx]/gear_ratio[jnt_idx_here] - s_temp[idx])
            elif abs_tauj_temp[idx] > T2[jnt_idx_here]:
                tau_spring[idx] = Ke3[jnt_idx_here] * (theta_temp[idx]/gear_ratio[jnt_idx_here] - s_temp[idx])

        # for idx in range(len(tauj_temp)):
        #     if abs_tauj_temp[idx] > 0 and abs_tauj_temp[idx] < T1[jnt_idx_here]:
        #         tau_spring[idx] = Ke1[jnt_idx_here] * (error[idx])
        #     elif abs_tauj_temp[idx] > T1[jnt_idx_here] and abs_tauj_temp[idx] < T2[jnt_idx_here]:
        #         tau_spring[idx] = Ke2[jnt_idx_here] * (error[idx])
        #     elif abs_tauj_temp[idx] > T2[jnt_idx_here]:
        #         tau_spring[idx] = Ke3[jnt_idx_here] * (error[idx])
 
        tauf_temp = motor_torque - tauj_temp
         
        # Plot tau_spring
        # plt.figure(figsize=(8, 6))
        # plt.plot(tau_spring)
        # plt.ylabel(r"$\tau_{spring}$ (Nm)")
        # plt.title(joints[jnt_idx_here])
        # plt.grid(True)
        # plt.show()

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

    plt.figure(figsize=(8, 6))
    plt.scatter(ds[:len(ds)], tauF_from_filtered[:len(ds)], c=np.abs(im_filtered[:len(ds)]), cmap="YlOrRd", alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label("Motor Current (A)")
    plt.xlabel("ds (rad/sec)")
    plt.ylabel(r"$\tau_F$ (Nm)")
    plt.title(joints[jnt_idx_here])
    plt.grid(True)

    # Plot theta / r - s
    # Use time as x-axis knowing that the samples are taken at 1kHz
    # plt.figure(figsize=(8, 6))
    # plt.plot((np.array(theta)/gear_ratio[jnt_idx_here] - s))
    # plt.ylabel(r"$\theta/r - s$ (rad)")
    # plt.title(joints[jnt_idx_here])
    # plt.grid(True)

    # plt.figure()
    # plt.hist(ds, bins=100)

    plt.show()
