joints = {'torso_pitch', 'torso_roll', 'torso_yaw', 'l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'};

reduce_and_resample_dataset("/home/isorrentino/dev/dataset/friction/r_ankle_pitch/sinusoid", 'r_ankle_pitch', joints);