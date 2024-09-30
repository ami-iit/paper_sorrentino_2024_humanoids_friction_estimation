function reduce_and_resample_dataset(dataset_folder, moving_joint, associated_joints)


    % diag_cov = dictionary('torso_roll' , {[1e-1, 1e-1, 1e1]}, ...
    %                       'torso_yaw'  , {[1e-1, 1e1, 1e1]}, ...
    %                         'l_hip_pitch', {[1e-3, 1e0, 1e1]}, ...
    %                         'l_hip_roll' , {[1e-2, 1e0, 1e2]}, ...
    %                         'l_hip_yaw'  , {[1e-2, 1e0, 1e2]}, ...
    %                         'l_knee'     , {[1e-4, 1e0, 1e2]}, ...
    %                         'l_ankle_pitch', {[1e-4, 1e0, 1e2]}, ...
    %                         'l_ankle_roll', {[1e-4, 1e0, 1e4]}, ...
    %                         'r_hip_pitch', {[1e-1, 1e-1, 1e1]}, ...
    %                         'r_hip_roll' , {[1e-1, 1e-1, 1e1]}, ...
    %                         'r_hip_yaw'  , {[1e-2, 1e0, 1e2]}, ...
    %                         'r_knee'     , {[1e-1, 1e-1, 1e1]}, ...
    %                         'r_ankle_pitch', {[1e-3, 1e-1, 1e3]}, ...
    %                         'r_ankle_roll', {[1e-2, 1e-2, 1e3]});
    % 
    % 
    % meas_cov = dictionary('torso_roll' , 1e-2, ...
    %                       'torso_yaw'  , 1e-2, ...
    %                         'l_hip_pitch', 1e-2, ...
    %                         'l_hip_roll' , 1e-2, ...
    %                         'l_hip_yaw'  , 1e-2, ...
    %                         'l_knee'     , 1e-4, ...
    %                         'l_ankle_pitch', 1e-4, ...
    %                         'l_ankle_roll', 1e-4, ...
    %                         'r_hip_pitch', 1e-2, ...
    %                         'r_hip_roll' , 1e-2, ...
    %                         'r_hip_yaw'  , 1e-2, ...
    %                         'r_knee'     , 1e-2, ...
    %                         'r_ankle_pitch', 1e-4, ...
    %                         'r_ankle_roll', 1e-4);
    % 
    % process_cov = dictionary('torso_roll' , {[1e-4, 1e0, 1e1]}, ...
    %                          'torso_yaw'  , {[1e-4, 1e0, 1e1]}, ...
    %                          'l_hip_pitch', {[1e-4, 1e0, 1e1]}, ...
    %                          'l_hip_roll' , {[1e-4, 5e1, 5e2]}, ...
    %                          'l_hip_yaw'  , {[1e-4, 1e0, 1e2]}, ...
    %                          'l_knee'     , {[1e-5, 1e-1, 1e2]}, ...
    %                          'l_ankle_pitch', {[1e-5, 5e0, 1e3]}, ...
    %                          'l_ankle_roll', {[1e-5, 5e0, 1e5]}, ...
    %                          'r_hip_pitch', {[1e-4, 1e0, 1e1]}, ...
    %                          'r_hip_roll' , {[1e-4, 1e0, 1e1]}, ...
    %                          'r_hip_yaw'  , {[1e-4, 1e0, 1e2]}, ...
    %                          'r_knee'     , {[1e-4, 1e0, 1e1]}, ...
    %                          'r_ankle_pitch', {[1e-3, 1e-1, 1e3]}, ...
    %                          'r_ankle_roll', {[1e-2, 1e-2, 1e3]});

    diag_cov = dictionary('torso_roll' , {[1e-1, 1e-1, 1e1]}, ...
                          'torso_yaw'  , {[1e-1, 1e1, 1e1]}, ...
                            'l_hip_pitch', {[1e-3, 1e0, 1e1]}, ...
                            'l_hip_roll' , {[1e-2, 1e0, 1e2]}, ...
                            'l_hip_yaw'  , {[1e-2, 1e0, 1e2]}, ...
                            'l_knee'     , {[1e-4, 1e0, 1e2]}, ...
                            'l_ankle_pitch', {[1e-4, 1e0, 1e2]}, ...
                            'l_ankle_roll', {[1e-4, 1e0, 1e4]}, ...
                            'r_hip_pitch', {[1e-1, 1e-1, 1e1]}, ...
                            'r_hip_roll' , {[1e-1, 1e-1, 1e1]}, ...
                            'r_hip_yaw'  , {[1e-2, 1e0, 1e2]}, ...
                            'r_knee'     , {[1e-1, 1e-1, 1e1]}, ...
                            'r_ankle_pitch', {[1e-5, 1e-3, 1e3]}, ...
                            'r_ankle_roll', {[1e-5, 1e-4, 1e3]});


    meas_cov = dictionary('torso_roll' , 1e-2, ...
                          'torso_yaw'  , 1e-2, ...
                            'l_hip_pitch', 1e-2, ...
                            'l_hip_roll' , 1e-2, ...
                            'l_hip_yaw'  , 1e-2, ...
                            'l_knee'     , 1e-4, ...
                            'l_ankle_pitch', 1e-4, ...
                            'l_ankle_roll', 1e-4, ...
                            'r_hip_pitch', 1e-2, ...
                            'r_hip_roll' , 1e-2, ...
                            'r_hip_yaw'  , 1e-2, ...
                            'r_knee'     , 1e-2, ...
                            'r_ankle_pitch', 1e-4, ...
                            'r_ankle_roll', 1e-4);

    process_cov = dictionary('torso_roll' , {[1e-4, 1e0, 1e1]}, ...
                             'torso_yaw'  , {[1e-4, 1e0, 1e1]}, ...
                             'l_hip_pitch', {[1e-4, 1e0, 1e1]}, ...
                             'l_hip_roll' , {[1e-4, 5e1, 5e2]}, ...
                             'l_hip_yaw'  , {[1e-4, 1e0, 1e2]}, ...
                             'l_knee'     , {[1e-5, 1e-1, 1e2]}, ...
                             'l_ankle_pitch', {[1e-5, 5e0, 1e3]}, ...
                             'l_ankle_roll', {[1e-5, 5e0, 1e5]}, ...
                             'r_hip_pitch', {[1e-4, 1e0, 1e1]}, ...
                             'r_hip_roll' , {[1e-4, 1e0, 1e1]}, ...
                             'r_hip_yaw'  , {[1e-4, 1e0, 1e2]}, ...
                             'r_knee'     , {[1e-4, 1e0, 1e1]}, ...
                             'r_ankle_pitch', {[1e-5, 1e-3, 1e3]}, ...
                             'r_ankle_roll', {[1e-5, 1e-4, 1e3]});
               
    
    % check if the moving joint is in the list of associated joints
    if ~any(strcmp(associated_joints, moving_joint))
        error('The moving joint is not in the list of associated joints');
    end

    % check if the moving joint is in the dictionary of covariances
    if ~isKey(diag_cov, moving_joint)
        error('The moving joint is not in the dictionary of covariances');
    end

    % find the covariances for the moving joint
    covariances_kinematics.P = diag(diag_cov{moving_joint});
    covariances_kinematics.Q = diag(process_cov{moving_joint});
    covariances_kinematics.R = diag(meas_cov(moving_joint));



    disp('Reducing and resampling dataset...');


    covariances_current.P = 0.1 * eye(1);
    covariances_current.Q = 1e-1 * eye(1);
    covariances_current.R = 1.0e-1;

    moving_joints = {moving_joint};
    % associated_joints = {'torso_pitch', 'torso_roll', 'torso_yaw', 'l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'};

    associated_joints

    dt = 0.001;

    %%%% DO NOT MODIFY BELOW THIS LINE %%%%

    % Get the list of all .mat files in the folder
    files = dir(fullfile(dataset_folder, '*.mat'));

    % Loop over all the files
    for i = 1:length(files)
        % Get the full path of the file
        dataset = fullfile(dataset_folder, files(i).name);
        disp(['Processing dataset: ', dataset]);

        % remove the .mat extension
        dataset_path = extractBefore(dataset, ".mat");
        reduced_dataset = strcat(dataset_path, '_reduced.mat');
        resampled_dataset = strcat(dataset_path, '_resampled.mat');

        robot_logger_device = reduce_dataset_fun(dataset, moving_joints, associated_joints);
        save(reduced_dataset, 'robot_logger_device', '-v7.3');

        robot_logger_device = resampled_dataset_fun(robot_logger_device, dt, covariances_kinematics, covariances_current);
        robot_logger_device.covariances_kinematics = covariances_kinematics;
        robot_logger_device.covariances_current = covariances_current;

        save(resampled_dataset, 'robot_logger_device', '-v7.3');
    end
end