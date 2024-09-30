function dataset = resampled_dataset_fun(dataset, dt, covariances_kinematics, covariances_current)
    description_list = dataset.description_list;
    yarp_robot_name = dataset.yarp_robot_name;
    moving_joint = dataset.moving_joints(1);

    chunks = split_vector(dataset);
    disp(['Number of chunks: ', num2str(length(chunks))]);
    dataset = private_resampled_dataset_fun(dataset, dt, chunks);

    % apply a filter to the position, velocity and acceleration
    % find the index of the moving joint in the description list 

    idx = find(strcmp(description_list, moving_joint));

    disp('Applying filter to the position, velocity and acceleration');
    [~, velocity, acceleration] = apply_filter_kinematics_for_all_chunks(dataset.joints_state.positions, idx, dt, covariances_kinematics);
    dataset.joints_state.velocities.filtered_data = reshape(velocity, 1, 1, size(velocity, 1));
    dataset.joints_state.accelerations.filtered_data = reshape(acceleration, 1, 1, size(acceleration, 1));

    [~, velocity, acceleration] = apply_filter_kinematics_for_all_chunks(dataset.motors_state.positions, idx, dt, covariances_kinematics);
    dataset.motors_state.velocities.filtered_data = reshape(velocity, 1, 1, size(velocity, 1));
    dataset.motors_state.accelerations.filtered_data = reshape(acceleration, 1, 1, size(acceleration, 1));

    % apply a filter to the current
    %disp('Applying filter to the current');
    %current = apply_filter_current_for_all_chunks(dataset.motors_state.currents, 1, covariances_current);
    %dataset.motors_state.currents.filtered_data = reshape(current, 1, 1, size(current, 1));

    dataset.description_list = description_list;
    dataset.yarp_robot_name = yarp_robot_name;
    dataset.moving_joints = {moving_joint};
end

function chunks = split_vector(dataset)
    timestamps = dataset.current_control.motors.desired.current.timestamps;
    channel = size(dataset.current_control.motors.desired.current.data, 1);

    if channel > 1
        error('The signal must be a vector. The number of channels is greater than 1');
    end

    dt = diff(timestamps);
    
    % the chunk starts when the difference between two consecutive timestamps is greater than 1s
    % find the indices of the timestamps that are greater than 1s
    idx = find(dt > 1);

    chunks = {};
    start = timestamps(1);
    for i = 1:length(idx)
        chunks{end+1} = [start, timestamps(idx(i))];
        start = timestamps(idx(i) + 1);
    end
end

function [dataset, t0, tf] = get_time_limits(dataset, t0, tf)
   % iterate over key and value of the dataset
   fn = fieldnames(dataset);
   for i = 1 : numel(fn)
       % if the field is not a structure continue
       if ~isstruct(dataset.(fn{i}))
           continue;
       end
       % check if one of the key of the field is data
       if isfield(dataset.(fn{i}), 'data')
           % resample the signal
            t0 = max(t0, min(dataset.(fn{i}).timestamps));
            tf = min(tf, max(dataset.(fn{i}).timestamps));
       else
           % call the function recursively
           [dataset.(fn{i}), t0, tf] = get_time_limits(dataset.(fn{i}), t0, tf);
       end
   end
end

function dataset = private_resampled_dataset_fun(dataset, dt, chunks)
    % iterate over key and value of the dataset
    fn = fieldnames(dataset);
    for i = 1 : numel(fn)
        % if the field is not a structure continue
        if ~isstruct(dataset.(fn{i})) | isfield(dataset.(fn{i}), 'current_control')
            continue;
        end
        % check if one of the key of the field is data
        if isfield(dataset.(fn{i}), 'data')
            % resample the signal
            dataset.(fn{i}) = resample_signal(dataset.(fn{i}), dt, chunks);
        else
            % call the function recursively
            dataset.(fn{i}) = private_resampled_dataset_fun(dataset.(fn{i}), dt, chunks);
        end
    end
end

% resampled a signal to a given dt
function signal = resample_signal(signal, dt, chunks)
    % get the time vector
    channel = size(signal.data, 1);

    timestamps = squeeze(signal.timestamps)';
    data = reshape(signal.data, channel, length(timestamps))';

    % for each chunk resample the data
    chunks_vector = [];
    resampled_data_vector = [];
    new_timestamp_vector = [];
    for i = 1:length(chunks)
        t0 = chunks{i}(1);
        tf = chunks{i}(2);

        % get the data
        new_time = [t0:dt:tf]';

        % append a vector with length(new_time) elements equal to i
        chunks_vector = [chunks_vector, i * ones(1, length(new_time))];

        % resample the data
        resampled_data = interp1(timestamps, data, new_time, 'linear');
        resampled_data_vector = [resampled_data_vector; resampled_data];
        new_timestamp_vector = [new_timestamp_vector; new_time];        
    
    end
    
    signal.data = reshape(resampled_data_vector', channel, 1, length(new_timestamp_vector));
    signal.timestamps = new_timestamp_vector';
    signal.dimensions = size(signal.data);
    signal.chunks = chunks_vector;
end

function [position, velocity, acceleration] = apply_filter_kinematics_for_all_chunks(dataset, position_idx, dt, covariances_kinematics)
    chunks = dataset.chunks;
    position = squeeze(dataset.data(position_idx, :, :));
    velocity = squeeze(dataset.data(position_idx, :, :));
    acceleration = squeeze(dataset.data(position_idx, :, :));

    % chunk is a vector that contains the indices of the chunks. Values of the same chunk are equal values in the chunk vector
    unique_chunks = unique(chunks);
    for i = 1:length(unique_chunks)
        display(['[Kinematics] Apply UKS for chunk: ', num2str(unique_chunks(i))]);
        idx = find(chunks == unique_chunks(i));
        [position(idx,:), velocity(idx,:), acceleration(idx,:)] = apply_filter_kinematics(position(idx,:), dt, covariances_kinematics);
    end
end

function [current] = apply_filter_current_for_all_chunks(dataset, current_idx, covariances_current)
    chunks = dataset.chunks;
    current = squeeze(dataset.data(current_idx, :, :));

    % chunk is a vector that contains the indices of the chunks. Values of the same chunk are equal values in the chunk vector
    unique_chunks = unique(chunks);
    for i = 1:length(unique_chunks)
        display(['[Current] Apply UKS for chunk: ', num2str(unique_chunks(i))]);
        idx = find(chunks == unique_chunks(i));
        current(idx,:) = apply_filter_current(current(idx,:), covariances_current);
    end
end

function [position, velocity, acceleration] = apply_filter_kinematics(position, dt, covariances_kinematics)

    function x = transition_fun(x, u, dt)

        % simple kinematic model
        A = [1, dt, 0.5*dt^2;...
             0, 1, dt;...
             0, 0, 1];
        x = A * x;
    end

    % Create the UKF object
    f_trans = @(x, u) transition_fun(x, u, dt);
    f_meas = @(x) [1, 0, 0] * x;

    ukf = UKS('f_meas', f_meas, ...
              'f_trans', f_trans, ...
              'initial_covariance', covariances_kinematics.P, ...
              'measurement_noise', covariances_kinematics.R, ...
              'process_noise', covariances_kinematics.Q);

    x0 = [position(1); 0; 0];
    [xs, ~] = ukf.smoothing_step(x0, position, zeros(1, length(position)));

    for i = 1:size(xs, 1)
        velocity(i,:) = xs{i}(2);
        acceleration(i,:) = xs{i}(3);
    end

    % Reconstruct velocity as recon_vel(i) = recon_vel(i-1) + dt * acceleration(i-1)
    % and position as recon_pos(i) = recon_pos(i-1) + dt * velocity(i-1) + 0.5 * dt^2 * acceleration(i-1)
    % and position as double integration of acceleration (recon_pos(i) = recon_pos(i-1) + dt * velocity(i-1) + 0.5 * dt^2 * acceleration(i-1))
    recon_vel = zeros(size(velocity));
    recon_pos = zeros(size(position));
    recon_pos(1) = position(1);
    recon_pos_from_acc = zeros(size(position));
    recon_pos_from_acc(1) = position(1);

    for i = 2:length(velocity)
        recon_vel(i) = recon_vel(i-1) + dt * acceleration(i-1);
        recon_pos(i) = recon_pos(i-1) + dt * velocity(i-1) + 0.5 * dt^2 * acceleration(i-1);
        recon_pos_from_acc(i) = recon_pos_from_acc(i-1) + dt * recon_vel(i-1) + 0.5 * dt^2 * acceleration(i-1);
    end

    % Plot 3 subplots
    % 1. acceleration from uks
    % 2. velocity from uks and reconstructed velocity
    % 3. position from uks, reconstructed position and position from double integration of acceleration

    figure;
    subplot(3,1,1);
    plot(acceleration);
    subplot(3,1,2);
    plot(velocity);
    hold on;
    plot(recon_vel);
    leg = legend('Velocity from UKS', 'Reconstructed velocity from acceleration integration');
    set(leg, 'Location', 'best');
    subplot(3,1,3);
    plot(position);
    hold on;
    plot(recon_pos);
    hold on;
    plot(recon_pos_from_acc);
    leg = legend('Position from robot', 'Reconstructed  from velocity integration', 'Position from acceleration double integration');
    set(leg, 'Location', 'best');
    hold off;

end

function [current] = apply_filter_current(current, covariances_current)

    function x = transition_fun(x, ~)
        % simple kinematic model
        A = [1];
        x = A * x;
    end

    % Create the UKF object
    f_trans = @(x, u) transition_fun(x, u);
    f_meas = @(x) [1] * x;

    ukf = UKS('f_meas', f_meas, ...
              'f_trans', f_trans, ...
              'initial_covariance', 0.1 * eye(1), ...
              'measurement_noise', 1e-3 * eye(1), ...
              'process_noise', diag([1.0]));

    x0 = [current(1)];
    [xs, ~] = ukf.smoothing_step(x0, current, zeros(1, length(current)));

    orig_current = current;

    for i = 1:size(xs, 1)
        current(i,:) = xs{i}(1);
    end
    
    % Plot raw current and filtered current
    figure;
    plot(orig_current);
    hold on;
    plot(current);
    leg = legend('Current from robot', 'Filtered current');
    set(leg, 'Location', 'best');
end
