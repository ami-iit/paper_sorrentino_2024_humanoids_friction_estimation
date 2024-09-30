function dataset = reduce_dataset_fun(dataset, moving_joints, associated_joints)
    % Load the dataset
    load(dataset);
    
    original_joints = robot_logger_device.description_list;

    % Find the indices of the moving joints
    moving_joints_indices = [];
    for i = 1:length(moving_joints)
        moving_joints_indices = [moving_joints_indices, find(strcmp(original_joints, moving_joints(i)))];
    end

    % Find the indices of the associated joints
    associated_joints_indices = [];
    for i = 1:length(associated_joints)
        associated_joints_indices = [associated_joints_indices, find(strcmp(original_joints, associated_joints(i)))];
    end

    % remove the cartesian_wrenches field from the structure
    robot_logger_device = check_and_remvove_field(robot_logger_device, 'cartesian_wrenches');

    % remove the temperature field from the structure
    robot_logger_device = check_and_remvove_field(robot_logger_device, 'temperatures');

    % remove the FTs field from the structure
    robot_logger_device = check_and_remvove_field(robot_logger_device, 'FTs');

    % remove the PWM field from the structure robot_logger_device.motors_state
    robot_logger_device.motors_state = check_and_remvove_field(robot_logger_device.motors_state, 'PWM');

    robot_logger_device.motors_state.currents = extract_joints(robot_logger_device.motors_state.currents, moving_joints_indices, moving_joints);
    robot_logger_device.motors_state.velocities = extract_joints(robot_logger_device.motors_state.velocities, moving_joints_indices, moving_joints);
    robot_logger_device.motors_state.positions = extract_joints(robot_logger_device.motors_state.positions, associated_joints_indices, associated_joints);
    robot_logger_device.motors_state.accelerations = extract_joints(robot_logger_device.motors_state.accelerations, moving_joints_indices, moving_joints);

    robot_logger_device.joints_state.positions = extract_joints(robot_logger_device.joints_state.positions, associated_joints_indices, associated_joints);
    robot_logger_device.joints_state.velocities = extract_joints(robot_logger_device.joints_state.velocities, moving_joints_indices, moving_joints);
    robot_logger_device.joints_state.accelerations = extract_joints(robot_logger_device.joints_state.accelerations, moving_joints_indices, moving_joints);

    robot_logger_device.joints_state = check_and_remvove_field(robot_logger_device.joints_state, 'torques');

    robot_logger_device.description_list = associated_joints';
    robot_logger_device.moving_joints = moving_joints;

    dataset = robot_logger_device;
end

function dataset = check_and_remvove_field(dataset, field_name)
    if isfield(dataset, field_name)
        dataset = rmfield(dataset, field_name);
    end
end

function dataset = extract_joints(dataset, joints_indices, joint_names)
    dataset.data = dataset.data(joints_indices,:, :);
    dataset.dimensions = [length(joints_indices), dataset.dimensions(2), dataset.dimensions(3)];
    dataset.elements_names = joint_names';
end
