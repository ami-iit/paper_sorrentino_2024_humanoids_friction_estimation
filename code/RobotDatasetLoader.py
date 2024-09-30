import h5py
import os
import numpy as np
import idyntree.bindings as idyn
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from rich.progress import track
import h5py

class RobotDatasetLoader:
    def __init__(
        self,
        config_robot: dict,
        config_dataset: dict
    ):

        self.model_path = config_robot["model_path"]
        self.base_link = config_robot["base_link"]
        self.contact_link = config_robot["contact_link"]
        self.ft_list = []
        self.dataset_type = config_dataset['dataset_type']
        self.set_base_from_imu = config_dataset['set_base_from_imu']
        self.dT = 0.0

        if self.set_base_from_imu:
            self.imu_as_base = config_dataset["imu_as_base"]

    def load_and_parse_dataset(self, dataset):
        f = h5py.File(dataset, "r")

        if self.dataset_type == "synthetic":
            data_logged = self._populate_numerical_data_synthetic(f)
            data_parsed = self._prepare_data_synthetic(data_logged)
        else:
            data_logged = self._populate_numerical_data(f)
            print("Data loaded. Preparing data...")
            data_parsed = self._prepare_data(data_logged)
            print("Data prepared.")

        data_parsed = data_parsed

        print("Generating additional data...")
        data_parsed = self._generate_additional_data(data_parsed)

        return data_parsed

    def _populate_numerical_data_synthetic(self, file_object):
        data = {}

        for key, value in file_object.items():
            if not (type(file_object[key]) is h5py._hl.group.Group):
                data[key] = np.squeeze(np.array(value)).T
            else:
                if key == ".." or key == "test":
                    data = self._populate_numerical_data_synthetic(file_object=value)
                else:
                    data[key] = self._populate_numerical_data_synthetic(
                        file_object=value
                    )

        return data

    def _populate_numerical_data(self, file_object):
        data = {}

        for key, value in file_object.items():
            if not isinstance(value, h5py._hl.group.Group):
                continue
            if key == "#refs#":
                continue
            if key == "log":
                continue
            if "data" in value.keys() or "filtered_data" in value.keys() or "chunks" in value.keys():
                data[key] = {}
                data[key]["data"] = np.squeeze(np.array(value["data"]))
                try:
                    data[key]["filtered_data"] = np.squeeze(np.array(value["filtered_data"]))
                except:
                    pass
                try:
                    data[key]["chunks"] = np.squeeze(np.array(value["chunks"]))
                except:
                    pass
                data[key]["timestamps"] = np.squeeze(np.array(value["timestamps"]))

                # In yarp telemetry v0.4.0 the elements_names was saved.
                if "elements_names" in value.keys():
                    elements_names_ref = value["elements_names"]
                    if elements_names_ref.shape[0] > 1:
                        elements_names_ref = np.transpose(elements_names_ref)
                    data[key]["elements_names"] = [
                        "".join(chr(c[0]) for c in value[ref])
                        for ref in elements_names_ref[0]
                    ]
            else:
                if key == "robot_logger_device":
                    data = self._populate_numerical_data(file_object=value)
                else:
                    data[key] = self._populate_numerical_data(file_object=value)

        return data

    def _prepare_data_synthetic(self, data_logged):
        data = dict()
        data["s"] = data_logged["sim"]["s"]
        data["ds"] = data_logged["sim"]["ds"]
        data["dds"] = data_logged["sim"]["dds"]
        data["im"] = data_logged["sim"]["i_m"]
        data["taum"] = data_logged["sim"]["tau_m"]
        data["theta"] = data_logged["sim"]["s"] * 100

        n_samples = data["s"].shape[0]

        data["base_pose"] = np.zeros((n_samples, 4, 4))
        data["base_vel"] = np.zeros((n_samples, 6))
        data["base_acc"] = np.zeros((n_samples, 6))
        for i in range(n_samples):
            data["base_pose"][i] = H_base = np.eye(4)
            data["base_vel"][i] = np.zeros(6)
            data["base_acc"][i] = np.zeros(6)

        data["joint_list"] = self.joint_list

        return data
    
    def _prepare_data(self, data_logged):
        data = dict()
        data['joint_list'] = data_logged["joints_state"]["positions"]["elements_names"]

        self.dT = 0.001

        data["time"] = data_logged["joints_state"]["positions"]["timestamps"]
        data["time"] = data["time"] - data["time"][0]

        self.joint_current_control = data_logged["current_control"]["motors"]["desired"]["current"]["elements_names"]
        data["im_desired"] = data_logged["current_control"]["motors"]["desired"]["current"]["data"]

        data["chunks"] = data_logged["current_control"]["motors"]["desired"]["current"]["chunks"]

        idx = data['joint_list'].index(self.joint_current_control[0])

        # Select from data_logged['joints_state']['positions']['data'] only the indeces where timestamp_walking elements match timestamp_logger elements
        data["s"] = data_logged["joints_state"]["positions"]["data"]
        data["ds"] = data_logged["joints_state"]["velocities"]["filtered_data"]
        data["dds"] = data_logged["joints_state"]["accelerations"]["filtered_data"]
        data["theta"] = data_logged["motors_state"]["positions"]["data"]
        data["omega"] = data_logged["motors_state"]["velocities"]["data"]
        data["omega_dot"] = data_logged["motors_state"]["accelerations"]["data"]
        data["im"] = data_logged["motors_state"]["currents"]["data"]
        data['acc'] = {}
        for acc in data_logged["accelerometers"].keys():
            data['acc'][acc] = data_logged["accelerometers"][acc]["data"]
        data['gyro'] = {}
        for gyro in data_logged["gyros"].keys():
            data['gyro'][gyro] = data_logged["gyros"][gyro]["data"]
        data["orientations"] ={}
        for orientation in data_logged["orientations"].keys():
            data["orientations"][orientation] = data_logged["orientations"][orientation]["data"]

        # Compute motor velocity and acceleration
        data["theta"] = data["theta"][:, idx]

        # Reconstruct joint position integrating joint acceleration and velocity only fir tge joint self.joint_current_control
        s_recon = np.zeros(len(data['ds']))
        ds_recon = np.zeros(len(data['ds']))
        s_recon_from_acc = np.zeros(len(data['ds']))
        s_recon[0] = data['s'][0, idx]
        s_recon_from_acc[0] = data['s'][0, idx]
        ds_recon[0] = data['ds'][0]
        for i in range(1, len(s_recon)):
            s_recon[i] = s_recon[i-1] + data["ds"][i-1] * self.dT
            ds_recon[i] = ds_recon[i-1] + data["dds"][i-1] * self.dT
        # Define the transition function
        def f_trans(x, u):
            # State vector: [position, velocity, acceleration]
            # Dynamics model: x_k = A * x_{k-1} + w_k
            A = np.array([[1, self.dT, 0.5 * self.dT**2], [0, 1, self.dT], [0, 0, 1]])
            return np.dot(A, x)

        # Define the measurement function (identity function as we directly observe position)
        def f_meas(x):
            return x[0]

        # Filter motor currents with a low pass filter
        print('Filtering currents')

        order = 5
        fs = 1/self.dT
        fc = [
            fs*0.1, # torso_pitch
            fs*0.1, # torso_roll
            fs*0.1, # torso_yaw
            fs*0.3, # l_hip_pitch
            fs*0.3, # l_hip_roll
            fs*0.22, # l_hip_yaw
            fs*0.3, # l_knee
            fs*0.3, # l_ankle_pitch
            fs*0.1, # l_ankle_roll
            fs*0.3, # r_hip_pitch
            fs*0.3, # r_hip_roll
            fs*0.22, # r_hip_yaw
            fs*0.3, # r_knee
            fs*0.1, # r_ankle_pitch
            fs*0.1  # r_ankle_roll
        ]

        # Find index of self.joint_current_control in data['joint_list']
        idx = data['joint_list'].index(self.joint_current_control[0])
        
        data['im_filtered'] = np.zeros_like(data["im"])

        normalized_cutoff_freq = 2 * fc[idx] / fs
        sos = butter(order, normalized_cutoff_freq, btype='low', output='sos')

        data["im"][:2] = data["im_desired"][0]
        data["im"][-3:] = data["im_desired"][-1]

        data['im_filtered'] = sosfiltfilt(sos, data["im"], axis=0)

        # Plot im and im_filtered
        plt.figure()
        plt.plot(data["im"], label='im')
        plt.plot(data["im_filtered"], label='im_filtered')
        plt.legend()
        # Add title and labels
        plt.title('Motor currents')
        plt.xlabel('Sample')
        plt.ylabel('Current [A]')
        plt.show()

        # data["im_filtered_matlab"] = data_logged["motors_state"]["currents"]["filtered_data"]

        # Filter motor currents with a low pass filter
        print('Filtering accelerometers')
        order = 2
        fs = 1/self.dT
        fc = 0.01 * fs
        
        normalized_cutoff_freq = 2 * fc / fs
        sos = butter(order, normalized_cutoff_freq, btype='low', output='sos')
        data['acc_filtered'] = {}
        for acc in data_logged["accelerometers"].keys():
            data['acc_filtered'][acc] = np.zeros_like(data_logged["accelerometers"][acc]["data"])
            for i in range(data_logged["accelerometers"][acc]["data"].shape[1]):
                data['acc_filtered'][acc][:, i] = sosfiltfilt(sos, data_logged["accelerometers"][acc]["data"][:, i], axis=0)   
            data['acc_filtered'][acc] = data['acc_filtered'][acc]

        print('Filtering gyros')
        fc = 0.02 * fs
        
        normalized_cutoff_freq = 2 * fc / fs
        sos = butter(order, normalized_cutoff_freq, btype='low', output='sos')
        data['gyros_filtered'] = {}
        for gyro in data_logged["gyros"].keys():
            data['gyros_filtered'][gyro] = np.zeros_like(data_logged["gyros"][gyro]["data"])
            for i in range(data_logged["gyros"][gyro]["data"].shape[1]):
                data['gyros_filtered'][gyro][:, i] = sosfiltfilt(sos, data_logged["gyros"][gyro]["data"][:, i], axis=0)   
            data['gyros_filtered'][gyro] = data['gyros_filtered'][gyro]                
        
        return data

    def _generate_additional_data(self, data):
        # Compute joint torques
        gravity_acceleration = 9.80665

        # def _process_sample(sample_idx):
        considered_joints_idyn = idyn.StringVector()
        for joint in data["joint_list"]:
            considered_joints_idyn.push_back(joint)

        # Load model
        model_loader = idyn.ModelLoader()
        ok = model_loader.loadReducedModelFromFile(
            self.model_path, considered_joints_idyn
        )
        if not ok:
            msg = "Unable to load the model from the file: " + self.model_path + "."
            raise ValueError(msg)
        
        # Define gravity vector
        world_gravity = idyn.Vector3()
        world_gravity.zero()
        world_gravity.setVal(2, -gravity_acceleration)

        # Define kindyn object
        kindyn = idyn.KinDynComputations()
        ok = kindyn.loadRobotModel(model_loader.model())
        if not ok:
            raise ValueError("Failed loading the model")

        # Create estimator class
        estimator = idyn.ExtWrenchesAndJointTorquesEstimator()

        # Load model and sensors from urdf
        estimator.setModelAndSensors(model_loader.model(), model_loader.sensors())

        # Set kinematics information
        if self.dataset_type == "walking":
            if data["foot_in_contact"] == "right":
                base_index = estimator.model().getFrameIndex("r_sole")
                contact_index = estimator.model().getFrameIndex("r_sole")
            else:
                base_index = estimator.model().getFrameIndex("l_sole")
                contact_index = estimator.model().getFrameIndex("l_sole")
        else:
            base_index = estimator.model().getFrameIndex(self.base_link)
            contact_index = estimator.model().getFrameIndex(self.contact_link)

        n_joints = kindyn.getNrOfDegreesOfFreedom()

        s_idyn = idyn.JointPosDoubleArray(n_joints)
        ds_idyn = idyn.JointDOFsDoubleArray(n_joints)
        dds_idyn = idyn.JointDOFsDoubleArray(n_joints)

        s_idyn.zero()
        ds_idyn.zero()
        dds_idyn.zero()

        if self.set_base_from_imu:
            ok = kindyn.setFrameVelocityRepresentation(idyn.BODY_FIXED_REPRESENTATION)
            assert ok, "Impossible to set frame velocity representation"
            ok = kindyn.setFloatingBase(self.base_link)
            assert ok, "Impossible to set floating base"
            H = kindyn.getRelativeTransform(self.base_link, self.imu_as_base)
            BASE_R_IMU = H.getRotation().toNumPy()
            BASE_o_IMU = H.getPosition().toNumPy()

        tauj = []

        # Find joint index
        # Specify unknown wrenches
        unknownWrench = idyn.UnknownWrenchContact()
        unknownWrench.unknownType = idyn.FULL_WRENCH
        unknownWrench.contactPoint.zero()
        fullBodyUnknowns = idyn.LinkUnknownWrenchContacts(estimator.model())
        fullBodyUnknowns.clear()
        fullBodyUnknowns.addNewContactInFrame(
            estimator.model(), contact_index, unknownWrench
        )

        # There are three output of the estimation
        estFTmeasurements = idyn.SensorsMeasurements(estimator.sensors())
        estJointTorques = idyn.JointDOFsDoubleArray(n_joints)
        estContactForces = idyn.LinkContactWrenches(estimator.model())

        idx = data['joint_list'].index(self.joint_current_control[0])

        for sample_idx in track(range(len(data["s"])), description="Processing...", total=len(data["s"])):
            s = np.full(n_joints, data["s"][sample_idx])
            ds = np.zeros(n_joints)
            dds = np.zeros(n_joints)

            ds[idx] = data["ds"][sample_idx]
            dds[idx] = data["dds"][sample_idx]

            if self.set_base_from_imu:
                BASE_omega_BASE = BASE_R_IMU @ data["gyros_filtered"][self.imu_as_base][sample_idx]
                BASE_dv_BASE = np.zeros(6)
                first = BASE_R_IMU @ data["acc_filtered"][self.imu_as_base][sample_idx]
                second = np.cross(BASE_omega_BASE, np.cross(BASE_omega_BASE, BASE_o_IMU))
                BASE_dv_BASE[:3] = first - second
                ok = estimator.updateKinematicsFromFloatingBase(s,
                                                                ds,
                                                                dds,
                                                                base_index,
                                                                BASE_dv_BASE[:3],
                                                                BASE_omega_BASE,
                                                                BASE_dv_BASE[3:])
                assert ok, "Impossible to update the estimator state."

            else:
                estimator.updateKinematicsFromFixedBase(
                    s, ds, dds, base_index, world_gravity
                )


            estimator.computeExpectedFTSensorsMeasurements(
                fullBodyUnknowns, estFTmeasurements, estContactForces, estJointTorques
            )

            tauj.append(estJointTorques.toNumPy())

        tauj_temp = np.array(tauj)


        data["s"] = data["s"][:, idx]
        data["tauj"] = tauj_temp[:, idx]

        del data['acc_filtered']
        del data['gyros_filtered']
        del data['orientations']
        del data['joint_list']
        del data['acc']
        del data['gyro']

        return data
