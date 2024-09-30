<h1 align="center">
  Physics-Informed Learning for the Friction Modeling
  of High-Ratio Harmonic Drives

</h1>

<div align="center">
<b>Ines Sorrentino</b>, Giulio Romualdi, Fabio Bergonti, 
Giuseppe L'Erario, Silvio Traversaro, Daniele Pucci <br> <br>
</div>

<div align="center">
    📅 Submitted to the 2024 IEEE-RAS International Conference on Humanoid Robots (Humanoids) 🤖
</div>

## Installation
This repository requires to install [idyntree](https://github.com/robotology/idyntree) library and MATLAB.

Use the `requirements.txt` file to recreate the environment:

```
conda create --name <new_environment_name> --file requirements.txt
```

## Repo usage

The application for acquiring data for friction identification can be found in https://github.com/LoreMoretti/bipedal-locomotion-framework/tree/add/MotorCurrentTrackingApplication/utilities/motor-current-tracking. You can follow instruction in the [repo](https://github.com/LoreMoretti/bipedal-locomotion-framework/tree/add/MotorCurrentTrackingApplication) to install and use it.

Datasets used for this paper for the training can be found at https://huggingface.co/datasets/ami-iit/sensorless-torque-control/tree/main.

After taking data, the first step is data post-processing. Run the bash script `postprocess_data.sh`. Example usage for parsing data for the `r_ankle_pitch` joint.

```
bash postprocess_data.sh -f '/home/isorrentino/dev/dataset/friction/r_ankle_pitch/sinusoid' -j 'r_ankle_pitch' -a 'torso_pitch torso_roll torso_yaw l_hip_pitch l_hip_roll l_hip_yaw l_knee l_ankle_pitch l_ankle_roll r_hip_pitch r_hip_roll r_hip_yaw r_knee r_ankle_pitch r_ankle_roll'
```

Find the `Stribeck-Coulomb-Viscous` model for the physics information used by the PINN. Change the joint to model in the script `simple_friction_modeling.py`.

```
python simple_friction_modeling.py
```

Before running the PINN training you need to specify the configuration file for the join to model. The `config` folder contains an example for the `r_ankle_roll` joint. After creating the configuration file you can run the training by means of `wight&biases` tool:

```
python feedforwardNN_wandb.py --joint_name "r_ankle_roll"
```

The trained networks are saved in the results forlder and can be converted in a `onnx` model by using the script `convert_to_onnx.py`.

The `onnx` model is loaded by the device [`JointTorqueControlDevice`](https://github.com/ami-iit/bipedal-locomotion-framework/tree/master/devices/JointTorqueControlDevice) running on the robot torso computer.

## Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://user-images.githubusercontent.com/43743081/89022636-a17e9e00-d322-11ea-9abd-92cda85d3705.jpeg" width="40">](https://github.com/isorrentino) | [@inessorrentino](https://github.com/isorrentino) |
