from pathlib import Path
import pickle
import toml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils
import wandb
import matplotlib.pyplot as plt
import yaml
from utils_functions import *
from NeuralNetwork import NeuralNetwork
from FrictionTorqueDataset import FrictionTorqueDataset
from StandardScaler import StandardScaler
import argparse


input_model_type = 5

def load_config(file_path):
    try:
        with open(file_path, "r") as f:
            config = toml.load(f)
        return config
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def evaluate(test_data_loader, model, loss_fcn1, loss_fcn2, run, device):
    model.eval()

    total_loss = 0

    predictions_list = []
    output_list = []

    with torch.no_grad():
        for i, (input, output, physics) in enumerate(test_data_loader):
            predictions = model(input)
            loss1 = loss_fcn1(predictions, output)
            loss2 = loss_fcn2(predictions, physics)
            loss = (
                1 - run.config.lambda_loss_physics
            ) * loss1 + run.config.lambda_loss_physics * loss2
            total_loss += loss.item()

            # Collect predictions and outputs for validation dataset
            predictions_list.append(predictions.cpu().numpy())
            output_list.append(output.cpu().numpy())

    model.train()

    predictions_list = np.concatenate(predictions_list, axis=0)
    output_list = np.concatenate(output_list, axis=0)

    # Plot predictions vs output for validation dataset
    fig, ax = plt.subplots()
    sample_numbers = np.arange(len(output_list))
    ax.plot(sample_numbers, output_list, label='True Output', color='blue')
    ax.plot(sample_numbers, predictions_list, label='Predictions', color='orange')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Value')
    ax.set_title(f'Predictions vs Output (Validation)')
    ax.legend()
    wandb.log({f"Predictions vs Output (Validation)": wandb.Image(fig)})
    plt.close(fig)

    return total_loss / len(input)


def run_train(run):

    # Define device
    # device = torch.device("cpu")
    # # torch.cuda.set_device(0)
    # print(f"Device: {device}")

    # Define device
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    print(f"Device: {device}")

    # Load configuration
    filepath = run.config.toml_path
    config = load_config(filepath)

    # Load dataset
    data_concatenated = load_dataset(config["data"]["folders"])

    config_dataset_creation = {}
    config_dataset_creation["joint_to_model"] = config["joint"]["joint_to_model"]
    config_dataset_creation["ktau"] = config["joint"]["torque_constant"]
    config_dataset_creation["gear_ratio"] = config["joint"]["gear_ratio"]
    config_dataset_creation["joint_lim_inf"] = config["joint"]["joint_lim_inf"]
    config_dataset_creation["joint_lim_sup"] = config["joint"]["joint_lim_sup"]
    config_dataset_creation["joint_acc_limit"] = config["joint"]["joint_acc_limit"]
    config_dataset_creation["im_limit"] = config["joint"]["im_limit"]
    config_dataset_creation["tauf_limit"] = config["joint"]["tauf_limit"]
    config_dataset_creation["joint_vel_limit"] = config["joint"]["joint_vel_limit"]
    config_dataset_creation["history_size"] = run.config.num_past_samples
    config_dataset_creation["Fc"] = config["friction_params"]["Fc"]
    config_dataset_creation["Fs"] = config["friction_params"]["Fs"]
    config_dataset_creation["Fv"] = config["friction_params"]["Fv"]
    config_dataset_creation["Vs"] = config["friction_params"]["Vs"]

    # Create dataset for NN
    # [full_input, full_output, full_physics] = create_dataset(
    #     data_concatenated, config_dataset_creation
    # )

    # input_model_type = config["general"]["input_model"]

    [input, output, physics] = create_dataset(
        data_concatenated, config_dataset_creation, input_model_type
    )

    model_folder = Path(
            config["general"]["results_folder"]
            + "/"
            + str(input_model_type)
            + "/"
            + "b"
            + str(run.config.batch_size)
            + "_h"
            + str(run.config.num_past_samples)
            + "/"
    )
    model_folder.mkdir(parents=True, exist_ok=True)

    # [input, output, physics] = split_and_save_data(full_input, full_output, full_physics, split_percentage_validation=0.4, output_dir=model_folder)

    input_train, input_test, output_train, output_test, physics_train, physics_test = (
        train_test_split(input, output, physics, test_size=0.2, shuffle=True)
    )

    x_scaler = StandardScaler().fit(torch.from_numpy(input_train).to(device))
    x_scaler.set_device(device)

    # Create data loaders
    scale_data = True
    training_dataset = FrictionTorqueDataset(
        input_train, output_train, physics_train, x_scaler, scale_input=scale_data, device=device
    )
    training_dataloader = DataLoader(
        training_dataset, batch_size=run.config.batch_size, shuffle=False
    )

    test_dataset = FrictionTorqueDataset(
        input_test, output_test, physics_test, x_scaler, scale_input=scale_data, device=device
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=run.config.batch_size, shuffle=False
    )

    # create model
    print("Input shape: ", input_train.shape)
    output_size = 1
    model = NeuralNetwork(
        input_train.shape[1],
        [run.config.hidden_size0, run.config.hidden_size1],
        output_size,
        run.config.dropout_rate,
    ).to(device)

    # Loss and optimizer
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=run.config.learning_rate)

    for epoch in range(run.config.num_epochs):

        total_loss = 0

        predictions_train = []
        output_train_list = []

        for i, (input, output, physics) in enumerate(training_dataloader):
            # Forward pass
            predictions = model(input)

            loss1 = criterion1(predictions, output)
            loss2 = criterion2(predictions, physics)

            loss = (
                1 - run.config.lambda_loss_physics
            ) * loss1 + run.config.lambda_loss_physics * loss2

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Collect predictions and outputs for training dataset
            predictions_train.append(predictions.detach().cpu().numpy())
            output_train_list.append(output.detach().cpu().numpy())

        total_loss = total_loss / len(input)

        val_loss = evaluate(test_dataloader, model, criterion1, criterion2, run, device)

        print(
            f"Epoch [{epoch+1}/{run.config.num_epochs}], Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        # Flatten lists for plotting training dataset
        predictions_train = np.concatenate(predictions_train, axis=0)
        output_train_list = np.concatenate(output_train_list, axis=0)

        # Plot predictions vs output for training dataset
        fig, ax = plt.subplots()
        sample_numbers = np.arange(len(output_train_list))
        ax.plot(sample_numbers, output_train_list, label='True Output', color='blue')
        ax.plot(sample_numbers, predictions_train, label='Predictions', color='orange')
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Value')
        ax.set_title('Predictions vs Output (Training)')
        ax.legend()
        wandb.log({"Predictions vs Output (Training)": wandb.Image(fig)})
        plt.close(fig)

        log = dict()
        log["epoch"] = epoch
        log["train_loss"] = total_loss
        log["val_loss"] = val_loss
        log["scaler_input_mean"] = x_scaler.mean
        log["scaler_input_std"] = x_scaler.std
        log["scale_data"] = scale_data

        if val_loss > 2.0 or total_loss > 3.0:
            # Exit for loop if validation loss is too high
            break

        wandb.log(log)
        if (epoch + 1) % 50 == 0 and (epoch + 1) > 150:
            to_save = {}
            to_save["model"] = model.state_dict()
            to_save["scaler_input_mean"] = x_scaler.mean
            to_save["scaler_input_std"] = x_scaler.std
            to_save["hidden_size0"] = run.config.hidden_size0
            to_save["hidden_size1"] = run.config.hidden_size1
            to_save["dropout_rate"] = run.config.dropout_rate
            to_save["num_past_samples"] = run.config.num_past_samples
            to_save["scaler_input_mean"] = x_scaler.mean
            to_save["scaler_input_std"] = x_scaler.std
            to_save["nn_input_size"] = input_train.shape[1]
            to_save["scale_data"] = scale_data
            torch.save(to_save, str(model_folder / ("model_e" + str(epoch) + ".pt")))
            print(f"Model saved at epoch {epoch}")
            with open(str(model_folder) + "/scaler.pkl", "wb") as f:
                pickle.dump(x_scaler, f)

        

def main():
    run = wandb.init(project="test", job_type="train", tags=["v1", "Sweep"])
    run_train(run)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select joint for identification")
    parser.add_argument(
        "--joint_name",
        type=str,
        help="Select the joint for running the friction identification",
    )
    args = parser.parse_args()
    joint_name = args.joint_name

    print("Training launched for joint")
    print(joint_name)

    tune = True
    if tune:
        if joint_name == "r_ankle_roll":
            config_file_name = "code/python/PINN_friction/config/ergoCubSN001/r_ankle_roll/config_training_wandb.yaml"
        elif joint_name == "r_ankle_pitch":
            config_file_name = "code/python/PINN_friction/config/ergoCubSN001/r_ankle_pitch/config_training_wandb.yaml"
        with open(config_file_name, "r") as file:
            SWEEP_CONFIG = yaml.safe_load(file)
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=SWEEP_CONFIG["project_name"])
        wandb.agent(sweep_id, main, count=100)
    else:
        with open(
            "code/python/PINN_friction/config/ergoCubSN001/r_ankle_roll/r_ankle_roll.yaml",
            "r",
        ) as file:
            RUN_CONFIG = yaml.safe_load(file)
        print(RUN_CONFIG)
        run = wandb.init(
            project=RUN_CONFIG["project_name"],
            job_type="train",
            tags=[RUN_CONFIG["project_name"], "test"],
            config=RUN_CONFIG,
        )
        run_train(run)
        wandb.finish()
