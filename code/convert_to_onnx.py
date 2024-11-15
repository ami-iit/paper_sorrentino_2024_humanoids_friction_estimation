import torch
import torch.nn as nn
from pathlib import Path
import argparse
import numpy as np
import pickle
from StandardScaler import StandardScaler
from NeuralNetwork import NeuralNetwork

def convert_model(model_path: Path, onnx_model_path: Path, normalization_folder: Path, opset_version: int):
    device = torch.device("cpu")
    torch.cuda.set_device(0)
    print(f"Device: {device}")
    
    # Restore the model with the trained weights
    model_configuration = torch.load(model_path)
    output_size = 1
    model = NeuralNetwork(model_configuration["nn_input_size"],
                          [
                              model_configuration["hidden_size0"],
                              model_configuration["hidden_size1"],
                           ], 
                           output_size,
                           model_configuration["dropout_rate"]).to(
        device
    )

    model.load_state_dict(model_configuration["trained_pinn_model"])

    model.eval()

    # Load normalization data
    with open(normalization_folder / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Convert from tensor to numpy
    X_mean = scaler.mean.to("cpu").numpy().flatten()
    X_std = scaler.std.to("cpu").numpy().flatten()
    X_std[np.where(X_std <= 1e-4)] = 1

    # Create normalization layer
    lin_normalization = nn.Linear(model_configuration["nn_input_size"], model_configuration["nn_input_size"])
    with torch.no_grad():
        lin_normalization.weight.copy_(torch.tensor(np.diag(np.reciprocal(X_std))))
        lin_normalization.bias.copy_(torch.tensor(-X_mean / X_std))

    # Create extended model
    extended_model = nn.Sequential(lin_normalization, model)

    # Input to the model
    batch_size = 1
    x = torch.randn(batch_size, model_configuration["nn_input_size"], requires_grad=True)

    # Export the model
    torch.onnx.export(extended_model,
                      x,
                      str(onnx_model_path),
                      export_params=True,
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

def main():
    parser = argparse.ArgumentParser(description='Convert mann-pytorch model into a onnx model.')
    parser.add_argument('--joint', '-j', type=str, required=True,
                         help='Joint name (e.g., r_ankle_roll).')
    parser.add_argument('--model', '-m', type=str, required=True,
                         help='Model index (e.g., 1).')
    parser.add_argument('--identifier', '-id', type=str, required=True,
                         help='Identifier for the specific model version (e.g., b4998_h9).')
    parser.add_argument('--onnx_mdoel_prefix', '-p', type=str, required=True,
                         help='Prefix for the onnx model name (e.g., 2_rk).')
    parser.add_argument('--onnx_opset_version', type=int, default=12, required=False,
                         help='The ONNX version to export the model to. At least 12 is required.')
    args = parser.parse_args()

    # Construct the file paths based on provided arguments
    base_path = Path("/home/iit.local/isorrentino/dev/pinn_friction_results")
    model_folder = base_path / args.joint / args.model / args.identifier
    model_path = model_folder / "model_e349.pt"
    normalization_path = model_folder / "scaler.pkl"
    
    # Automatically construct the ONNX filename using the identifier
    onnx_model_name = f"{args.onnx_mdoel_prefix}_{args.identifier}.onnx"  # Example: 2_rar_b4998_h9.onnx
    onnx_model_path = model_folder / onnx_model_name

    # Call the function to convert and export the model
    convert_model(model_path=model_path,
                  onnx_model_path=onnx_model_path,
                  normalization_folder=model_folder,
                  opset_version=args.onnx_opset_version)

if __name__ == "__main__":
    main()
