import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import matplotlib.pyplot as plt

velocity_threshold = 0.05
force_threshold = 35.0
acceleration_threshold = 100.0
# gear_ratio = 160.0
# ktau = 0.066468037
gear_ratio = 100.0
ktau = 0.156977705

import torch
import torch.nn as nn

class ContinuousFrictionModel(nn.Module):
    def __init__(self):
        super(ContinuousFrictionModel, self).__init__()
        # Inizializza i parametri con valori di partenza ragionevoli
        self.F_c = nn.Parameter(torch.tensor(1.0))  # Attrito di Coulomb
        self.F_s = nn.Parameter(torch.tensor(1.6))  # Attrito statico
        self.F_v = nn.Parameter(torch.tensor(1.0))  # Coefficiente di attrito viscoso
        self.v_s = nn.Parameter(torch.tensor(0.00001))  # Velocità di Stribeck, controlla l'effetto Stribeck

    def forward(self, v):
        """Passaggio forward del modello"""
        # Assicurati che i parametri siano positivi
        Fc = torch.nn.functional.softplus(self.F_c)
        Fs = torch.nn.functional.softplus(self.F_s)
        Fv = torch.nn.functional.softplus(self.F_v)
        Vs = torch.nn.functional.softplus(self.v_s)
        alpha = 1e-7
        
        # Calcola l'effetto Stribeck
        stribeck_term = (Fs - Fc) * torch.exp(-(v / Vs) ** 2)
        
        # Modifica per usare la tangente iperbolica per una transizione continua
        tanh_term = torch.tanh(v / alpha)
        
        # Calcolo della forza di attrito
        F_f = (Fc + stribeck_term) * tanh_term + Fv * v
        
        return F_f


class ComprehensiveFrictionModel(nn.Module):
    def __init__(self):
        super(ComprehensiveFrictionModel, self).__init__()
        # Initialize the parameters with reasonable starting values
        self.F_c = nn.Parameter(torch.tensor(1.0))  # Coulomb friction
        self.F_s = nn.Parameter(torch.tensor(1.5))  # Static friction
        self.F_v = nn.Parameter(torch.tensor(0.1))  # Viscous friction coefficient
        self.v_s = nn.Parameter(torch.tensor(0.5))  # Stribeck velocity, controls the Stribeck effect

    def forward(self, v):
        """ Forward pass for the model """
        # Ensure that the parameters are positive
        Fc = torch.nn.functional.softplus(self.F_c)
        Fs = torch.nn.functional.softplus(self.F_s)
        Fv = torch.nn.functional.softplus(self.F_v)
        Vs = torch.nn.functional.softplus(self.v_s)
        
        # Calculate the Stribeck effect
        stribeck_term = (Fs - Fc) * torch.exp(-(v / Vs) ** 2)
        
        # Friction force calculation
        F_f = (Fc + stribeck_term) * torch.sign(v) + Fv * v
        return F_f

# Create synthetic data
def load_data():
    folder_datasets = '/home/isorrentino/dev/dataset/friction/r_ankle_pitch/parsed'
    import pickle
    import os
    # find all the pickle files in the folder
    pickle_files = [f for f in os.listdir(folder_datasets) if f.endswith('.pickle')]
    print(pickle_files)

    # load the data
    X = []
    Y = []
    Z = []

    for dataset in pickle_files:
        with open(os.path.join(folder_datasets, dataset), 'rb') as file:
            f = pickle.load(file)
            v_data = torch.tensor(f['ds'][:], dtype=torch.float32)
            a_data = torch.tensor(f['dds'][:], dtype=torch.float32)
            # compute the friction force
            tau_j = torch.tensor(f['tauj'][:], dtype=torch.float32)
            im_filtered = torch.tensor(f['im_filtered'][:], dtype=torch.float32)

            F_f_data = -(tau_j - ktau * gear_ratio * im_filtered)

            X.append(v_data)
            Y.append(F_f_data)
            Z.append(a_data)

    v_data = torch.cat(X)
    F_f_data = torch.cat(Y)
    a_data = torch.cat(Z)

    # normalize the data
    max_v = torch.max(torch.abs(v_data))
    max_F_f = torch.max(torch.abs(F_f_data))

    v_data = v_data / max_v
    F_f_data = F_f_data / max_F_f

    return v_data, F_f_data, max_v, max_F_f, a_data

def train_model(model, v_data, F_f_data, max_v, max_F_f, a_data, epochs=1000, lr=0.01, velocity_threshold=velocity_threshold, force_threshold=force_threshold, acceleration_threshold=acceleration_threshold):
    # Create a mask to filter out data where both velocity and force are below their thresholds
    combined_mask = (torch.abs(v_data * max_v) > velocity_threshold) | (torch.abs(F_f_data * max_F_f) > force_threshold)

    # The mask must contains poiints only on the first and third quadrant
    combined_mask = combined_mask & (v_data * F_f_data > 0) & (torch.abs(a_data) < acceleration_threshold)
    
    # Apply the mask to filter the data
    filtered_v_data = v_data[combined_mask]
    filtered_F_f_data = F_f_data[combined_mask]

    
    # Calculate weights (you may want to tune this)
    # weights = (torch.abs(filtered_v_data) + 0.01)  # Higher weights for non-zero velocities
    
    criterion = nn.L1Loss(reduction='none')  # No reduction to apply weights
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        F_f_pred = model(filtered_v_data)
        
        loss = criterion(F_f_pred, filtered_F_f_data)
        weighted_loss = (loss).mean()
        
        weighted_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {weighted_loss.item()}')
    
    return model


def test_model(model, v_data, F_f_data, max_v, max_F_f, a_data):
    # Compute the friction force using the trained model
    F_f_pred = model(v_data)
    
    # Define the thresholds for the velocity and force
    combined_mask = (torch.abs(a_data) < acceleration_threshold)
    combined_mask = combined_mask & (v_data * F_f_data > 0)

    filtered_v_data = v_data[combined_mask]
    filtered_F_f_data = F_f_data[combined_mask]

    # plot the results
    plt.figure()
    plt.scatter(v_data.detach().numpy() * max_v.detach().numpy(), F_f_data.detach().numpy() * max_F_f.detach().numpy(), label='True', alpha=0.5, s=1)
    plt.scatter(v_data.detach().numpy() * max_v.detach().numpy(), F_f_pred.detach().numpy() * max_F_f.detach().numpy(), label='Predicted', alpha=0.5, s=1)
    plt.show()

    plt.figure()
    plt.scatter(v_data[combined_mask].detach().numpy() * max_v.detach().numpy(), F_f_data[combined_mask].detach().numpy() * max_F_f.detach().numpy(), label='True', alpha=0.5, s=1)
    plt.scatter(v_data[combined_mask].detach().numpy() * max_v.detach().numpy(), F_f_pred[combined_mask].detach().numpy() * max_F_f.detach().numpy(), label='Predicted', alpha=0.5, s=1)
    plt.show()

# Main function to train and save the model
def main():
    # Initialize the model
    model = ContinuousFrictionModel()
    
    # Create synthetic data
    v_data, F_f_data, max_v, max_F_f, a_data = load_data()
    
    # Train the model
    trained_model = train_model(model, v_data, F_f_data, max_v, max_F_f, a_data, epochs=3000, lr=0.01)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'enhanced_friction_model.pth')
    print("Model saved as 'enhanced_friction_model.pth'")

    print("Print the model parameters")
    print("Fc", torch.nn.functional.softplus(trained_model.F_c).item())
    print("Fs", torch.nn.functional.softplus(trained_model.F_s).item())
    print("Fv", torch.nn.functional.softplus(trained_model.F_v).item())
    print("Vs", torch.nn.functional.softplus(trained_model.v_s).item())

    test_model(trained_model, v_data, F_f_data, max_v, max_F_f, a_data)


# Run the main function
if __name__ == "__main__":
    main()
