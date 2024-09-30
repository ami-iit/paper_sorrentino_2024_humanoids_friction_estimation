import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
from typing import List


class FrictionTorqueDataset(Dataset):
    def __init__(
        self,
        input_np,
        output_np,
        physics_np,
        input_scaler,
        scale_input=False,
        training=True,
        device=torch.device("cpu")
    ):
        self.device = device

        self.training = training

        self.input_torch = torch.from_numpy(input_np.astype(np.float32)).to(self.device)
        self.output_torch = torch.from_numpy(output_np.astype(np.float32)).to(self.device)
        self.physics_torch = torch.from_numpy(physics_np.astype(np.float32)).to(self.device)
        
        if scale_input:
            self.input = input_scaler.transform(self.input_torch.to(self.device))
        else:
            self.input = self.input_torch
        

        self.n_samples = input_np.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.training:
            return self.input[idx], self.output_torch[idx], self.physics_torch[idx]
        else:
            return self.input[idx], self.output_torch[idx]
