import numpy as np
import torch

# class StandardScaler(object):
#     """
#     Standardize data by removing the mean and scaling to unit variance.
#     """
#     def __init__(self):
#         self.mean = None
#         self.std = None

#     def fit(self, sample):
#         self.mean = sample.mean(0, keepdim=True)
#         self.std = sample.std(0, unbiased=False, keepdim=True)

#         return self
    
#     def set_device(self, device):
#         self.mean = self.mean.to(device)
#         self.std = self.std.to(device)

#     def __call__(self, sample):
#         return self.transform(sample)
    
#     def transform(self, sample):
#         return (sample - self.mean) / self.std

#     def inverse_transform(self, sample):
#         return sample * self.std + self.mean


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, sample):
        sample_cpu = sample.cpu().numpy() if torch.is_tensor(sample) else sample
        
        self.mean = np.mean(sample_cpu, axis=0, keepdims=True)
        self.std = np.percentile(sample_cpu, 75, axis=0, keepdims=True) - np.percentile(sample_cpu, 25, axis=0, keepdims=True)

        # Converti la meana e l'std in tensori PyTorch
        self.mean = torch.tensor(self.mean, dtype=torch.float32)
        self.std = torch.tensor(self.std, dtype=torch.float32)

        return self
    
    def transform(self, sample):
        return (sample - self.mean.to(sample.device)) / self.std.to(sample.device)

    def inverse_transform(self, sample):
        return sample * self.std.to(sample.device) + self.mean.to(sample.device)

    def set_device(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
