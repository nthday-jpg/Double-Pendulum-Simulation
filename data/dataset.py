from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import json

class PendulumDataset(Dataset):
    """
    Dataset for a single double-pendulum trajectory.

    Returns:
        t      : (1,)
        state  : (2,) = [theta1, theta2]
    """
    def __init__(self, data_path, parameters_path):
        data = np.load(data_path)
        self.t = data["t"]          # (N,)
        self.q = data["q"]          # (N, 2)

        with open(parameters_path, "r") as f:
            self.parameters = json.load(f)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        t = torch.tensor([self.t[idx]], dtype=torch.float32)
        state = torch.tensor(
            np.concatenate([self.q[idx]]),
            dtype=torch.float32
        )
        return t, state

class CollocationDataset(Dataset):
    """
        Deterministic collocation points 
        Returns:
            t      : (1,)
    """
    def __init__(self, tmin, tmax, num_points):
        self.t = np.random.uniform(tmin, tmax, size=(num_points, 1))

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return torch.tensor(self.t[idx], dtype=torch.float32)

    
class MixedDataset(Dataset):
    """ 
        Dataset combining data points and collocation points.
        Returns:
            t      : (1,)
            state  : (2,) or None   
            point_type : 0 for data, 1 for collocation
    """

    def __init__(self, data_dataset, collocation_dataset, data_fraction):
        self.data_dataset = data_dataset
        self.data_size = len(data_dataset)
        
        self.collocation_dataset = collocation_dataset
        self.collocation_size = len(collocation_dataset)
        
        self.n_collocations_per_epoch = self.data_size * (1 - data_fraction) / data_fraction
        self.total_size = self.data_size + int(self.n_collocations_per_epoch)

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx < self.data_size:
            t, state = self.data_dataset[idx]
            return t, state, 0
        else:
            collocation_idx = (idx - self.data_size) % self.collocation_size
            t = self.collocation_dataset[collocation_idx]
            dummy_state = torch.zeros(2)  # Placeholder
            return t, dummy_state, 1
        
def get_dataloader(data_path, parameters_path, 
                   tmin, tmax, n_collocation,
                   batch_size=32, num_workers=1, shuffle=True,
                   val_split=0.2, data_fraction=0.1):
    """ 
        Data_fraction: Fraction of training samples per epoch that are data points. \\
        Ndata / (Ndata + Ncollocation) = data_fraction \\
        n_collocation: Number of collocation points to use 
    """
    data_dataset = PendulumDataset(data_path, parameters_path)
    collocation_dataset = CollocationDataset(tmin, tmax, n_collocation)

    data_size = len(data_dataset)
    val_size = int(data_size * val_split)
    train_size = data_size - val_size
    train_data_dataset, val_data_dataset = torch.utils.data.random_split(
        data_dataset, [train_size, val_size]
    )

    train_dataset = MixedDataset(train_data_dataset, collocation_dataset, data_fraction)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)
    
    val_loader = DataLoader(val_data_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)


    return train_loader, val_loader