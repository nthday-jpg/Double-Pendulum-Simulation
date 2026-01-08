from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import json
import os
from utils.config import Config

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
        state = torch.tensor(np.concatenate([self.q[idx]]), dtype=torch.float32)
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

    
class DataPointDataset(Dataset):
    """ 
        Wrapper for data points with type annotation.
        Returns:
            t      : (1,)
            state  : (2,)   
            point_type : 0 for data
    """
    def __init__(self, data_dataset):
        self.data_dataset = data_dataset

    def __len__(self):
        return len(self.data_dataset)
    
    def __getitem__(self, idx):
        t, state = self.data_dataset[idx]
        return t, state, 0


class CollocationPointDataset(Dataset):
    """ 
        Wrapper for collocation points with type annotation.
        Returns:
            t      : (1,)
            state  : (2,) dummy zeros   
            point_type : 1 for collocation
    """
    def __init__(self, collocation_dataset):
        self.collocation_dataset = collocation_dataset

    def __len__(self):
        return len(self.collocation_dataset)
    
    def __getitem__(self, idx):
        t = self.collocation_dataset[idx]
        dummy_state = torch.zeros(2)
        return t, dummy_state, 1  # type = 1 for collocation


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

        
def get_dataloader(data_path, parameters_path, config: Config,
                   num_workers=None, shuffle=True, val_split=0.2,
                   separate_loaders=False):
    """
    Create dataloaders for PINN training.
    
    Args:
        separate_loaders: If True, returns (data_loader, collocation_loader, val_loader)
                        If False, returns (train_loader, val_loader)
    
    Usage:
        # Mixed mode (default):
        train_loader, val_loader = get_dataloader(..., separate_loaders=False)
        trainer = Trainer(model, config, train_loader, val_loader, optimizer)
        
        # Separate mode (for different batch sizes):
        cfg.batch_size_collocation = 256  # larger than data batch
        data_loader, colloc_loader, val_loader = get_dataloader(..., separate_loaders=True)
        trainer = Trainer(model, config, data_loader, val_loader, optimizer, collocation_loader=colloc_loader)
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
        if os.name == 'nt':
            num_workers = 0
    
    batch_size = config.batch_size
    batch_size_collocation = config.batch_size_collocation or batch_size

    data_dataset = PendulumDataset(data_path, parameters_path)
    collocation_dataset = CollocationDataset(config.t_min, config.t_max, config.n_collocation)

    data_size = len(data_dataset)
    val_size = int(data_size * val_split)
    train_size = data_size - val_size
    train_data_dataset, val_data_dataset = torch.utils.data.random_split(
        data_dataset, [train_size, val_size]
    )
    
    val_loader = DataLoader(val_data_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())

    if separate_loaders:
        data_loader = DataLoader(
            DataPointDataset(train_data_dataset), 
            batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        collocation_loader = DataLoader(
            CollocationPointDataset(collocation_dataset),
            batch_size=batch_size_collocation,
            shuffle=shuffle, num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"Separate loaders: data_bs={batch_size}, colloc_bs={batch_size_collocation}, workers={num_workers}")
        return data_loader, collocation_loader, val_loader
    else:
        train_dataset = MixedDataset(train_data_dataset, collocation_dataset, config.data_fraction)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=torch.cuda.is_available())
        print(f"Mixed loader: batch_size={batch_size}, workers={num_workers}")
        return train_loader, val_loader
