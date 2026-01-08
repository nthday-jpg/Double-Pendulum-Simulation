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
        point_type : 0 (data point)
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
        return t, state, 0  # point_type = 0 for data


class CollocationDataset(Dataset):
    """
    Collocation points for physics loss.
    
    Returns:
        t      : (1,)
        state  : (2,) dummy zeros
        point_type : 1 (collocation point)
    """
    def __init__(self, tmin, tmax, num_points):
        self.t = np.random.uniform(tmin, tmax, size=(num_points, 1))

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        t = torch.tensor(self.t[idx], dtype=torch.float32)
        dummy_state = torch.zeros(2)
        return t, dummy_state, 1  # point_type = 1 for collocation


def get_dataloader(data_path, parameters_path, config: Config,
                   num_workers=None, shuffle=True, val_split=0.2):
    """
    Create separate dataloaders for data and collocation with different batch sizes.
    
    Returns:
        data_loader: DataLoader for data points (batch_size from config.batch_size)
        collocation_loader: DataLoader for collocation points (batch_size from config.batch_size_collocation)
        val_loader: DataLoader for validation
    
    Usage:
        config.batch_size = 32  # small for data
        config.batch_size_collocation = 256  # large for collocation
        data_loader, colloc_loader, val_loader = get_dataloader(...)
        trainer = Trainer(model, config, data_loader, colloc_loader, val_loader, optimizer)
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

    data_loader = DataLoader(
        train_data_dataset, 
        batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    collocation_loader = DataLoader(
        collocation_dataset,
        batch_size=batch_size_collocation,
        shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"DataLoaders: data_bs={batch_size}, colloc_bs={batch_size_collocation}, workers={num_workers}")
    return data_loader, collocation_loader, val_loader
