import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader, Subset
from utils.config import Config

class PendulumDataset(Dataset):
    """
    Dataset for multiple double-pendulum trajectories.
    Each trajectory has its own physics parameters stored in parameters_XXX.json.

    Returns:
        t      : (1,)
        initial_state : (4,) = [theta1_0, theta2_0, omega1_0, omega2_0]
        state  : (2,) = [theta1, theta2]
        point_type : 0 (data point)
    """
    def __init__(self, data_dir, normalize_time=True):
        """
        Args:
            data_dir: Directory containing trajectory files (trajectory_000.npz, trajectory_001.npz, ...)
                     and corresponding parameter files (parameters_000.json, parameters_001.json, ...)
        """
        self.normalize_time = normalize_time

        # Load all trajectory files
        self.trajectories = []
        self.trajectory_lengths = []
        self.cumulative_lengths = [0]
        self.parameters_list = []
        
        # Find all trajectory files in the directory
        trajectory_files = sorted([f for f in os.listdir(data_dir) if f.startswith('trajectory_') and f.endswith('.npz')])
        
        if not trajectory_files:
            raise ValueError(f"No trajectory files found in {data_dir}")
        
        for traj_file in trajectory_files:
            # Extract trajectory index from filename
            traj_idx = int(traj_file.split('_')[1].split('.')[0])
            
            # Load trajectory data
            data = np.load(os.path.join(data_dir, traj_file))
            trajectory = {
                't': data["t"],              # (N,)
                'initial_state': data["initial_state"],  # (4,)
                'q': data["q"]               # (N, 2)
            }
            self.trajectories.append(trajectory)
            self.trajectory_lengths.append(len(data["t"]))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(data["t"]))
            
            # Load individual parameter file for this trajectory
            params_file = os.path.join(data_dir, f"parameters_{traj_idx:03d}.json")
            if os.path.exists(params_file):
                with open(params_file, "r") as f:
                    self.parameters_list.append(json.load(f))
            else:
                raise FileNotFoundError(f"Parameter file not found: {params_file}")
        
        self.total_length = self.cumulative_lengths[-1]

        if self.normalize_time:
            # Assume all trajectories share the same time scale for normalization
            # Nondimensionalize time using characteristic time scale T = sqrt((l1 + l2) / (2g))
            char_parameter = self.parameters_list[0]
            self.time_scale = np.sqrt((char_parameter['l1'] + char_parameter['l2']) / (2*char_parameter['g']))
        
        # Check if all parameters are the same
        all_same = all(p == self.parameters_list[0] for p in self.parameters_list)
        params_msg = "with shared parameters" if all_same else "with different parameters"
        print(f"📊 Loaded {len(self.trajectories)} trajectories {params_msg}, {self.total_length} total data points")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which trajectory this index belongs to
        traj_idx = 0
        for i, cumlen in enumerate(self.cumulative_lengths[1:]):
            if idx < cumlen:
                traj_idx = i
                break
        
        # Get local index within the trajectory
        local_idx = idx - self.cumulative_lengths[traj_idx]
        
        # Get data from the appropriate trajectory
        traj = self.trajectories[traj_idx]
        t_raw = torch.tensor([traj['t'][local_idx]], dtype=torch.float32)
        if self.normalize_time:
            t = t_raw / self.time_scale
        else:
            t = t_raw
        initial_state = torch.tensor(traj['initial_state'], dtype=torch.float32)
        state = torch.tensor(np.concatenate([traj['q'][local_idx]]), dtype=torch.float32)
        return t, initial_state, state, 0  # point_type = 0 for data


def get_dataloader(data_dir, config,
                   num_workers=None, shuffle=True):
    """
    Create dataloaders for training, validation, and test sets.
    Test set contains the LAST time points from each trajectory for temporal extrapolation testing.
    
    Args:
        data_dir: Directory containing trajectory files
        config: Training configuration
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        val_split: Validation split ratio (from train set)
        test_split: Fraction of time points to reserve for test (from end of each trajectory)
    
    Returns:
        train_loader: DataLoader for training data points (early time)
        val_loader: DataLoader for validation (early time)
        test_loader: DataLoader for test (late time - extrapolation)
        parameters_list: List of parameter dicts for each trajectory
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
        if os.name == 'nt':
            num_workers = 0
    
    batch_size = config.batch_size

    data_dataset = PendulumDataset(data_dir, normalize_time=config.normalize_time)

    train_val_indices = []
    test_indices = []
    
    for traj_idx in range(len(data_dataset.trajectories)):
        start_idx = data_dataset.cumulative_lengths[traj_idx]
        end_idx = data_dataset.cumulative_lengths[traj_idx + 1]
        traj_length = end_idx - start_idx

        train_val_length = int(traj_length * (1 - config.test_split))
        test_start = start_idx + train_val_length

        train_val_indices.extend(range(start_idx, test_start))
        test_indices.extend(range(test_start, end_idx))

    test_dataset = Subset(data_dataset, test_indices)
    
    # Split train/val from early time points
    train_val_size = len(train_val_indices)
    val_size = int(train_val_size * config.val_split)
    train_size = train_val_size - val_size
    
    # Use generator with seed for reproducible random_split
    generator = torch.Generator().manual_seed(config.seed)
    train_val_dataset = Subset(data_dataset, train_val_indices)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size], generator=generator
    )
        
    # Store actual dataset time range in config for checkpoint saving
    if config.normalize_time:
        config.time_scale = data_dataset.time_scale
    
    # Import seed_worker for DataLoader workers
    from utils.seed import seed_worker
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    parameters_list = data_dataset.parameters_list

    print(f"DataLoaders: batch_size={batch_size}, workers={num_workers}")
    print(f"Dataset splits: train={train_size}, val={val_size}, test={len(test_indices)} (late time)")
    print(f"Temporal split: train/val use first {int((1-config.test_split)*100)}% of time, test uses last {int(config.test_split*100)}%")
    
    return train_loader, val_loader, test_loader, parameters_list
