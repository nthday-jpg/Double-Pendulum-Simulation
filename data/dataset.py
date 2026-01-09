from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import json
import os
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
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing trajectory files (trajectory_000.npz, trajectory_001.npz, ...)
                     and corresponding parameter files (parameters_000.json, parameters_001.json, ...)
        """
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
        t = torch.tensor([traj['t'][local_idx]], dtype=torch.float32)
        initial_state = torch.tensor(traj['initial_state'], dtype=torch.float32)
        state = torch.tensor(np.concatenate([traj['q'][local_idx]]), dtype=torch.float32)
        return t, initial_state, state, 0  # point_type = 0 for data


class CollocationDataset(Dataset):
    """
    Collocation points for physics loss.
    Samples from all trajectories' initial states.
    
    Returns:
        t      : (1,)
        initial_state : (4,) sampled from available initial states
        state  : (2,) dummy zeros
        point_type : 1 (collocation point)
    """
    def __init__(self, tmin, tmax, num_points, initial_states=None):
        """
        Args:
            tmin: Minimum time
            tmax: Maximum time
            num_points: Number of collocation points
            initial_states: List of initial states (4,) from all trajectories, or None for zeros
        """
        self.t = np.random.uniform(tmin, tmax, size=(num_points, 1))
        self.initial_states = initial_states if initial_states is not None else [np.zeros(4)]
        # Randomly assign an initial state to each collocation point
        self.assigned_initial_states = np.array([
            self.initial_states[np.random.randint(len(self.initial_states))]
            for _ in range(num_points)
        ])

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        t = torch.tensor(self.t[idx], dtype=torch.float32)
        initial_state = torch.tensor(self.assigned_initial_states[idx], dtype=torch.float32)
        dummy_state = torch.zeros(2)
        return t, initial_state, dummy_state, 1  # point_type = 1 for collocation


def get_dataloader(data_dir, config,
                   num_workers=None, shuffle=True, val_split=0.2):
    """
    Create separate dataloaders for data and collocation with different batch sizes.
    
    Args:
        data_dir: Directory containing trajectory files (trajectory_000.npz, trajectory_001.npz, ...)
                 and parameter files (parameters_000.json, parameters_001.json, ...)
        config: Training configuration
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        val_split: Validation split ratio
    
    Returns:
        data_loader: DataLoader for data points (batch_size from config.batch_size)
        collocation_loader: DataLoader for collocation points (batch_size from config.batch_size_collocation)
        val_loader: DataLoader for validation
    
    Usage:
        config.batch_size = 32  # small for data
        config.batch_size_collocation = 256  # large for collocation
        data_loader, colloc_loader, val_loader = get_dataloader(data_dir, config)
        trainer = Trainer(model, config, data_loader, colloc_loader, val_loader, optimizer)
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
        if os.name == 'nt':
            num_workers = 0
    
    batch_size = config.batch_size
    batch_size_collocation = config.batch_size_collocation or batch_size

    data_dataset = PendulumDataset(data_dir)
    
    # Get all initial states from data_dataset to use in collocation
    initial_states_np = [traj['initial_state'] for traj in data_dataset.trajectories]
    collocation_dataset = CollocationDataset(config.t_min, config.t_max, config.n_collocation, initial_states_np)

    data_size = len(data_dataset)
    val_size = int(data_size * val_split)
    train_size = data_size - val_size
    
    # Use generator with seed for reproducible random_split
    generator = torch.Generator().manual_seed(config.seed)
    train_data_dataset, val_data_dataset = torch.utils.data.random_split(
        data_dataset, [train_size, val_size], generator=generator
    )
    
    # Import seed_worker for DataLoader workers
    from utils.seed import seed_worker
    
    val_loader = DataLoader(val_data_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(),
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(config.seed))

    data_loader = DataLoader(
        train_data_dataset, 
        batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed)
    )
    collocation_loader = DataLoader(
        collocation_dataset,
        batch_size=batch_size_collocation,
        shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"DataLoaders: data_bs={batch_size}, colloc_bs={batch_size_collocation}, workers={num_workers}")
    print(f"Dataset splits: train={train_size}, val={val_size}")
    return data_loader, collocation_loader, val_loader
