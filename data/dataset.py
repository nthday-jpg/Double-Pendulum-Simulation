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
                   num_workers=None, shuffle=True):
    """
    Create separate dataloaders for data, collocation, validation, and test sets.
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
        collocation_loader: DataLoader for collocation points
        val_loader: DataLoader for validation (early time)
        test_loader: DataLoader for test (late time - extrapolation)
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
        if os.name == 'nt':
            num_workers = 0
    
    batch_size = config.batch_size
    batch_size_collocation = config.batch_size_collocation or batch_size

    data_dataset = PendulumDataset(data_dir)

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
    
    # Get all initial states for collocation (from all trajectories)
    initial_states_np = [traj['initial_state'] for traj in data_dataset.trajectories]
    collocation_dataset = CollocationDataset(config.t_min, config.t_max, 
                                            config.n_collocation, initial_states_np)
    
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
    
    collocation_loader = DataLoader(
        collocation_dataset,
        batch_size=batch_size_collocation,
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"DataLoaders: data_bs={batch_size}, colloc_bs={batch_size_collocation}, workers={num_workers}")
    print(f"Dataset splits: train={train_size}, val={val_size}, test={len(test_indices)} (late time)")
    print(f"Temporal split: train/val use first {int((1-config.test_split)*100)}% of time, test uses last {int(config.test_split*100)}%")
    
    return train_loader, collocation_loader, val_loader, test_loader
