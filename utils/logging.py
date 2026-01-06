# logging.py
import os
import csv
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch
from utils.config import Config

def init_run(cfg):
    """
        Initialize the directory structure and logging utilities for a new experiment run.
        Args:
            cfg: Configuration object containing run parameters.
        Returns:    
            writer: CSV DictWriter for logging metrics.
            csv_file: Open CSV file handle.
            tb: TensorBoard SummaryWriter.
            run_dir: Path to the run directory.
    """
    
    run_dir = os.path.join("runs", cfg.run_name)
    os.makedirs(run_dir, exist_ok=False)

    # for saving model checkpoints
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    # for TensorBoard logs
    os.makedirs(os.path.join(run_dir, "tb"))

    # save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(cfg), f)

    # CSV setup - add physics parameters to fieldnames
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "rollout_mse",
        "energy_drift"
    ]
    
    # Add physics parameters if they exist in config
    if hasattr(cfg, 'm1'):
        fieldnames.extend(['m1', 'm2', 'l1', 'l2', 'g'])
    
    csv_file = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # TensorBoard
    tb = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    
    # Determine architecture representation
    if hasattr(cfg, 'hidden_dims') and cfg.hidden_dims is not None:
        arch_str = str(cfg.hidden_dims)
        total_hidden_units = sum(cfg.hidden_dims)
        num_layers = len(cfg.hidden_dims)
    else:
        # Fallback to old uniform architecture
        hidden_dim = getattr(cfg, 'hidden_dim', 64)
        depth = getattr(cfg, 'depth', 2)
        arch_str = f"{hidden_dim}x{depth}"
        total_hidden_units = hidden_dim * depth
        num_layers = depth
    
    # Prepare hyperparameters dictionary
    hparams = {
        'learning_rate': cfg.lr,
        'batch_size': cfg.batch_size,
        'total_hidden_units': total_hidden_units,
        'num_layers': num_layers,
        'physics_weight': cfg.physics_weight,
        'data_weight': cfg.data_weight,
        'n_collocation': cfg.n_collocation,
        'data_fraction': cfg.data_fraction,
        'm1': cfg.m1 if hasattr(cfg, 'm1') else 1.0,
        'm2': cfg.m2 if hasattr(cfg, 'm2') else 1.0,
        'l1': cfg.l1 if hasattr(cfg, 'l1') else 1.0,
        'l2': cfg.l2 if hasattr(cfg, 'l2') else 1.0,
        'g': cfg.g if hasattr(cfg, 'g') else 9.81
    }
    
    # Add individual layer dimensions if using custom architecture
    if hasattr(cfg, 'hidden_dims') and cfg.hidden_dims is not None:
        for i, dim in enumerate(cfg.hidden_dims):
            hparams[f'layer_{i}_dim'] = dim
    else:
        hparams['hidden_dim'] = getattr(cfg, 'hidden_dim', 64)
        hparams['depth'] = getattr(cfg, 'depth', 2)
    
    # Log hyperparameters to TensorBoard
    tb.add_hparams(hparams, {})
    
    # Also log architecture as text for easy reading
    tb.add_text('Model/Architecture', arch_str, 0)
    tb.add_text('Model/Type', cfg.model, 0)
    tb.add_text('Training/Optimizer', cfg.optimizer, 0)
    tb.add_text('Physics/Residual_Type', cfg.residual_type, 0)

    return writer, csv_file, tb, run_dir

def save_checkpoint(model, optimizer, cfg, run_dir, epoch, is_best=False, save_frequency=100):
    """Save model checkpoint with config and training state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(cfg),  # Save config as dict
    }
    
    # Save periodic checkpoint
    if epoch % save_frequency == 0:
        filename = f"checkpoint_epoch_{epoch}.pth"
        filepath = os.path.join(run_dir, "checkpoints", filename)
        torch.save(checkpoint, filepath)
    
    # Always save best
    if is_best:
        best_filepath = os.path.join(run_dir, "checkpoints", "best_model.pth")
        torch.save(checkpoint, best_filepath)

def load_checkpoint(checkpoint_path, model_class, device='cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    cfg = Config(**checkpoint['config'])
    
    # Reconstruct model using config
    model = model_class(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, cfg, checkpoint['epoch']