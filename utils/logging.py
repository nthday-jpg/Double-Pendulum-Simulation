# logging.py
import os
import csv
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from utils.config import Config

def init_run(cfg: Config):
    """Initialize logging for a training run."""
    run_dir = os.path.join("runs", cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # CSV logging
    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    fieldnames = [
        "epoch", 
        "train_loss", "train_physics", "train_data",
        "val_loss", "val_physics", "val_data",
        "lr"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    # TensorBoard logging
    tb = SummaryWriter(log_dir=run_dir)
    
    print(f"📊 Logging to: {run_dir}")
    return writer, csv_file, tb, run_dir

def save_checkpoint(model, optimizer, cfg, run_dir, epoch, is_best=False, save_frequency=100):
    """Save model checkpoint with config, training state, and normalization parameters."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(cfg),  # Save config as dict
        # Save normalization parameters for inference
        'normalize_time': getattr(cfg, 'normalize_time', True),
        'normalize_angles': getattr(cfg, 'normalize_angles', True),
        't_min': getattr(cfg, 't_min', 0.0),
        't_max': getattr(cfg, 't_max', 1.0),
        'theta_min': getattr(cfg, 'theta_min', -np.pi),
        'theta_max': getattr(cfg, 'theta_max', np.pi),
    }
    
    # Ensure checkpoints directory exists
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Save periodic checkpoint
    if epoch % save_frequency == 0:
        filename = f"checkpoint_epoch_{epoch}.pth"
        filepath = os.path.join(checkpoints_dir, filename)
        torch.save(checkpoint, filepath)
    
    # Always save best
    if is_best:
        best_filepath = os.path.join(checkpoints_dir, "best_model.pth")
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