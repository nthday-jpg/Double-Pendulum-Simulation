# logging.py
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
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
        "train_loss", "train_physics", "train_trajectory", "train_kinetic",
        "val_loss", "val_physics", "val_trajectory", "val_kinetic",
        "lr"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    # TensorBoard logging
    tb = SummaryWriter(log_dir=run_dir)
    
    print(f"📊 Logging to: {run_dir}")
    return writer, csv_file, tb, run_dir

def save_checkpoint(model, optimizer, scheduler, cfg, run_dir, epoch, is_best=False, best_val_loss=None, save_frequency=100, time_scale=None):
    """Save model checkpoint with config, training state, and dataset normalization parameters.
    
    Args:
        time_scale : Actual time scale from training dataset (for proper inference normalization)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': vars(cfg),  # Config contains all settings including normalize_time, t_min, t_max, etc.
        'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        'time_scale': time_scale  if time_scale is not None else getattr(cfg, 'time_scale', 1.0),  # Actual dataset max time
    }
    
    # Ensure checkpoints directory exists
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Save periodic checkpoint
    if epoch % save_frequency == 0:
        filename = f"checkpoint_epoch_{epoch}.pth"
        filepath = os.path.join(checkpoints_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    # Always save best
    if is_best:
        best_filepath = os.path.join(checkpoints_dir, "best_model.pth")
        torch.save(checkpoint, best_filepath)
        print(f"Saved best model checkpoint: {best_filepath}")
        
# Load checkpoint is implemented in trainer.py to access Trainer attributes

def plot_losses(run_dir):
    """Plot training and validation losses."""
    metrics_file = os.path.join(run_dir, "metrics.csv")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    
    df = pd.read_csv(metrics_file)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(run_dir, "loss_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Loss plot saved to: {plot_path}")
    plt.close()

def print_beautiful_log(config, epoch, train_loss, train_physics, train_trajectory, train_kinetic, val_metrics):
        """Print beautifully formatted training logs."""
        print("\n" + "="*80)
        print(f"{'Epoch':<15} {epoch}/{config.epochs}")
        print("="*80)
        
        # Header
        print(f"{'Dataset':<15} {'Total Loss':<15} {'Physics Loss':<15} {'Data Loss':<15} {'Kinetic Loss':<15}")
        print("-"*80)
        
        # Training
        print(f"{'Train':<15} {train_loss:<15.6f} {train_physics:<15.6f} {train_trajectory:<15.6f} {train_kinetic:<15.6f}")
        
        # Validation
        print(f"{'Validation':<15} {val_metrics['total_loss']:<15.6f} {val_metrics['physics_loss']:<15.6f} {val_metrics['trajectory_loss']:<15.6f} {val_metrics['kinetic_loss']:<15.6f}")
        
        print("="*80 + "\n")
    