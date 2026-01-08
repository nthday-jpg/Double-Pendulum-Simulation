import torch.nn as nn
import torch
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from accelerate import Accelerator
from models.pinn import PINN
from training.losses import compute_loss
from utils.config import Config
from utils.logging import init_run, save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, model: PINN, config: Config,
                 train_loader, val_loader,
                 optimizer, scheduler=None):
        # Initialize Accelerator for automatic device placement and mixed precision
        self.accelerator = Accelerator(
            mixed_precision='fp16' if hasattr(config, 'mixed_precision') and config.mixed_precision else 'no',
            gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1)
        )
        
        self.config = config
        
        # Prepare model, optimizer, and dataloaders with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        
        self.scheduler = scheduler
        self.device = self.accelerator.device

        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.patience_counter = 0
        
        # Load checkpoint if specified (for resuming training)
        if config.checkpoint_path and os.path.exists(config.checkpoint_path):
            self._resume_from_checkpoint(config.checkpoint_path)

    def _resume_from_checkpoint(self, checkpoint_path):
        """Resume training from checkpoint (loads model + optimizer state)."""
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state (important for resuming training)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Resumed from epoch {checkpoint.get('epoch', 'unknown')}, best_val_loss: {self.best_val_loss:.6f}")

    def train(self):
        """
            Training loop for the PINN model.
        """
        writer, csv_file, tb, run_dir = init_run(self.config)
        self.run_dir = run_dir
        self.best_model_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
        epoch = 0  # Initialize to avoid unbound variable
        
        print("="*60, flush=True)
        print(f"Starting training for {self.config.epochs} epochs...", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Using Accelerator with mixed precision: {self.accelerator.mixed_precision}", flush=True)
        print("="*60, flush=True)
        
        try: 
            for epoch in range(self.config.epochs):
                self.model.train()
                total_train_loss = 0.0
                total_physics_loss = 0.0
                total_data_loss = 0.0
                
                for batch in self.train_loader:
                    t, state, point_type = batch
                    # Accelerator handles device placement automatically
                    # No need to call requires_grad here, compute_loss handles it
                    
                    self.optimizer.zero_grad()
                    loss, loss_dict = compute_loss(
                        self.model, (t, state, point_type),
                        weight_data=self.config.data_weight,
                        weight_phys=self.config.physics_weight
                    )
                    
                    # Use accelerator's backward for mixed precision
                    self.accelerator.backward(loss)
                    
                    if self.config.grad_clip:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    self.optimizer.step()
                    
                    # Accumulate losses
                    batch_size = t.size(0)
                    total_train_loss += loss.item() * batch_size
                    total_physics_loss += loss_dict["physics_loss"] * batch_size
                    total_data_loss += loss_dict["data_loss"] * batch_size
                
                # Compute epoch averages
                dataset_size = len(self.train_loader.dataset)
                avg_train_loss = total_train_loss / dataset_size
                avg_physics_loss = total_physics_loss / dataset_size
                avg_data_loss = total_data_loss / dataset_size
                
                # Validation
                avg_val_loss = self.evaluate(self.val_loader)
                
                # Logging
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "physics_loss": avg_physics_loss,
                    "data_loss": avg_data_loss
                }
                writer.writerow(log_dict)
                csv_file.flush()
                
                tb.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
                tb.add_scalar("Loss/Val", avg_val_loss, epoch + 1)
                tb.add_scalar("Loss/Physics", avg_physics_loss, epoch + 1)
                tb.add_scalar("Loss/Data", avg_data_loss, epoch + 1)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs} - "
                        f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, "
                        f"Physics: {avg_physics_loss:.6f}, Data: {avg_data_loss:.6f}")
                
                # Early stopping check
                if self._check_early_stopping(avg_val_loss, epoch):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                if self.scheduler:
                    self.scheduler.step()
        
        except KeyboardInterrupt:
            print(f"\nTraining interrupted at epoch {epoch + 1}")
            print("Saving current model state...")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_checkpoint(unwrapped_model, self.optimizer, self.config, run_dir, epoch + 1)
        
        finally:
            csv_file.close()
            tb.close()
            print(f"Training complete. Logs saved to {run_dir}")
            if self.best_model_path and os.path.exists(self.best_model_path):
                print(f"Best model saved at: {self.best_model_path}")
            
            # Plot losses
            if hasattr(self, 'run_dir'):
                self.plot_losses(self.run_dir)
    
    def _check_early_stopping(self, val_loss, epoch):
        """
            Check if early stopping criteria is met.
            Returns True if training should stop.
        """
        if not hasattr(self.config, 'early_stopping_patience') or self.config.early_stopping_patience is None:
            # If early stopping not configured, just save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Note: save_checkpoint saves to run_dir/checkpoints/best_model.pth
                if self.best_model_path and hasattr(self, 'run_dir'):
                    # Unwrap model for saving
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    save_checkpoint(unwrapped_model, self.optimizer, self.config, 
                                   self.run_dir, epoch + 1, is_best=True)
                    print(f"  → New best model saved (val_loss: {val_loss:.6f})")
            return False
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.best_model_path and hasattr(self, 'run_dir'):
                # Unwrap model for saving
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                save_checkpoint(unwrapped_model, self.optimizer, self.config, 
                               self.run_dir, epoch + 1, is_best=True)
                print(f"  → New best model saved (val_loss: {val_loss:.6f})")
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience

    def evaluate(self, val_loader):
        """
            Evaluate the model on validation dataset.
            PINNs require gradient graph even in validation for physics loss.
        """
        # Use unwrapped model to bypass accelerator's inference optimizations
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.eval()  # Still use eval mode for Dropout/BatchNorm
        
        total_val_loss = 0.0
        
        for batch in val_loader:
            t, state = batch  # Val loader only returns (t, state)
            
            # Prepare tensors
            t = t.to(self.device).view(-1, 1)
            state = state.to(self.device)
            point_type = torch.zeros(t.size(0), dtype=torch.long, device=self.device)
            
            t = t.detach().requires_grad_(True)

            # Enable gradients for physics loss computation
            with torch.enable_grad():
                with self.accelerator.autocast():
                    # Pass unwrapped model to compute_loss
                    loss, _ = compute_loss(
                        unwrapped_model, (t, state, point_type),
                        weight_data=self.config.data_weight,
                        weight_phys=self.config.physics_weight
                    )
            total_val_loss += loss.item() * t.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        return avg_val_loss

    def plot_losses(self, run_dir):
        """Plot training and validation losses."""
        metrics_file = os.path.join(run_dir, "metrics.csv")
        
        if not os.path.exists(metrics_file):
            print(f"Metrics file not found: {metrics_file}")
            return
        
        # Read metrics
        df = pd.read_csv(metrics_file)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2)
        ax.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plot_path = os.path.join(run_dir, "loss_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Loss plot saved to: {plot_path}")
        plt.close()

    

    def save_model(self, path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), path)

    def load_model(self, path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(torch.load(path, map_location=self.device))


