import torch.nn as nn
import torch
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from accelerate import Accelerator
from models.pinn import PINN
from training.losses import compute_loss
from utils.config import Config
from utils.logging import init_run, save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, model: PINN, config: Config,
                 data_loader, collocation_loader, val_loader,
                 optimizer, scheduler=None):
        """
        Trainer with separate dataloaders for data and collocation points.
        Allows different batch sizes for each.
        
        Usage:
            config.batch_size = 32  # small for data
            config.batch_size_collocation = 256  # large for collocation
            data_loader, colloc_loader, val_loader = get_dataloader(...)
            trainer = Trainer(model, config, data_loader, colloc_loader, val_loader, optimizer)
        """
        self.accelerator = Accelerator(
            mixed_precision='fp16' if hasattr(config, 'mixed_precision') and config.mixed_precision else 'no',
            gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1)
        )
        
        self.config = config
        
        # Disable torch.compile for PINNs (incompatible with double backward)
        if getattr(config, 'use_compile', False):
            print("⚠ torch.compile disabled: incompatible with PINN double backward")
        
        # Prepare model, optimizer, and dataloaders
        self.model, self.optimizer, self.data_loader, self.collocation_loader, self.val_loader = self.accelerator.prepare(
            model, optimizer, data_loader, collocation_loader, val_loader
        )
        
        self.scheduler = scheduler
        self.device = self.accelerator.device

        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.patience_counter = 0
        
        if config.checkpoint_path and os.path.exists(config.checkpoint_path):
            self._resume_from_checkpoint(config.checkpoint_path)

    def _resume_from_checkpoint(self, checkpoint_path):
        """Resume training from checkpoint."""
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        print(f"✓ Resumed from epoch {checkpoint.get('epoch', 'unknown')}, best_val_loss: {self.best_val_loss:.6f}")

    def train(self):
        """Training loop for PINN model."""
        # Only main process initializes logging to avoid race conditions
        if self.accelerator.is_main_process:
            writer, csv_file, tb, run_dir = init_run(self.config)
            self.run_dir = run_dir
            self.best_model_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
        else:
            writer, csv_file, tb, run_dir = None, None, None, None
            self.run_dir = None
            self.best_model_path = None
        
        self.accelerator.wait_for_everyone()
        
        epoch = 0
        
        if self.accelerator.is_main_process:
            print("="*60, flush=True)
            print(f"Starting training for {self.config.epochs} epochs...", flush=True)
            print(f"Device: {self.device}", flush=True)
            print(f"Data batch: {self.config.batch_size}, Collocation batch: {self.config.batch_size_collocation or self.config.batch_size}", flush=True)
            print("="*60, flush=True)
        
        try:
            for epoch in range(self.config.epochs):
                self.model.train()
                total_train_loss = 0.0
                total_physics_loss = 0.0
                total_data_loss = 0.0
                total_samples = 0
                
                # Alternate between data and collocation batches
                data_iter = iter(self.data_loader)
                colloc_iter = itertools.cycle(self.collocation_loader)
                
                batch_count = 0
                for data_batch in data_iter:
                    batch_count += 1
                    
                    total_train_loss, total_physics_loss, total_data_loss, total_samples = self._train_step(
                        data_batch, total_train_loss, total_physics_loss, total_data_loss, total_samples
                    )
                    
                    # Process collocation batches based on data_fraction
                    n_colloc = max(1, int((1 - self.config.data_fraction) / self.config.data_fraction))
                    for _ in range(n_colloc):
                        colloc_batch = next(colloc_iter)
                        total_train_loss, total_physics_loss, total_data_loss, total_samples = self._train_step(
                            colloc_batch, total_train_loss, total_physics_loss, total_data_loss, total_samples
                        )
                
                # Compute epoch averages
                avg_train_loss = total_train_loss / total_samples
                avg_physics_loss = total_physics_loss / total_samples
                avg_data_loss = total_data_loss / total_samples
                
                # Synchronize before validation
                self.accelerator.wait_for_everyone()
                
                # Validation
                avg_val_loss = self.evaluate(self.val_loader)
                
                # Synchronize after validation
                self.accelerator.wait_for_everyone()
                
                # Logging - only main process
                if self.accelerator.is_main_process:
                    log_interval = getattr(self.config, 'log_interval', 1)
                    if (epoch + 1) % log_interval == 0 or epoch == 0:
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
                    
                    print_interval = getattr(self.config, 'print_interval', 10)
                    if (epoch + 1) % print_interval == 0 or epoch == 0:
                        print(f"Epoch {epoch+1}/{self.config.epochs} - "
                            f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, "
                            f"Physics: {avg_physics_loss:.6f}, Data: {avg_data_loss:.6f}")
                
                # Synchronize before early stopping check
                self.accelerator.wait_for_everyone()
                
                # Early stopping check
                if self._check_early_stopping(avg_val_loss, epoch):
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                if self.scheduler:
                    self.scheduler.step()
        
        except Exception as e:
            # Log error to file for debugging
            import traceback
            error_msg = f"\n{'='*60}\n"
            error_msg += f"ERROR at epoch {epoch + 1}\n"
            error_msg += f"Process rank: {self.accelerator.process_index}\n"
            error_msg += f"Error type: {type(e).__name__}\n"
            error_msg += f"Error: {str(e)}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}\n"
            error_msg += f"{'='*60}\n"
            
            print(error_msg, flush=True)  # Print regardless of rank
            
            # Save to file
            if hasattr(self, 'run_dir') and self.run_dir:
                error_file = os.path.join(self.run_dir, f"error_rank_{self.accelerator.process_index}.txt")
                with open(error_file, 'w') as f:
                    f.write(error_msg)
            
            if self.accelerator.is_main_process:
                print("Saving current model state...", flush=True)
                try:
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                except (KeyError, AttributeError):
                    unwrapped_model = self.model
                if hasattr(self, 'run_dir') and self.run_dir:
                    save_checkpoint(unwrapped_model, self.optimizer, self.config, run_dir, epoch + 1)
            raise  # Re-raise to see full traceback
        
        finally:
            if self.accelerator.is_main_process:
                if csv_file:
                    csv_file.close()
                if tb:
                    tb.close()
                print(f"Training complete. Logs saved to {run_dir}")
                if self.best_model_path and os.path.exists(self.best_model_path):
                    print(f"Best model saved at: {self.best_model_path}")
                
                if hasattr(self, 'run_dir') and self.run_dir:
                    self.plot_losses(self.run_dir)
    
    def _train_step(self, batch, total_train_loss, total_physics_loss, total_data_loss, total_samples):
        """Single training step - shared between mixed and separate modes."""
        t, state, point_type = batch
        
        self.optimizer.zero_grad()
        loss, loss_dict = compute_loss(
            self.model, (t, state, point_type),
            weight_data=self.config.data_weight,
            weight_phys=self.config.physics_weight
        )
        
        self.accelerator.backward(loss)
        
        if self.config.grad_clip:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        
        # Accumulate losses
        batch_size = t.size(0)
        total_train_loss += loss.item() * batch_size
        total_physics_loss += loss_dict["physics_loss"] * batch_size
        total_data_loss += loss_dict["data_loss"] * batch_size
        total_samples += batch_size
        
        return total_train_loss, total_physics_loss, total_data_loss, total_samples
    
    def _check_early_stopping(self, val_loss, epoch):
        """Check if early stopping criteria is met."""
        if not hasattr(self.config, 'early_stopping_patience') or self.config.early_stopping_patience is None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.accelerator.is_main_process and self.best_model_path and hasattr(self, 'run_dir') and self.run_dir:
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    save_checkpoint(unwrapped_model, self.optimizer, self.config, 
                                   self.run_dir, epoch + 1, is_best=True)
                    print(f"  → New best model saved (val_loss: {val_loss:.6f})")
            return False
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            if self.accelerator.is_main_process and self.best_model_path and hasattr(self, 'run_dir') and self.run_dir:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                save_checkpoint(unwrapped_model, self.optimizer, self.config, 
                               self.run_dir, epoch + 1, is_best=True)
                print(f"  → New best model saved (val_loss: {val_loss:.6f})")
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience

    def evaluate(self, val_loader):
        """Evaluate the model on validation dataset."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.eval()
        
        total_val_loss = 0.0
        total_samples = 0
        
        for batch in val_loader:
            t, state, point_type = batch  # Changed from t, state = batch
            t = t.to(self.device).view(-1, 1)
            state = state.to(self.device)
            t = t.detach().requires_grad_(True)

            with torch.enable_grad():
                with self.accelerator.autocast():
                    loss, _ = compute_loss(
                        unwrapped_model, (t, state, point_type),
                        weight_data=self.config.data_weight,
                        weight_phys=self.config.physics_weight
                    )
            
            batch_size = t.size(0)
            total_val_loss += loss.item() * batch_size
            total_samples += batch_size
        
        # Gather losses from all processes
        total_val_loss = torch.tensor(total_val_loss, device=self.device)
        total_samples = torch.tensor(total_samples, device=self.device)
        
        total_val_loss = self.accelerator.gather(total_val_loss).sum().item()
        total_samples = self.accelerator.gather(total_samples).sum().item()
        
        avg_val_loss = total_val_loss / total_samples if total_samples > 0 else 0.0
        return avg_val_loss

    def plot_losses(self, run_dir):
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

    def save_model(self, path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), path)

    def load_model(self, path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(torch.load(path, map_location=self.device))


