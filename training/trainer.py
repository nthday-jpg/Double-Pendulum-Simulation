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
                 data_loader, collocation_loader, val_loader, test_loader,
                 optimizer, scheduler=None):
        """
        Trainer with separate dataloaders for data and collocation points.
        Allows different batch sizes for each.
        
        Usage:
            config.batch_size = 32  # small for data
            config.batch_size_collocation = 256  # large for collocation
            data_loader, colloc_loader, val_loader, test_loader = get_dataloader(...)
            trainer = Trainer(model, config, data_loader, colloc_loader, val_loader, test_loader, optimizer)
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
        self.model, self.optimizer, self.data_loader, self.collocation_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
            model, optimizer, data_loader, collocation_loader, val_loader, test_loader
        )
        
        self.scheduler = scheduler
        self.device = self.accelerator.device

        self.best_val_loss = float('inf')
        # Initialize on all processes to avoid None causing deadlocks
        self.best_model_path = ""
        self.patience_counter = 0
        
        # Check and load checkpoint if provided
        if hasattr(config, 'checkpoint_path') and config.checkpoint_path:
            if os.path.exists(config.checkpoint_path):
                self._resume_from_checkpoint(config.checkpoint_path)
            else:
                if self.accelerator.is_main_process:
                    print(f"⚠ Checkpoint path provided but not found: {config.checkpoint_path}")
                    print(f"  Starting training from scratch...")

    def _resume_from_checkpoint(self, checkpoint_path):
        """Resume training from checkpoint."""
        if self.accelerator.is_main_process:
            print(f"📁 Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        if self.accelerator.is_main_process:
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
            self.best_model_path = ""
        
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
                val_metrics = self.evaluate(self.val_loader, prefix="val")
                avg_val_loss = val_metrics["total_loss"]
                
                # Synchronize after validation
                self.accelerator.wait_for_everyone()
                
                # Logging - only main process
                if self.accelerator.is_main_process:
                    log_interval = getattr(self.config, 'log_interval', 1)
                    if (epoch + 1) % log_interval == 0 or epoch == 0:
                        log_dict = {
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "train_physics": avg_physics_loss,
                            "train_data": avg_data_loss,
                            "val_loss": avg_val_loss,
                            "val_physics": val_metrics["physics_loss"],
                            "val_data": val_metrics["data_loss"]
                        }
                        
                        writer.writerow(log_dict) # type: ignore
                        csv_file.flush() # type: ignore
                        
                        tb.add_scalar("Loss/Train", avg_train_loss, epoch + 1) # type: ignore
                        tb.add_scalar("Loss/Train_Physics", avg_physics_loss, epoch + 1) # type: ignore
                        tb.add_scalar("Loss/Train_Data", avg_data_loss, epoch + 1) # type: ignore
                        tb.add_scalar("Loss/Val", avg_val_loss, epoch + 1)  # type: ignore
                        tb.add_scalar("Loss/Val_Physics", val_metrics["physics_loss"], epoch + 1) # type: ignore
                        tb.add_scalar("Loss/Val_Data", val_metrics["data_loss"], epoch + 1) # type: ignore
                    
                    print_interval = getattr(self.config, 'print_interval', 10)
                    if (epoch + 1) % print_interval == 0 or epoch == 0:
                        self._print_beautiful_log(epoch + 1, avg_train_loss, avg_physics_loss, avg_data_loss,
                                                 val_metrics)
                    
                    if (epoch + 1) % getattr(self.config, 'test_interval', 50) == 0:
                        self.evaluate_test_set()

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
        t, initial_state, state, point_type = batch
        
        self.optimizer.zero_grad()
        loss, loss_dict = compute_loss(
            self.model, (t, initial_state, state, point_type),
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

    def evaluate(self, val_loader, prefix="val"):
        """Evaluate the model on validation/test dataset with detailed metrics."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.eval()
        
        total_loss = 0.0
        total_physics_loss = 0.0
        total_data_loss = 0.0
        total_samples = 0
        
        for batch in val_loader:
            t, initial_state, state, point_type = batch
            t = t.view(-1, 1)
            
            # Detach and require grad for physics computation
            t = t.detach().requires_grad_(True)
            
            # Enable gradients only for this batch computation
            with torch.enable_grad():
                with self.accelerator.autocast():
                    loss, loss_dict = compute_loss(
                        unwrapped_model, (t, initial_state, state, point_type),
                        weight_data=self.config.data_weight,
                        weight_phys=self.config.physics_weight
                    )
            
            # Extract loss values and immediately free the computation graph
            batch_size = t.size(0)
            total_loss += loss.item() * batch_size
            total_physics_loss += loss_dict["physics_loss"] * batch_size
            total_data_loss += loss_dict["data_loss"] * batch_size
            total_samples += batch_size
            
            # Explicitly delete tensors to free GPU memory
            del loss, t, initial_state, state, point_type
        
        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()
        
        # Gather losses from all processes
        total_loss = torch.tensor(total_loss, device=self.device)
        total_physics_loss = torch.tensor(total_physics_loss, device=self.device)
        total_data_loss = torch.tensor(total_data_loss, device=self.device)
        total_samples = torch.tensor(total_samples, device=self.device)
        
        total_loss = self.accelerator.gather(total_loss).sum().item()
        total_physics_loss = self.accelerator.gather(total_physics_loss).sum().item()
        total_data_loss = self.accelerator.gather(total_data_loss).sum().item()
        total_samples = self.accelerator.gather(total_samples).sum().item()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_physics = total_physics_loss / total_samples if total_samples > 0 else 0.0
        avg_data = total_data_loss / total_samples if total_samples > 0 else 0.0
        
        return {
            "total_loss": avg_loss,
            "physics_loss": avg_physics,
            "data_loss": avg_data
        }

    def _print_beautiful_log(self, epoch, train_loss, train_physics, train_data, val_metrics):
        """Print beautifully formatted training logs."""
        print("\n" + "="*80)
        print(f"{'Epoch':<15} {epoch}/{self.config.epochs}")
        print("="*80)
        
        # Header
        print(f"{'Dataset':<15} {'Total Loss':<15} {'Physics Loss':<15} {'Data Loss':<15}")
        print("-"*80)
        
        # Training
        print(f"{'Train':<15} {train_loss:<15.6f} {train_physics:<15.6f} {train_data:<15.6f}")
        
        # Validation
        print(f"{'Validation':<15} {val_metrics['total_loss']:<15.6f} {val_metrics['physics_loss']:<15.6f} {val_metrics['data_loss']:<15.6f}")
        
        print("="*80 + "\n")
    
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
        plt.show()
        plt.close()

    def save_model(self, path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), path)

    def load_model(self, path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(torch.load(path, map_location=self.device))
    
    def evaluate_test_set(self):
        """Evaluate the model on test set (temporal extrapolation)."""
        
        if self.accelerator.is_main_process:
            print("\n" + "="*80)
            print("Evaluating on Test Set (Temporal Extrapolation)")
            print("="*80)
        
        # Clear GPU cache before test evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ALL processes load the checkpoint independently (simple and safe)
        if self.best_model_path and os.path.exists(self.best_model_path):
            if self.accelerator.is_main_process:
                print(f"Loading best model from: {self.best_model_path}")
            
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        elif self.accelerator.is_main_process:
            print("No checkpoint found, using current model state")
        
        # Set model to eval mode
        self.model.eval()
        
        # Evaluate using same method as validation
        if self.accelerator.is_main_process:
            print("Evaluating test set...")
        test_metrics = self.evaluate(self.test_loader, prefix="test")
        
        if (self.accelerator.is_main_process):
            print("\nTest Set Results:")
            print("-"*80)
            print(f"{'Metric':<30} {'Value':<15}")
            print("-"*80)
            print(f"{'Total Loss':<30} {test_metrics['total_loss']:<15.6f}")
            print(f"{'Physics Loss':<30} {test_metrics['physics_loss']:<15.6f}")
            print(f"{'Data Loss':<30} {test_metrics['data_loss']:<15.6f}")
            print("="*80 + "\n")
            
            # Save test results to file if run_dir is available
            if hasattr(self, 'run_dir') and self.run_dir:
                test_results_file = os.path.join(self.run_dir, "test_results.txt")
                with open(test_results_file, 'w') as f:
                    f.write("Test Set Evaluation Results\n")
                    f.write("="*80 + "\n")
                    f.write(f"Total Loss:    {test_metrics['total_loss']:.6f}\n")
                    f.write(f"Physics Loss:  {test_metrics['physics_loss']:.6f}\n")
                    f.write(f"Data Loss:     {test_metrics['data_loss']:.6f}\n")
                    f.write("="*80 + "\n")
                print(f"Test results saved to: {test_results_file}")
