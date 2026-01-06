import torch.nn as nn
import torch
import os
import csv
from models.pinn import PINN
from training.losses import compute_loss
from utils.config import Config
from utils.logging import init_run

class Trainer:
    def __init__(self, model: PINN, config: Config,
                 train_loader, val_loader,
                 optimizer, scheduler=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        """
            Training loop for the PINN model.
        """
        writer, csv_file, tb, run_dir = init_run(self.config)
        
        for epoch in range(self.config.epochs):
            self.model.train()
            total_train_loss = 0.0
            total_physics_loss = 0.0
            total_data_loss = 0.0
            
            for batch in self.train_loader:
                t, state, point_type = batch
                t = t.to(self.device)
                state = state.to(self.device)
                point_type = point_type.to(self.device)
                
                self.optimizer.zero_grad()
                loss, loss_dict = compute_loss(
                    self.model, (t, state, point_type),
                    weight_data=self.config.data_weight,
                    weight_phys=self.config.physics_weight
                )
                loss.backward()
                
                if self.config.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
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
                      f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
            
            if self.scheduler:
                self.scheduler.step()
        
        csv_file.close()
        tb.close()
        print(f"Training complete. Logs saved to {run_dir}")

    def evaluate(self, val_loader):
        """
            Evaluate the model on validation dataset.
        """
        self.model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                t, state = batch  # Val loader only returns (t, state)
                t = t.to(self.device)
                state = state.to(self.device)
                
                # Create dummy point_type for validation (all data points)
                point_type = torch.zeros(t.size(0), dtype=torch.long, device=self.device)
                
                loss, _ = compute_loss(
                    self.model, (t, state, point_type),
                    weight_data=self.config.data_weight,
                    weight_phys=self.config.physics_weight
                )
                total_val_loss += loss.item() * t.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        return avg_val_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


