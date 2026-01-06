import torch.nn as nn
import torch
from models.pinn import PINN
from losses import compute_loss
from utils.config import Config
from utils.logging import init_run

def train(model, cfg, train_loader, val_loader, optimizer, scheduler=None):
    """
        Training loop for the PINN model.
        Args:
            model: PINN model instance.
            cfg: Configuration object.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
    """
    writer, csv_file, tb, run_dir = init_run(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            t, state, point_type = batch
            t, state, point_type = t.to(device), state.to(device), point_type.to(device)
            optimizer.zero_grad()
            loss, loss_dict = compute_loss(model, (t, state, point_type),
                                          weight_data=cfg.data_weight,
                                          weight_phys=cfg.physics_weight)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_train_loss += loss.item() * t.size(0)
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                t, state, point_type = [x.to(device) for x in batch]
                loss, _ = compute_loss(model, (t, state, point_type),
                                      weight_data=cfg.data_weight,
                                      weight_phys=cfg.physics_weight)
                total_val_loss += loss.item() * t.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        
        # Logging
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }
        log_dict.update(loss_dict)  # type: ignore
        writer.writerow(log_dict)
        csv_file.flush()
        
        tb.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        tb.add_scalar("Loss/Val", avg_val_loss, epoch + 1)
        
        if scheduler:
            scheduler.step()

    

    