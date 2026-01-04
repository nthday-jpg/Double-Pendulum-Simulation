import os
import csv
import yaml
from torch.utils.tensorboard import SummaryWriter

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
def init_run(cfg):
    run_dir = os.path.join("runs", cfg.run_name)
    os.makedirs(run_dir, exist_ok=False)

    # for saving model checkpoints
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    # for TensorBoard logs
    os.makedirs(os.path.join(run_dir, "tb"))

    # save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(cfg), f)

    # CSV setup
    csv_file = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "rollout_mse",
            "energy_drift"
        ]
    )
    writer.writeheader()

    # TensorBoard
    tb = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    return writer, csv_file, tb, run_dir
