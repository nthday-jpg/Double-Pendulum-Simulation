import sys
import os
import argparse
import warnings
import torch

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Grad strides do not match.*')
warnings.filterwarnings('ignore', message='.*The hostname of the client socket.*')
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# NCCL timeout settings (increase to 30 minutes for long operations)
os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes in seconds
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'  # Set to INFO for more debugging

# Set up paths for Kaggle environment
base_path = '/kaggle/working'
project_name = 'Double-Pendulum-Simulation'
project_root = os.path.join(base_path, project_name)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.pinn import PINN
from training.trainer import Trainer
from data.dataset import get_dataloader
from utils.config import Config
from utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PINN model for double pendulum')
    
    # Model architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=2,
                        help='Output dimension')
    
    # Physics
    parser.add_argument('--residual_type', type=str, default='lagrangian',
                        choices=['lagrangian', 'hamiltonian'],
                        help='Type of physics residual')
    parser.add_argument('--t_max', type=float, default=5.0,
                        help='Maximum time')
    parser.add_argument('--t_min', type=float, default=0.0,
                        help='Minimum time')
    
    # Training
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data')
    parser.add_argument('--batch_size_collocation', type=int, default=512,
                        help='Batch size for collocation points')
    parser.add_argument('--use_compile', action='store_true',
                        help='Use torch.compile')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data_fraction', type=float, default=0.1,
                        help='Fraction of data to use for training')
    
    # Data paths
    parser.add_argument('--input_path', type=str, default='/kaggle/input/double-pendulum',
                        help='Input data directory')
    parser.add_argument('--data_file', type=str, default='trajectory_000.npz',
                        help='Trajectory data file')
    parser.add_argument('--params_file', type=str, default='parameters_000.json',
                        help='Parameters file')
    
    return parser.parse_args()


def main():
    """Main training function for distributed training with Accelerate."""
    
    # Parse arguments
    args = parse_args()
    
    # Configuration
    cfg = Config(
        hidden_dims=args.hidden_dims,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        residual_type=args.residual_type,
        t_max=args.t_max,
        t_min=args.t_min,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        batch_size_collocation=args.batch_size_collocation,
        use_compile=args.use_compile,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        data_fraction=args.data_fraction
    )
    
    # Override lr if provided
    if hasattr(args, 'lr'):
        cfg.lr = args.lr
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    print(f"🌱 Random seed set to: {cfg.seed}")
    
    # Initialize model
    model = PINN(cfg)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Get data loaders
    data_loader, colloc_loader, val_loader = get_dataloader(
        data_path=f"{args.input_path}/{args.data_file}",
        parameters_path=f"{args.input_path}/{args.params_file}",
        config=cfg
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=cfg,
        data_loader=data_loader,
        collocation_loader=colloc_loader,
        val_loader=val_loader,
        optimizer=optimizer
    )
    
    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
