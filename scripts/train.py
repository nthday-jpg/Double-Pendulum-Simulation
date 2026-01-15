import sys
import os
import argparse
import warnings
import torch
from datetime import datetime  

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
    
    # Experiment
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name (auto-generated if not provided)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing trajectory files (trajectory_*.npz) and parameters (parameters_*.json)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--normalize_time', action='store_true',
                        help='Normalize time to [0, 1]')
    
    # Model architecture
    parser.add_argument('--model', type=str, default='pinn',
                        choices=['mlp', 'neural_ode', 'hnn', 'pinn'],
                        help='Model type')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'gelu', 'silu', 'softplus'],
                        help='Activation function')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use batch normalization')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--final_activation', type=str, default=None,
                        choices=['tanh', 'sigmoid'],
                        help='Final layer activation')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=2,
                        help='Output dimension')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Gradient clipping value')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['cosine', 'step'],
                        help='Learning rate scheduler')
    
    # PyTorch optimizations
    parser.add_argument('--use_compile', action='store_true',
                        help='Use torch.compile (disabled for PINNs)')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='Compile mode')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable FP16 mixed precision')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    


    # Model loading
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Regularization
    parser.add_argument('--l1_lambda', type=float, default=0.0,
                        help='L1 regularization weight')
    parser.add_argument('--l2_lambda', type=float, default=0.0,
                        help='L2 regularization weight')
    
    # Physics / PINN
    parser.add_argument('--data_loss_ratio', type=float, default=0.1,
                        help='Fraction of total loss from data (rest is physics)')
    parser.add_argument('--residual_type', type=str, default='lagrangian',
                        choices=['eom', 'lagrangian', 'hamiltonian'],
                        help='Type of physics residual')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log to CSV/TensorBoard every N epochs')
    parser.add_argument('--print_interval', type=int, default=10,
                        help='Print to console every N epochs')
    parser.add_argument('--save_checkpoints', action='store_true', default=True,
                        help='Save model checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--test_interval', type=int, default=50,
                        help='Evaluate on test set every N epochs')

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Early stopping patience (None = disabled)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4,
                        help='Minimum change to qualify as improvement for early stopping') 

    # Physical parameters
    parser.add_argument('--m1', type=float, default=1.0,
                        help='Mass of first pendulum')
    parser.add_argument('--m2', type=float, default=1.0,
                        help='Mass of second pendulum')
    parser.add_argument('--l1', type=float, default=1.0,
                        help='Length of first pendulum')
    parser.add_argument('--l2', type=float, default=1.0,
                        help='Length of second pendulum')
    parser.add_argument('--g', type=float, default=9.81,
                        help='Gravitational acceleration')
    
    return parser.parse_args()


def main():
    """Main training function for distributed training with Accelerate."""
    
    # Parse arguments
    args = parse_args()
    
    # Generate run_name if not provided
    run_name = args.run_name if args.run_name else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Configuration - pass all arguments to Config
    cfg = Config(
        # Experiment
        run_name=run_name,  
        seed=args.seed,
        
        # Data
        val_split=args.val_split,
        test_split=args.test_split,
        normalize_time=args.normalize_time,
        
        # Model architecture
        model=args.model,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        use_batch_norm=args.use_batch_norm,
        dropout_rate=args.dropout_rate,
        final_activation=args.final_activation,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        
        # Training
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        scheduler=args.scheduler,
        
        # PyTorch optimizations
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Model loading
        checkpoint_path=args.checkpoint_path,
        
        # Regularization
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
        
        # Physics / PINN
        data_loss_ratio=args.data_loss_ratio,
        residual_type=args.residual_type,
        
        # Logging
        log_interval=args.log_interval,
        print_interval=args.print_interval,
        save_checkpoints=args.save_checkpoints,
        checkpoint_interval=args.checkpoint_interval,

        test_interval=args.test_interval,
        
        # Early stopping
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,

        # Physical parameters
        m1=args.m1,
        m2=args.m2,
        l1=args.l1,
        l2=args.l2,
        g=args.g
    )
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    print(f"Random seed set to: {cfg.seed}")
    
    # Initialize model
    model = PINN(cfg)
    
    # Initialize optimizer based on config
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    # Initialize scheduler if specified
    scheduler = None
    if cfg.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=cfg.scheduler_patience
        )
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloader(
        data_dir=args.data_dir,
        config=cfg
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=cfg,
        data_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
