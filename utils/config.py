# config.py
from dataclasses import dataclass, field
from datetime import datetime

def _generate_run_name():
    """Generate default run name with timestamp."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

@dataclass
class Config:
    # experiment
    run_name: str = field(default_factory=_generate_run_name)
    seed: int = 0

    # data
    data_path: str = ""
    parameters_path: str = ""
    val_split: float = 0.2
    test_split: float = 0.1
    normalize_time: bool = True
    normalize_state: bool = True

    # model architecture
    model: str = "mlp"   # mlp | neural_ode | hnn | pinn
    
    # Flexible architecture
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"  # tanh | relu | gelu | silu | softplus
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    final_activation: str | None = None  # None | tanh | sigmoid
    
    # Input/output dimensions (auto-detect from data, but can override)
    input_dim: int | None = None
    output_dim: int | None = None

    # training
    lr: float = 1e-3
    batch_size: int = 128  # Batch size for data points
    batch_size_collocation: int | None = None  # Batch size for collocation points (None = same as batch_size)
    epochs: int = 500
    optimizer: str = "adam"  # adam | adamw | sgd
    weight_decay: float = 0.0
    grad_clip: float | None = None
    scheduler: str | None = None  # None | cosine | step
    
    # torch compile (PyTorch 2.0+)
    # Note: Disabled for PINNs due to double backward incompatibility
    use_compile: bool = False  # Enable torch.compile() for speedup
    compile_mode: str = "default"  # default | reduce-overhead | max-autotune
    
    # accelerate
    mixed_precision: bool = False  # Enable FP16 mixed precision training
    gradient_accumulation_steps: int = 1  # Gradient accumulation for larger effective batch size
    
    # model loading
    checkpoint_path: str | None = None  # Path to checkpoint file (.pth) for resuming training or transfer learning
    
    # regularization
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0  # alternative to weight_decay

    # physics / PINN
    use_physics: bool = True
    n_collocation: int = 5000
    data_fraction: float = 0.1
    physics_weight: float = 1.0
    data_weight: float = 1.0
    residual_type: str = "lagrangian"  # eom | hamiltonian | lagrangian

    # time domain
    t_min: float = 0.0
    t_max: float = 1.0
    collocation_sampling: str = "uniform"  # uniform | random | latin_hypercube

    # rollout evaluation
    rollout_T: float = 5.0
    rollout_dt: float = 0.01

    # logging
    log_interval: int = 10  # Log metrics to CSV/TensorBoard every N epochs (1 = every epoch)
    print_interval: int = 10  # Print progress to console every N epochs
    save_checkpoints: bool = True
    checkpoint_interval: int = 50
    
    # early stopping
    early_stopping_patience: int | None = None  # None = disabled, or number of epochs
    
    # Physical parameters (for double pendulum)
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81