# config.py
from dataclasses import dataclass, field

@dataclass
class Config:
    # experiment
    run_name: str
    seed: int = 0

    # data
    data_path: str = ""
    parameters_path: str = ""
    val_split: float = 0.2
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
    batch_size: int = 128
    epochs: int = 500
    optimizer: str = "adam"  # adam | adamw | sgd
    weight_decay: float = 0.0
    grad_clip: float | None = None
    scheduler: str | None = None  # None | cosine | step
    
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
    log_interval: int = 100
    save_checkpoints: bool = True
    checkpoint_interval: int = 50
    
    # Physical parameters (for double pendulum)
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81