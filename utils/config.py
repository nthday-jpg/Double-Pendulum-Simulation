from dataclasses import dataclass


@dataclass
class Config:
    """
        This is cfg object passed to logging utilities.
    """
    
    # experiment
    run_name: str
    seed: int = 0

    # model
    model: str = "mlp"      # mlp | neural_ode | hnn
    hidden_dim: int = 64
    depth: int = 2

    # training
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 500

    # rollout evaluation
    rollout_T: float = 5.0
    rollout_dt: float = 0.01
