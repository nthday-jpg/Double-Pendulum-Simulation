# model.py
import torch
import torch.nn as nn
from utils.config import Config

class PINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Store architecture info
        self.input_dim = cfg.input_dim or 5  # [t, θ1, θ2, ω1, ω2]
        self.output_dim = cfg.output_dim or 4  # [θ̇1, θ̇2, ω̇1, ω̇2]
        
        # Build network
        layers = []
        dims = [self.input_dim] + cfg.hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Add batch norm if specified
            if cfg.use_batch_norm and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # Add activation (except for last layer)
            if i < len(dims) - 2:
                layers.append(self._get_activation(cfg.activation))
                
                # Add dropout if specified
                if cfg.dropout_rate > 0:
                    layers.append(nn.Dropout(cfg.dropout_rate))
        
        # Final activation if specified
        if cfg.final_activation:
            layers.append(self._get_activation(cfg.final_activation))
        
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.Tanh())
    
    def forward(self, x):
        return self.network(x)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cpu'):
        """Load model from checkpoint (class method)."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = Config(**checkpoint['config'])
        model = cls(cfg).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, cfg