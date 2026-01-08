"""SIREN (Sinusoidal Representation Network) for implicit video representation.

SIREN uses periodic activation functions (sine) which allows it to naturally
represent signals with fine details without requiring positional encoding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SineLayer(nn.Module):
    """Linear layer followed by sine activation with proper initialization.
    
    The key insight of SIREN is that using sine activations requires special
    initialization to maintain the distribution of activations through depth.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias
        is_first: Whether this is the first layer (uses different init)
        omega_0: Frequency multiplier for the sine activation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights according to SIREN paper.
        
        First layer: uniform(-1/in, 1/in)
        Hidden layers: uniform(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0)
        """
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            
            self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN network for mapping coordinates to RGB values.
    
    Maps (x, y, t) coordinates to RGB pixel values, where:
    - x, y are spatial coordinates normalized to [-1, 1]
    - t is the temporal coordinate (frame) normalized to [-1, 1]
    
    Args:
        in_features: Input dimension (3 for x, y, t)
        hidden_features: Width of hidden layers
        hidden_layers: Number of hidden layers
        out_features: Output dimension (3 for RGB)
        first_omega_0: Frequency for the first layer
        hidden_omega_0: Frequency for hidden layers
    """
    
    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        out_features: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        
        # Build network
        layers = []
        
        # First layer
        layers.append(SineLayer(
            in_features, hidden_features,
            is_first=True, omega_0=first_omega_0
        ))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(SineLayer(
                hidden_features, hidden_features,
                is_first=False, omega_0=hidden_omega_0
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Final linear layer (no activation) to output RGB
        self.final_layer = nn.Linear(hidden_features, out_features)
        
        # Initialize final layer
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_features) / hidden_omega_0
            self.final_layer.weight.uniform_(-bound, bound)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            coords: Tensor of shape (batch, 3) with (x, y, t) coordinates
                    normalized to [-1, 1]
        
        Returns:
            Tensor of shape (batch, 3) with RGB values in [0, 1]
        """
        x = self.network(coords)
        x = self.final_layer(x)
        # Sigmoid to map outputs to [0, 1] for RGB
        return torch.sigmoid(x)
    
    def generate_frame(
        self,
        t: float,
        height: int,
        width: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate a single frame at time t.
        
        Args:
            t: Time coordinate in [-1, 1]
            height: Output frame height
            width: Output frame width
            device: Device to generate on
        
        Returns:
            Tensor of shape (height, width, 3) with RGB values in [0, 1]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten and add time coordinate
        coords = torch.stack([
            x_grid.flatten(),
            y_grid.flatten(),
            torch.full((height * width,), t, device=device)
        ], dim=-1)
        
        # Generate RGB values
        with torch.no_grad():
            rgb = self(coords)
        
        # Reshape to image
        return rgb.reshape(height, width, 3)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
