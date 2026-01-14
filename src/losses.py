"""Loss functions for training implicit video representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return F.mse_loss(pred, target)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio.
    
    Higher is better. Typical values:
    - < 20 dB: Poor quality
    - 20-30 dB: Acceptable
    - 30-40 dB: Good quality
    - > 40 dB: Excellent quality
    
    Args:
        pred: Predicted values in [0, max_val]
        target: Target values in [0, max_val]
        max_val: Maximum possible value (1.0 for normalized images)
    
    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(max_val ** 2 / mse)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features.
    
    Compares high-level features rather than pixel values, which can
    produce more visually pleasing results.
    
    Note: Requires torchvision and downloads pretrained VGG weights on first use.
    """
    
    def __init__(self, layers: Optional[list] = None):
        super().__init__()
        
        try:
            from torchvision import models
            from torchvision.models import VGG16_Weights
        except ImportError:
            raise ImportError("PerceptualLoss requires torchvision")
        
        # Load pretrained VGG16
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Default layers for feature extraction
        if layers is None:
            layers = [3, 8, 15, 22]  # After each maxpool
        
        self.layers = layers
        self.max_layer = max(layers) + 1
        
        # Extract feature layers
        self.features = nn.Sequential(*list(vgg.children())[:self.max_layer])
        
        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            pred: Predicted images (B, C, H, W) in [0, 1]
            target: Target images (B, C, H, W) in [0, 1]
        
        Returns:
            Perceptual loss (scalar)
        """
        # Normalize
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        x = pred
        y = target
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            
            if i in self.layers:
                loss = loss + F.mse_loss(x, y)
        
        return loss / len(self.layers)


class CombinedLoss(nn.Module):
    """Combined MSE and perceptual loss.
    
    Args:
        perceptual_weight: Weight for perceptual loss (0 to disable)
    """
    
    def __init__(self, perceptual_weight: float = 0.0):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        
        if perceptual_weight > 0:
            self.perceptual = PerceptualLoss()
        else:
            self.perceptual = None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_image: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute combined loss.
        
        For perceptual loss, images must be provided in (B, C, H, W) format.
        
        Args:
            pred: Predicted pixel values (flattened)
            target: Target pixel values (flattened)
            pred_image: Predicted images for perceptual loss
            target_image: Target images for perceptual loss
        """
        loss = mse_loss(pred, target)
        
        if self.perceptual is not None and pred_image is not None:
            loss = loss + self.perceptual_weight * self.perceptual(
                pred_image, target_image
            )
        
        return loss
