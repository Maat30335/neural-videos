"""Utility functions for visualization and video I/O."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List
import imageio


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy image array.
    
    Args:
        tensor: Tensor of shape (H, W, 3) or (3, H, W) in [0, 1]
    
    Returns:
        Numpy array of shape (H, W, 3) with uint8 values in [0, 255]
    """
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    
    img = tensor.detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def save_image(tensor: torch.Tensor, path: Union[str, Path]):
    """Save a tensor as an image file."""
    img = tensor_to_image(tensor)
    imageio.imwrite(str(path), img)


def save_video(
    frames: List[torch.Tensor],
    path: Union[str, Path],
    fps: float = 30.0
):
    """Save a list of frame tensors as a video.
    
    Args:
        frames: List of tensors, each (H, W, 3) in [0, 1]
        path: Output video path
        fps: Frames per second
    """
    path = Path(path)
    
    # Convert frames to numpy
    np_frames = [tensor_to_image(f) for f in frames]
    
    # Use imageio to write video
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in np_frames:
        writer.append_data(frame)
    writer.close()
    
    print(f"Saved video: {path} ({len(frames)} frames @ {fps} fps)")


def visualize_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    title: str = "Original vs Reconstructed",
    save_path: Optional[Union[str, Path]] = None
):
    """Show side-by-side comparison of original and reconstructed frames.
    
    Args:
        original: Original frame (H, W, 3)
        reconstructed: Reconstructed frame (H, W, 3)
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    orig_img = tensor_to_image(original)
    recon_img = tensor_to_image(reconstructed)
    
    axes[0].imshow(orig_img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(recon_img)
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_training_progress(
    losses: List[float],
    psnrs: List[float],
    save_path: Optional[Union[str, Path]] = None
):
    """Plot training loss and PSNR over time.
    
    Args:
        losses: List of loss values
        psnrs: List of PSNR values
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(losses)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_yscale('log')
    axes[0].grid(True)
    
    axes[1].plot(psnrs)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Reconstruction Quality")
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_interpolation_grid(
    model: torch.nn.Module,
    t_values: List[float],
    height: int,
    width: int,
    device: torch.device
) -> List[torch.Tensor]:
    """Generate frames at multiple time values.
    
    Args:
        model: Trained SIREN model
        t_values: List of time values in [-1, 1]
        height: Frame height
        width: Frame width
        device: Device to use
    
    Returns:
        List of frame tensors
    """
    model.eval()
    frames = []
    
    for t in t_values:
        frame = model.generate_frame(t, height, width, device)
        frames.append(frame)
    
    return frames
