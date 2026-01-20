#!/usr/bin/env python3
"""Inference script for generating frames from trained model.

Supports:
- Reconstructing original frames
- Frame interpolation (temporal super-resolution)
- Spatial super-resolution
- Generating videos at arbitrary resolution/framerate

Usage:
    # Reconstruct at original resolution
    python inference.py --checkpoint outputs/model_final.pt --mode reconstruct
    
    # Frame interpolation (2x temporal resolution)
    python inference.py --checkpoint outputs/model_final.pt --mode interpolate --temporal_scale 2
    
    # Super-resolution (2x spatial resolution)
    python inference.py --checkpoint outputs/model_final.pt --mode superres --spatial_scale 2
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.model import SIREN
from src.utils import save_image, save_video


def parse_args():
    parser = argparse.ArgumentParser(description="Generate frames from trained model")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to config.json (default: same dir as checkpoint)")
    
    # Mode
    parser.add_argument("--mode", type=str, default="reconstruct",
                        choices=["reconstruct", "interpolate", "superres", "custom"],
                        help="Generation mode")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: checkpoint dir)")
    parser.add_argument("--output_format", type=str, default="video",
                        choices=["video", "frames", "both"],
                        help="Output format")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output video FPS (default: from training)")
    
    # Resolution
    parser.add_argument("--spatial_scale", type=float, default=1.0,
                        help="Spatial resolution multiplier")
    parser.add_argument("--temporal_scale", type=float, default=1.0,
                        help="Temporal resolution multiplier (frames)")
    parser.add_argument("--width", type=int, default=None, help="Output width (overrides scale)")
    parser.add_argument("--height", type=int, default=None, help="Output height (overrides scale)")
    parser.add_argument("--num_frames", type=int, default=None, 
                        help="Number of output frames (overrides scale)")
    
    # Custom mode
    parser.add_argument("--t_start", type=float, default=-1.0, help="Start time for custom mode")
    parser.add_argument("--t_end", type=float, default=1.0, help="End time for custom mode")
    
    # Misc
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--batch_pixels", type=int, default=262144,
                        help="Pixels per batch (for memory management)")
    
    return parser.parse_args()


def get_device(device_arg: str = None) -> torch.device:
    """Get the best available device."""
    if device_arg:
        return torch.device(device_arg)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> SIREN:
    """Load model from checkpoint."""
    model = SIREN(
        in_features=3,
        hidden_features=config['hidden_features'],
        hidden_layers=config['hidden_layers'],
        out_features=3,
        first_omega_0=config['omega_0'],
        hidden_omega_0=config['omega_0']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def generate_frame_batched(
    model: SIREN,
    t: float,
    height: int,
    width: int,
    device: torch.device,
    batch_pixels: int = 262144
) -> torch.Tensor:
    """Generate a frame with batched inference for memory efficiency.
    
    This is needed for high-resolution outputs that don't fit in GPU memory.
    """
    # Create coordinate grid
    y_coords = torch.linspace(-1, 1, height)
    x_coords = torch.linspace(-1, 1, width)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Flatten and add time coordinate
    coords = torch.stack([
        x_grid.flatten(),
        y_grid.flatten(),
        torch.full((height * width,), t)
    ], dim=-1)
    
    # Process in batches
    num_pixels = height * width
    rgb_values = []
    
    for i in range(0, num_pixels, batch_pixels):
        batch_coords = coords[i:i + batch_pixels].to(device)
        with torch.inference_mode():
            batch_rgb = model(batch_coords)
        rgb_values.append(batch_rgb.cpu())
    
    # Combine and reshape
    rgb = torch.cat(rgb_values, dim=0)
    return rgb.reshape(height, width, 3)


def main():
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Enable TF32 for Ampere GPUs (4090, A100, etc.) - faster matrix ops
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    checkpoint_path = Path(args.checkpoint)
    
    # Load config
    config_path = args.config or (checkpoint_path.parent / "config.json")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(str(checkpoint_path), config, device)
    
    # Determine output resolution
    orig_height, orig_width = config['frame_shape']
    orig_frames = config['num_frames']
    
    if args.width and args.height:
        out_width, out_height = args.width, args.height
    else:
        out_width = int(orig_width * args.spatial_scale)
        out_height = int(orig_height * args.spatial_scale)
    
    if args.num_frames:
        out_frames = args.num_frames
    else:
        out_frames = int(orig_frames * args.temporal_scale)
    
    # Determine time values
    if args.mode == "reconstruct":
        # Use original frame times
        t_values = np.linspace(-1, 1, orig_frames).tolist()
        out_frames = orig_frames
    elif args.mode in ["interpolate", "superres", "custom"]:
        t_values = np.linspace(args.t_start, args.t_end, out_frames).tolist()
    
    # Output settings
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fps = args.fps or config.get('fps', 30.0)
    if args.mode == "interpolate":
        fps = fps * args.temporal_scale
    
    print(f"\nGeneration settings:")
    print(f"  Mode: {args.mode}")
    print(f"  Original: {orig_width}x{orig_height}, {orig_frames} frames")
    print(f"  Output: {out_width}x{out_height}, {out_frames} frames @ {fps:.1f} fps")
    print(f"  Spatial scale: {out_width/orig_width:.2f}x")
    print(f"  Temporal scale: {out_frames/orig_frames:.2f}x")
    
    # Generate frames
    print(f"\nGenerating {out_frames} frames...")
    frames = []
    
    for i, t in enumerate(tqdm(t_values, desc="Generating")):
        frame = generate_frame_batched(
            model, t, out_height, out_width, device, args.batch_pixels
        )
        frames.append(frame)
        
        # Save individual frames if requested
        if args.output_format in ["frames", "both"]:
            save_image(frame, output_dir / f"frame_{i:04d}.png")
    
    # Save video if requested
    if args.output_format in ["video", "both"]:
        video_name = f"{args.mode}_{out_width}x{out_height}_{out_frames}frames.mp4"
        save_video(frames, output_dir / video_name, fps)
    
    print(f"\nOutput saved to: {output_dir}")
    
    # Print summary
    if args.mode == "interpolate":
        print(f"\n✓ Generated {out_frames - orig_frames} interpolated frames")
    elif args.mode == "superres":
        print(f"\n✓ Upscaled from {orig_width}x{orig_height} to {out_width}x{out_height}")


if __name__ == "__main__":
    main()
