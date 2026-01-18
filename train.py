#!/usr/bin/env python3
"""Training script for implicit neural video representation.

Usage:
    python train.py --video path/to/video.mp4 --epochs 1000

For a quick test:
    python train.py --video path/to/video.mp4 --epochs 10 --test_run
"""

import argparse
import json
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import time

from src.model import SIREN, count_parameters
from src.dataset import VideoDataset
from src.losses import mse_loss, psnr
from src.utils import visualize_comparison, visualize_training_progress, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Train implicit neural video representation")
    
    # Data
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to load")
    parser.add_argument("--frame_step", type=int, default=1, help="Sample every Nth frame")
    
    # Model
    parser.add_argument("--hidden_features", type=int, default=256, help="Hidden layer width")
    parser.add_argument("--hidden_layers", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--omega_0", type=float, default=30.0, help="SIREN frequency parameter")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=65536, help="Batch size (pixels per batch)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batches_per_epoch", type=int, default=None, 
                        help="Batches per epoch (default: total_pixels / batch_size)")
    
    # Checkpointing
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N epochs")
    parser.add_argument("--vis_every", type=int, default=50, help="Visualize every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Performance
    parser.add_argument("--gpu_data", action="store_true", 
                        help="Store video data on GPU (faster but uses more VRAM)")
    
    # Misc
    parser.add_argument("--test_run", action="store_true", help="Quick test run (1 epoch, few samples)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    
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


def train_epoch(model, dataset, optimizer, device, use_gpu_data: bool, epoch: int):
    """Train for one epoch using direct batch sampling.
    
    This bypasses DataLoader entirely for maximum efficiency when
    data is already in memory (or on GPU).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar with batch-level updates
    pbar = tqdm(
        dataset,
        desc=f"Epoch {epoch + 1:4d}",
        total=len(dataset),
        unit="batch",
        leave=False,
        dynamic_ncols=True
    )
    
    for coords, rgb in pbar:
        # Transfer to device if not already there (for CPU sampling path)
        if not use_gpu_data:
            coords = coords.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        
        pred = model(coords)
        loss = mse_loss(pred, rgb)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar with current loss
        pbar.set_postfix(loss=f"{loss.item():.5f}", refresh=False)
    
    pbar.close()
    return total_loss / num_batches


def evaluate(model, dataset, device):
    """Evaluate model on full frames and compute PSNR."""
    model.eval()
    
    # Evaluate on first, middle, and last frames
    frame_indices = [0, len(dataset.t_coords) // 2, len(dataset.t_coords) - 1]
    frame_indices = list(set(frame_indices))  # Remove duplicates for short videos
    
    psnrs = []
    
    with torch.no_grad():
        for t_idx in frame_indices:
            # Get ground truth
            coords = dataset.get_frame_coords(t_idx).to(device, non_blocking=True)
            target = dataset.get_frame_rgb(t_idx).to(device, non_blocking=True)
            
            # Predict
            pred = model(coords)
            
            # Compute PSNR
            frame_psnr = psnr(pred, target).item()
            psnrs.append(frame_psnr)
    
    return sum(psnrs) / len(psnrs)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)


def main():
    args = parse_args()
    
    # Quick test mode
    if args.test_run:
        args.epochs = 1
        args.batches_per_epoch = 5
        args.save_every = 1
        args.vis_every = 1
    
    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Determine if we should use GPU data storage
    use_gpu_data = args.gpu_data and device.type == "cuda"
    if args.gpu_data and device.type != "cuda":
        print("Warning: --gpu_data only works with CUDA devices, ignoring flag")
        use_gpu_data = False
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset with new optimized VideoDataset
    print(f"\nLoading video: {args.video}")
    dataset = VideoDataset(
        args.video,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.batches_per_epoch,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        device=device if use_gpu_data else None
    )
    
    # Create model
    model = SIREN(
        in_features=3,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=3,
        first_omega_0=args.omega_0,
        hidden_omega_0=args.omega_0
    ).to(device)
    
    print(f"\nModel: {count_parameters(model):,} parameters")
    print(f"  Hidden: {args.hidden_features} x {args.hidden_layers} layers")
    print(f"  Omega_0: {args.omega_0}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint
    start_epoch = 0
    metrics = {'losses': [], 'psnrs': []}
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        metrics = checkpoint.get('metrics', metrics)
    
    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['frame_shape'] = dataset.frame_shape
    config['num_frames'] = dataset.num_frames
    config['gpu_data'] = use_gpu_data
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size:,} pixels")
    print(f"Batches per epoch: {len(dataset):,}")
    print(f"GPU data storage: {'enabled' if use_gpu_data else 'disabled'}")
    print("-" * 50)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        loss = train_epoch(model, dataset, optimizer, device, use_gpu_data, epoch)
        metrics['losses'].append(loss)
        
        # Evaluate periodically
        if (epoch + 1) % args.vis_every == 0 or epoch == 0:
            avg_psnr = evaluate(model, dataset, device)
            metrics['psnrs'].append(avg_psnr)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1:4d} | Loss: {loss:.6f} | PSNR: {avg_psnr:.2f} dB | Time: {elapsed:.1f}s")
            
            # Save sample reconstruction
            with torch.no_grad():
                frame_idx = 0
                reconstructed = model.generate_frame(
                    dataset.t_coords[frame_idx],
                    dataset.height,
                    dataset.width,
                    device
                )
                original = torch.from_numpy(dataset.get_frame_image(frame_idx))
                
                save_image(reconstructed, output_dir / f"recon_epoch_{epoch + 1:04d}.png")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            )
    
    # Final save
    print("\nSaving final model...")
    save_checkpoint(model, optimizer, args.epochs - 1, metrics, output_dir / "model_final.pt")
    
    # Save training curves
    if len(metrics['psnrs']) > 1:
        visualize_training_progress(
            metrics['losses'], 
            metrics['psnrs'],
            output_dir / "training_progress.png"
        )
    
    # Final evaluation with comparison
    print("\nFinal evaluation...")
    with torch.no_grad():
        for t_idx in [0, dataset.num_frames // 2, dataset.num_frames - 1]:
            if t_idx >= dataset.num_frames:
                continue
                
            reconstructed = model.generate_frame(
                dataset.t_coords[t_idx],
                dataset.height,
                dataset.width,
                device
            )
            original = torch.from_numpy(dataset.get_frame_image(t_idx))
            
            frame_psnr = psnr(
                reconstructed.flatten().cpu(),
                original.flatten()
            ).item()
            
            print(f"  Frame {t_idx}: PSNR = {frame_psnr:.2f} dB")
            
            visualize_comparison(
                original, reconstructed.cpu(),
                f"Frame {t_idx} - PSNR: {frame_psnr:.2f} dB",
                output_dir / f"comparison_frame_{t_idx:04d}.png"
            )
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time / 60:.1f} minutes")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
