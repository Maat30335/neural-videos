# Implicit Neural Videos

A SIREN-based neural network that learns to represent video as a continuous function, enabling **frame interpolation** and **super-resolution** after training.

## How It Works

The model learns a mapping from coordinates to colors:

```
(x, y, t) → RGB
```

Where:
- `x, y` are spatial coordinates (normalized to [-1, 1])
- `t` is the temporal coordinate (frame position, normalized to [-1, 1])
- The output is an RGB color value

After training, you can query the network at **any coordinate**, including:
- Times between original frames → **frame interpolation**
- Higher spatial resolution → **super-resolution**

## Installation

```bash
pip install -r requirements.txt
```

Requirements: PyTorch 2.0+, OpenCV, NumPy, tqdm, matplotlib, imageio

## Quick Start

### 1. Train on a video

```bash
python train.py --video your_video.mp4 --epochs 1000
```

This will:
- Load the video and create coordinate-RGB training pairs
- Train a SIREN network to fit the video
- Save checkpoints and sample reconstructions to `outputs/`

### 2. Generate new frames

**Reconstruct original frames:**
```bash
python inference.py --checkpoint outputs/model_final.pt --mode reconstruct
```

**Frame interpolation (2x temporal resolution):**
```bash
python inference.py --checkpoint outputs/model_final.pt --mode interpolate --temporal_scale 2
```

**Super-resolution (2x spatial resolution):**
```bash
python inference.py --checkpoint outputs/model_final.pt --mode superres --spatial_scale 2
```

**Custom resolution:**
```bash
python inference.py --checkpoint outputs/model_final.pt --mode custom \
    --width 1920 --height 1080 --num_frames 120
```

## Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video |
| `--epochs` | 1000 | Number of training epochs |
| `--hidden_features` | 256 | Width of hidden layers |
| `--hidden_layers` | 5 | Number of hidden layers |
| `--omega_0` | 30.0 | SIREN frequency parameter |
| `--batch_size` | 65536 | Pixels per batch |
| `--lr` | 1e-4 | Learning rate |
| `--max_frames` | None | Limit frames to load |
| `--frame_step` | 1 | Sample every Nth frame |

### Tips

- **Short videos first**: Start with 2-5 second clips for faster experimentation
- **Model size**: Increase `hidden_features` and `hidden_layers` for longer/higher-res videos
- **Memory**: Reduce `batch_size` if you run out of GPU memory
- **Quality target**: Aim for PSNR > 30 dB for good reconstruction quality

## Architecture

The model uses **SIREN** (Sinusoidal Representation Networks):

```
Input (x, y, t) → [SineLayer × N] → Linear → Sigmoid → RGB
```

Key features:
- **Sine activations** naturally capture high-frequency details
- **Special initialization** maintains signal distribution through layers
- **No positional encoding needed** (unlike vanilla MLPs or NeRFs)

## Output Structure

```
outputs/
├── config.json              # Training configuration
├── model_final.pt           # Final model checkpoint
├── checkpoint_epoch_*.pt    # Periodic checkpoints
├── recon_epoch_*.png        # Sample reconstructions during training
├── training_progress.png    # Loss and PSNR curves
├── comparison_frame_*.png   # Final frame comparisons
└── generated/               # Inference outputs
    ├── frame_*.png          # Individual frames
    └── *.mp4                # Generated videos
```

## License

MIT
