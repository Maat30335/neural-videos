"""Dataset for loading videos and generating coordinate-RGB pairs for training."""

import torch
from torch.utils.data import Dataset, IterableDataset
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


class VideoDataset(IterableDataset):
    """Dataset that loads a video and provides coordinate-RGB training pairs.
    
    Uses an IterableDataset with vectorized batch sampling for maximum efficiency.
    All random sampling is done in batches using NumPy vectorization.
    
    For each sample, returns:
    - coords: (x, y, t) normalized to [-1, 1]
    - rgb: RGB values normalized to [0, 1]
    
    Args:
        video_path: Path to video file
        batch_size: Number of samples per batch (used for internal batching)
        num_batches_per_epoch: Number of batches per epoch (if None, calculated from total pixels)
        frame_step: Sample every Nth frame (for faster training on long videos)
        max_frames: Maximum number of frames to load (None for all)
        device: Device to store video tensor on (None for CPU, 'cuda' for GPU)
    """
    
    def __init__(
        self,
        video_path: Union[str, Path],
        batch_size: int = 65536,
        num_batches_per_epoch: Optional[int] = None,
        frame_step: int = 1,
        max_frames: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.video_path = Path(video_path)
        self.frame_step = frame_step
        self.batch_size = batch_size
        self.device = device
        
        # Load video frames
        self.frames, self.fps = self._load_video(max_frames)
        self.num_frames, self.height, self.width, _ = self.frames.shape
        
        # Total pixels in video
        self.total_pixels = self.num_frames * self.height * self.width
        
        # Number of batches per epoch
        if num_batches_per_epoch is None:
            self.num_batches = max(1, self.total_pixels // batch_size)
        else:
            self.num_batches = num_batches_per_epoch
        
        # Pre-compute normalized coordinate grids as numpy for fast indexing
        self._setup_coordinates()
        
        # Convert frames to GPU tensor if device specified
        self._frames_tensor = None
        if device is not None and device.type == 'cuda':
            self._setup_gpu_data(device)
        
        print(f"Loaded video: {self.width}x{self.height}, {self.num_frames} frames @ {self.fps:.1f} fps")
        print(f"Total pixels: {self.total_pixels:,}")
        print(f"Batches per epoch: {self.num_batches:,} (batch_size={batch_size:,})")
        if self._frames_tensor is not None:
            print(f"Video data stored on: {device}")
    
    def _load_video(self, max_frames: Optional[int]) -> Tuple[np.ndarray, float]:
        """Load video frames from file."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_step == 0:
                # Convert BGR to RGB and normalize to [0, 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32) / 255.0)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from: {self.video_path}")
        
        return np.stack(frames), fps
    
    def _setup_coordinates(self):
        """Pre-compute normalized coordinate grids."""
        # Spatial coordinates [-1, 1]
        self.x_coords = np.linspace(-1, 1, self.width, dtype=np.float32)
        self.y_coords = np.linspace(-1, 1, self.height, dtype=np.float32)
        
        # Temporal coordinates [-1, 1]
        if self.num_frames > 1:
            self.t_coords = np.linspace(-1, 1, self.num_frames, dtype=np.float32)
        else:
            self.t_coords = np.array([0.0], dtype=np.float32)
    
    def _setup_gpu_data(self, device: torch.device):
        """Move video frames to GPU for faster sampling."""
        self._frames_tensor = torch.from_numpy(self.frames).to(device)
        self._x_coords_tensor = torch.from_numpy(self.x_coords).to(device)
        self._y_coords_tensor = torch.from_numpy(self.y_coords).to(device)
        self._t_coords_tensor = torch.from_numpy(self.t_coords).to(device)
    
    def sample_batch_cpu(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of coordinate-RGB pairs efficiently on CPU.
        
        Uses vectorized NumPy operations for maximum efficiency.
        """
        # Vectorized random indices
        t_idx = np.random.randint(0, self.num_frames, size=batch_size)
        y_idx = np.random.randint(0, self.height, size=batch_size)
        x_idx = np.random.randint(0, self.width, size=batch_size)
        
        # Batch gather coordinates (vectorized)
        coords = np.stack([
            self.x_coords[x_idx],
            self.y_coords[y_idx],
            self.t_coords[t_idx]
        ], axis=-1)
        
        # Batch gather RGB (vectorized)
        rgb = self.frames[t_idx, y_idx, x_idx]
        
        return torch.from_numpy(coords), torch.from_numpy(rgb)
    
    def sample_batch_gpu(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of coordinate-RGB pairs directly on GPU.
        
        Uses GPU-resident data for zero CPU-GPU transfer overhead.
        """
        device = self._frames_tensor.device
        
        # Random indices on GPU
        t_idx = torch.randint(0, self.num_frames, (batch_size,), device=device)
        y_idx = torch.randint(0, self.height, (batch_size,), device=device)
        x_idx = torch.randint(0, self.width, (batch_size,), device=device)
        
        # Gather coordinates
        coords = torch.stack([
            self._x_coords_tensor[x_idx],
            self._y_coords_tensor[y_idx],
            self._t_coords_tensor[t_idx]
        ], dim=-1)
        
        # Gather RGB values
        rgb = self._frames_tensor[t_idx, y_idx, x_idx]
        
        return coords, rgb
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch, using GPU if available."""
        if self._frames_tensor is not None:
            return self.sample_batch_gpu(batch_size)
        else:
            return self.sample_batch_cpu(batch_size)
    
    def __iter__(self):
        """Iterate through batches for one epoch."""
        for _ in range(self.num_batches):
            yield self.sample_batch(self.batch_size)
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.num_batches
    
    def get_frame_coords(self, t_idx: int) -> torch.Tensor:
        """Get all coordinates for a specific frame.
        
        Args:
            t_idx: Frame index
            
        Returns:
            Tensor of shape (H*W, 3) with (x, y, t) coordinates
        """
        t = self.t_coords[t_idx]
        
        # Create meshgrid
        xx, yy = np.meshgrid(self.x_coords, self.y_coords)
        
        coords = np.stack([
            xx.flatten(),
            yy.flatten(),
            np.full(self.height * self.width, t)
        ], axis=-1)
        
        return torch.tensor(coords, dtype=torch.float32)
    
    def get_frame_rgb(self, t_idx: int) -> torch.Tensor:
        """Get all RGB values for a specific frame.
        
        Args:
            t_idx: Frame index
            
        Returns:
            Tensor of shape (H*W, 3) with RGB values
        """
        return torch.tensor(
            self.frames[t_idx].reshape(-1, 3),
            dtype=torch.float32
        )
    
    def get_frame_image(self, t_idx: int) -> np.ndarray:
        """Get a frame as an image array.
        
        Args:
            t_idx: Frame index
            
        Returns:
            Array of shape (H, W, 3) with RGB values in [0, 1]
        """
        return self.frames[t_idx]
    
    @property
    def frame_shape(self) -> Tuple[int, int]:
        """Return (height, width) of frames."""
        return (self.height, self.width)


# Backwards compatibility: keep the old Dataset class for num_workers > 0 use case
class VideoDatasetLegacy(Dataset):
    """Legacy Dataset for use with DataLoader num_workers > 0.
    
    Less efficient than IterableDataset but compatible with multi-worker loading.
    """
    
    def __init__(
        self,
        video_path: Union[str, Path],
        num_samples: Optional[int] = None,
        frame_step: int = 1,
        max_frames: Optional[int] = None
    ):
        super().__init__()
        self.video_path = Path(video_path)
        self.frame_step = frame_step
        
        # Load video frames
        self.frames, self.fps = self._load_video(max_frames)
        self.num_frames, self.height, self.width, _ = self.frames.shape
        
        # Total pixels in video
        self.total_pixels = self.num_frames * self.height * self.width
        
        # Number of samples per epoch
        self.num_samples = num_samples if num_samples else self.total_pixels
        
        # Pre-compute normalized coordinate grids
        self._setup_coordinates()
        
        print(f"Loaded video: {self.width}x{self.height}, {self.num_frames} frames @ {self.fps:.1f} fps")
        print(f"Total pixels: {self.total_pixels:,}")
        print(f"Samples per epoch: {self.num_samples:,}")
    
    def _load_video(self, max_frames: Optional[int]) -> Tuple[np.ndarray, float]:
        """Load video frames from file."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32) / 255.0)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from: {self.video_path}")
        
        return np.stack(frames), fps
    
    def _setup_coordinates(self):
        """Pre-compute normalized coordinate grids."""
        self.x_coords = np.linspace(-1, 1, self.width, dtype=np.float32)
        self.y_coords = np.linspace(-1, 1, self.height, dtype=np.float32)
        
        if self.num_frames > 1:
            self.t_coords = np.linspace(-1, 1, self.num_frames, dtype=np.float32)
        else:
            self.t_coords = np.array([0.0], dtype=np.float32)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random coordinate-RGB pair."""
        t_idx = np.random.randint(0, self.num_frames)
        y_idx = np.random.randint(0, self.height)
        x_idx = np.random.randint(0, self.width)
        
        coords = torch.tensor([
            self.x_coords[x_idx],
            self.y_coords[y_idx],
            self.t_coords[t_idx]
        ], dtype=torch.float32)
        
        rgb = torch.tensor(
            self.frames[t_idx, y_idx, x_idx],
            dtype=torch.float32
        )
        
        return coords, rgb
    
    def get_frame_coords(self, t_idx: int) -> torch.Tensor:
        t = self.t_coords[t_idx]
        xx, yy = np.meshgrid(self.x_coords, self.y_coords)
        coords = np.stack([
            xx.flatten(),
            yy.flatten(),
            np.full(self.height * self.width, t)
        ], axis=-1)
        return torch.tensor(coords, dtype=torch.float32)
    
    def get_frame_rgb(self, t_idx: int) -> torch.Tensor:
        return torch.tensor(
            self.frames[t_idx].reshape(-1, 3),
            dtype=torch.float32
        )
    
    def get_frame_image(self, t_idx: int) -> np.ndarray:
        return self.frames[t_idx]
    
    @property
    def frame_shape(self) -> Tuple[int, int]:
        return (self.height, self.width)
