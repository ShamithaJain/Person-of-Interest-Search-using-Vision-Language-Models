#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#  -------------------------------------------------------------------------------------------------
"""Load and prepare face image corpus for Person of Interest retrieval."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def get_data_dir() -> Path:
    """Project data directory (create if missing)."""
    base = Path(__file__).resolve().parent.parent.parent
    data_dir = base / "data" / "faces"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def list_image_paths(data_dir: Path | None = None) -> List[Path]:
    """List all image paths under data/faces (recursive)."""
    data_dir = data_dir or get_data_dir()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
    paths = []
    # Recursive file search through all subfolders.
    for p in data_dir.rglob("*"):
        # p.is_file() To ignore folders and only process actual files.
        if p.suffix.lower() in exts and p.is_file():
            paths.append(p)
    return sorted(paths)


def load_image(path: Path, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load image as RGB numpy array (H, W, 3), resized and uint8."""
    # Different images may be grayscale or RGBA. Models usually need fixed 3-channel RGB input.
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.Resampling.BILINEAR)
    return np.array(img)

# Deep learning models need consistent input size.

# Many CNNs like ResNet or VGG use 224×224.
def load_image_batch(paths: List[Path], size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load multiple images; returns (N, H, W, 3) uint8."""
    return np.array([load_image(p, size) for p in paths])
