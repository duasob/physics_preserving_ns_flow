from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_MINDFLOW_ROOT = Path(os.environ.get("MINDFLOW_DATA_ROOT", "data/mindflow"))


@dataclass(frozen=True)
class MindFlowNSLoadConfig:
    root: Path = DEFAULT_MINDFLOW_ROOT
    train_split: str = "train"
    test_split: str = "test"
    batch_size: int = 8
    test_batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    dtype: torch.dtype = torch.float32
    mmap: bool = True


def _split_paths(root: Path, split: str) -> Tuple[Path, Path]:
    split_dir = root / split
    return split_dir / "inputs.npy", split_dir / "label.npy"


def _load_arrays(inputs_path: Path, labels_path: Path, mmap: bool) -> Tuple[np.ndarray, np.ndarray]:
    if not inputs_path.exists():
        raise FileNotFoundError(f"Missing inputs at {inputs_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels at {labels_path}")
    mmap_mode = "r" if mmap else None
    inputs = np.load(inputs_path, mmap_mode=mmap_mode)
    labels = np.load(labels_path, mmap_mode=mmap_mode)
    if len(inputs) != len(labels):
        raise ValueError(
            f"Inputs and labels size mismatch: {len(inputs)} vs {len(labels)}"
        )
    return inputs, labels


class MindFlowNSDataset(Dataset):
    def __init__(
        self,
        root: Path | str = DEFAULT_MINDFLOW_ROOT,
        split: str = "train",
        *,
        dtype: torch.dtype = torch.float32,
        mmap: bool = True,
    ) -> None:
        root = Path(root)
        inputs_path, labels_path = _split_paths(root, split)
        inputs, labels = _load_arrays(inputs_path, labels_path, mmap=mmap)
        self._inputs = inputs
        self._labels = labels
        self._dtype = dtype

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.as_tensor(self._inputs[idx], dtype=self._dtype)
        y = torch.as_tensor(self._labels[idx], dtype=self._dtype)
        return {"x": x, "y": y}


def create_mindflow_loaders(
    config: MindFlowNSLoadConfig,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MindFlowNSDataset(
        root=config.root,
        split=config.train_split,
        dtype=config.dtype,
        mmap=config.mmap,
    )
    test_ds = MindFlowNSDataset(
        root=config.root,
        split=config.test_split,
        dtype=config.dtype,
        mmap=config.mmap,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader, test_loader
