# tools/subsets.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import List, Sequence
import torch
from torch.utils.data import Subset, DataLoader
import torchvision as tv
from tools.cifar10_data import cifar10_dataset

def stratified_indices(labels: Sequence[int], per_class: int, seed: int = 2025) -> List[int]:
    rng = np.random.RandomState(seed)
    labels_arr = np.asarray(labels)          # <-- use a new var
    idx_all: List[int] = []
    for c in range(10):
        cls_idx = np.where(labels_arr == c)[0]
        rng.shuffle(cls_idx)
        idx_all.extend(cls_idx[:per_class].tolist())
    idx_all.sort()
    return idx_all

def save_indices_json(indices: Sequence[int], path: str, *, split: str, per_class: int, seed: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset":"CIFAR-10","split":split,"per_class":per_class,"seed":seed,"indices":list(map(int, indices))}
    Path(path).write_text(json.dumps(payload, indent=2))

def load_indices_json(path: str) -> List[int]:
    return list(map(int, json.loads(Path(path).read_text())["indices"]))

def subset_loader(data_root: str, split: str, indices: Sequence[int],
                  batch_size: int = 128, num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    ds = cifar10_dataset(data_root, split)
    sub = Subset(ds, list(indices))
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def labels_from_raw(data_root: str, split: str) -> List[int]:
    raw = tv.datasets.CIFAR10(root=data_root, train=(split == "train"), download=True)
    return list(map(int, raw.targets))