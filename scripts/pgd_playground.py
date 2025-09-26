#!/usr/bin/env python3
"""Quick CIFAR-10 PGD playground using ART."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from tools.cifar10_data import MEAN, STD


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Return a ResNet-18 variant tailored for 32x32 CIFAR inputs."""
    model = tv.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
    """Load a state_dict from disk into the provided model."""
    payload = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch - missing keys: {missing}, unexpected keys: {unexpected}"
        )


def collect_batch(loader: DataLoader, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grab a fixed number of samples from a loader."""
    xs, ys = [], []
    for images, targets in loader:
        xs.append(images)
        ys.append(targets)
        if sum(batch.size(0) for batch in xs) >= num_samples:
            break
    x_cat = torch.cat(xs, dim=0)[:num_samples]
    y_cat = torch.cat(ys, dim=0)[:num_samples]
    return x_cat, y_cat


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argmax(axis=1)
    return float((preds == labels).mean())


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = build_cifar_resnet18(num_classes=10)
    load_checkpoint(model, Path(args.checkpoint))
    model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    preprocessing = (
        np.asarray(MEAN, dtype=np.float32),
        np.asarray(STD, dtype=np.float32),
    )

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=preprocessing,
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    transform = tv.transforms.ToTensor()
    dataset = tv.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=args.loader_batch_size, shuffle=False)

    x_tensor, y_tensor = collect_batch(loader, args.num_samples)
    x_clean = x_tensor.numpy()
    y_np = y_tensor.numpy()

    preds_clean = classifier.predict(x_clean, batch_size=args.attack_batch_size)
    clean_acc = accuracy_from_logits(preds_clean, y_np)

    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=args.eps,
        eps_step=args.eps_step,
        max_iter=args.max_iter,
        num_random_init=args.num_random_init,
        targeted=False,
        batch_size=args.attack_batch_size,
    )

    x_adv = attack.generate(x=x_clean)
    preds_adv = classifier.predict(x_adv, batch_size=args.attack_batch_size)
    adv_acc = accuracy_from_logits(preds_adv, y_np)

    linf_radius = np.max(np.abs(x_adv - x_clean), axis=(1, 2, 3))

    print(f"Device: {device}")
    print(f"Samples evaluated: {len(x_clean)}")
    print(f"Clean accuracy: {clean_acc * 100:.2f}%")
    print(f"Adversarial accuracy (PGD): {adv_acc * 100:.2f}%")
    print(
        "L-inf stats (min/mean/max): "
        f"{linf_radius.min():.5f} / {linf_radius.mean():.5f} / {linf_radius.max():.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PGD attack playground using ART")
    parser.add_argument("--checkpoint", default="model_best.pth.tar")
    parser.add_argument("--data-root", default="./.data")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--loader-batch-size", type=int, default=128)
    parser.add_argument("--attack-batch-size", type=int, default=64)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--eps-step", type=float, default=2 / 255)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--num-random-init", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
