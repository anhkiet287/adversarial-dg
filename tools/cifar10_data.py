# tools/cifar10_data.py
from __future__ import annotations
import torchvision as tv

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

def cifar10_dataset(data_root:str="./.data", split:str="train"):
    assert split in {"train", "test"}
    tfm = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MEAN, STD),
    ])
    return tv.datasets.CIFAR10(
        root=data_root, train=(split=="train"),
        download=True, transform=tfm
    )