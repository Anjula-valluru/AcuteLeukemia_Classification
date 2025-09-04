from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def _tfms(image_size: int, aug: bool = True):
    if aug:
        train_tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tfms, eval_tfms

def infer_num_classes(train_dir: str) -> int:
    # Use ImageFolder to read class subdirs (transform doesn't matter for counting)
    dummy_tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    ds = datasets.ImageFolder(Path(train_dir), transform=dummy_tfms)
    return len(ds.classes)

def build_loaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str],
    image_size: int,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    train_tfms, eval_tfms = _tfms(image_size=image_size, aug=True)
    train_ds = datasets.ImageFolder(Path(train_dir), transform=train_tfms)
    val_ds   = datasets.ImageFolder(Path(val_dir),   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(Path(test_dir),  transform=eval_tfms) if test_dir else None

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True)
    va_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True) if test_ds else None
    return tr_loader, va_loader, te_loader
