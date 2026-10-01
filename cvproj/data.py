"""Data loading: stratified train/val split, in-memory image cache, transforms."""
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class InMemoryImages(Dataset):
    """Decodes every image once (the dataset is tiny) and applies a transform on access."""

    def __init__(self, paths, labels, transform=None):
        self.images = [Image.open(p).convert("L").copy() for p in paths]
        self.labels = list(labels)
        self.paths = [str(p) for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def stratified_split(labels, val_fraction, seed):
    """Returns (train_idx, val_idx) with the same fraction of every class held out."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def build_transforms(cfg, train: bool):
    """Builds the transform pipeline described by the `data` section of a config."""
    size = cfg["img_size"]
    channels = cfg.get("channels", 3)
    aug = cfg.get("augment", "none") if train else "none"
    if channels == 1:
        mean, std = [0.5], [0.5]
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    t = [transforms.Grayscale(num_output_channels=channels)]
    if aug == "none":
        t.append(transforms.Resize((size, size)))
    elif aug == "light":
        # flips + small crops: label-preserving for scenes
        t += [transforms.Resize((size, size)),
              transforms.RandomCrop(size, padding=size // 16, padding_mode="reflect"),
              transforms.RandomHorizontalFlip()]
    elif aug == "strong":
        t += [transforms.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3)),
              transforms.RandomHorizontalFlip(),
              transforms.RandomApply([transforms.ColorJitter(0.4, 0.4)], p=0.8),
              transforms.RandomRotation(10)]
    else:
        raise ValueError(f"unknown augment '{aug}'")
    t += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if train and cfg.get("random_erasing", 0) > 0:
        t.append(transforms.RandomErasing(p=cfg["random_erasing"], scale=(0.02, 0.2)))
    return transforms.Compose(t)


def build_datasets(cfg, root):
    """Returns train/val/test datasets plus class names. Val is a stratified split of train."""
    root = Path(root)
    folder = datasets.ImageFolder(root / "train")
    paths = [p for p, _ in folder.samples]
    labels = [y for _, y in folder.samples]
    tr_idx, va_idx = stratified_split(labels, cfg.get("val_fraction", 0.2), cfg.get("split_seed", 0))
    if cfg.get("use_full_train", False):
        tr_idx = list(range(len(labels)))  # train on train+val (no model selection possible)

    train_tf, eval_tf = build_transforms(cfg, True), build_transforms(cfg, False)
    train_ds = InMemoryImages([paths[i] for i in tr_idx], [labels[i] for i in tr_idx], train_tf)
    val_ds = InMemoryImages([paths[i] for i in va_idx], [labels[i] for i in va_idx], eval_tf)

    test_folder = datasets.ImageFolder(root / "test")
    assert test_folder.classes == folder.classes
    test_ds = InMemoryImages([p for p, _ in test_folder.samples],
                             [y for _, y in test_folder.samples], eval_tf)
    return train_ds, val_ds, test_ds, folder.classes


def make_loader(ds, batch_size, shuffle, num_workers=4, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0, drop_last=drop_last)
