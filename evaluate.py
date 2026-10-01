"""Evaluate a trained checkpoint on the validation split and the labelled test set.

    python evaluate.py --checkpoint checkpoints/final_convnext_tiny.pt --tta
    python evaluate.py --checkpoint checkpoints/final_convnext_tiny.pt --tta --predict-dir data/test2

Prints overall/per-class accuracy, saves a confusion matrix plot and (optionally) writes
predictions for an unlabelled folder of images to a CSV file.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from cvproj.data import InMemoryImages, build_datasets, build_transforms, make_loader
from cvproj.engine import predict
from cvproj.models import build_model
from cvproj.utils import get_device

ROOT = Path(__file__).resolve().parent


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = dict(cfg["model"], pretrained=False)  # weights come from the checkpoint
    model = build_model(model_cfg, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt["classes"]


def plot_confusion(cm, classes, path, title):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--tta", action="store_true", help="average predictions with horizontal flip")
    ap.add_argument("--predict-dir", help="folder of unlabelled images to predict")
    ap.add_argument("--out-dir", default=str(ROOT / "results/eval"))
    args = ap.parse_args()

    device = get_device()
    model, cfg, classes = load_checkpoint(args.checkpoint, device)
    data_cfg = dict(cfg["data"], use_full_train=False)
    _, val_ds, test_ds, ds_classes = build_datasets(data_cfg, args.data_root)
    assert ds_classes == classes
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.checkpoint).stem

    report = {"checkpoint": args.checkpoint, "tta": args.tta}
    for split, ds in [("val", val_ds), ("test", test_ds)]:
        probs, labels = predict(model, make_loader(ds, 64, False, 2), device, tta=args.tta)
        pred = probs.argmax(1).numpy()
        labels = labels.numpy()
        acc = float((pred == labels).mean())
        cm = confusion_matrix(labels, pred, labels=range(len(classes)))
        per_class = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes)}
        report[split] = {"acc": acc, "n": int(len(labels)), "per_class_acc": per_class,
                         "confusion_matrix": cm.tolist()}
        print(f"{split}: accuracy {acc:.4f} ({(pred == labels).sum()}/{len(labels)})")
        if split == "test":
            for c, a in sorted(per_class.items(), key=lambda kv: kv[1]):
                print(f"   {c:<13s} {a:.3f}")
            plot_confusion(cm, classes, out_dir / f"{name}_test_confusion.png",
                           f"{name} - test acc {acc:.3f}")

    if args.predict_dir:
        paths = sorted(Path(args.predict_dir).glob("*.jpg"),
                       key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
        ds = InMemoryImages(paths, [0] * len(paths), build_transforms(data_cfg, False))
        probs, _ = predict(model, make_loader(ds, 64, False, 2), device, tta=args.tta)
        csv_path = out_dir / f"{name}_{Path(args.predict_dir).name}_predictions.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "prediction", "confidence"])
            for p, pr in zip(paths, probs):
                w.writerow([p.name, classes[int(pr.argmax())], f"{float(pr.max()):.4f}"])
        print(f"wrote {len(paths)} predictions to {csv_path}")

    json.dump(report, open(out_dir / f"{name}_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
