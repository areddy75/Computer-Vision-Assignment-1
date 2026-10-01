"""Train a scene classifier from a YAML config.

    python train.py --config configs/best.yaml
    python train.py --config configs/resnet18_pretrained.yaml --set train.seed=1 --tag seed1

Writes checkpoints/<run>.pt, results/runs/<run>.json and appends a row to results/experiments.csv.
The test set is NOT touched here; use evaluate.py for the final test evaluation.
"""
import argparse
import functools
import csv
import json
import time
from pathlib import Path

import torch
import yaml

from cvproj.data import build_datasets, make_loader
from cvproj.engine import evaluate, train
from cvproj.models import build_model, head_parameters
from cvproj.utils import count_params, get_device, set_seed

ROOT = Path(__file__).resolve().parent


def parse_value(v):
    return yaml.safe_load(v)


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        key, val = item.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = parse_value(val)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", help="override config entries, e.g. train.epochs=5")
    ap.add_argument("--tag", default="", help="suffix for the run name")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--no-save", action="store_true", help="do not write a checkpoint")
    args = ap.parse_args()

    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.set)
    run = cfg["name"] + (f"_{args.tag}" if args.tag else "")
    seed = cfg["train"].get("seed", 0)
    set_seed(seed)
    device = get_device()

    train_ds, val_ds, _, classes = build_datasets(cfg["data"], args.data_root)
    bs, nw = cfg["train"].get("batch_size", 32), cfg["data"].get("num_workers", 4)
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, generator=g,
        persistent_workers=nw > 0, drop_last=True)
    val_loader = make_loader(val_ds, 64, False, nw)

    # NB: channels_last was 2-3x slower on MPS (scripts/bench.py), so the default layout is used.
    model = build_model(cfg["model"], len(classes))
    n_params = count_params(model)
    print(f"[{run}] device={device} train={len(train_ds)} val={len(val_ds)} params={n_params:,}")
    head = None if cfg["model"]["name"] in ("tnet", "scenecnn") else head_parameters(model, cfg["model"]["name"])

    t0 = time.time()
    model, hist, best, final = train(model, train_loader, val_loader, cfg["train"], device,
                                     len(classes), head_params=head,
                                     log=functools.partial(print, flush=True))
    train_time = time.time() - t0
    _, val_acc_tta = evaluate(model, val_loader, device, tta=True)

    out = {"run": run, "config": cfg, "classes": classes, "params": n_params,
           "best": best, "final_epoch": final, "val_acc_tta": val_acc_tta,
           "train_time_s": train_time, "history": hist,
           "torch": torch.__version__, "device": str(device)}
    (ROOT / "results/runs").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / f"results/runs/{run}.json", "w"), indent=1)
    if not args.no_save:
        (ROOT / "checkpoints").mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "config": cfg, "classes": classes,
                    "best": best}, ROOT / f"checkpoints/{run}.pt")

    row = {"run": run, "model": cfg["model"]["name"], "pretrained": cfg["model"].get("pretrained", False),
           "img_size": cfg["data"]["img_size"], "augment": cfg["data"].get("augment", "none"),
           "epochs": cfg["train"]["epochs"], "seed": seed, "params": n_params,
           "best_val_acc": round(best["val_acc"], 4), "best_epoch": best["epoch"],
           "final_val_acc": round(final["val_acc"], 4), "val_acc_tta": round(val_acc_tta, 4),
           "final_train_acc": round(hist["train_acc"][-1], 4), "train_time_min": round(train_time / 60, 1)}
    csv_path = ROOT / "results/experiments.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{run}] best val acc {best['val_acc']:.4f} @ epoch {best['epoch']} | "
          f"final {final['val_acc']:.4f} | TTA {val_acc_tta:.4f} | {train_time / 60:.1f} min")


if __name__ == "__main__":
    main()
