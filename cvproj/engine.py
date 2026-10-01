"""Training / evaluation loops."""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_optimizer(model, cfg, head_params=None):
    """AdamW/SGD/Adam with weight decay applied only to conv/linear weights (not BN/bias).

    If `head_params` is given and `backbone_lr_mult` != 1, the pretrained backbone gets a
    smaller learning rate than the freshly initialised classifier head.
    """
    lr, wd = cfg["lr"], cfg.get("weight_decay", 0.0)
    mult = cfg.get("backbone_lr_mult", 1.0)
    head_ids = set(id(p) for p in (head_params or []))
    groups = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        decay = p.ndim > 1
        key = (is_head, decay)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": wd if decay else 0.0,
                           "lr": lr if (is_head or not head_ids) else lr * mult}
        groups[key]["params"].append(p)
    param_groups = list(groups.values())
    for g in param_groups:
        g["base_lr"] = g["lr"]
    name = cfg.get("name", "adamw")
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(name)


def lr_factor(step, total_steps, warmup_steps, schedule):
    if schedule == "constant":
        return 1.0
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    raise ValueError(schedule)


def mix_batch(images, labels, num_classes, alpha_mixup, alpha_cutmix, rng):
    """Applies MixUp or CutMix (50/50 when both are enabled). Returns soft targets."""
    targets = F.one_hot(labels, num_classes).float()
    use_cutmix = alpha_cutmix > 0 and (alpha_mixup <= 0 or rng.rand() < 0.5)
    alpha = alpha_cutmix if use_cutmix else alpha_mixup
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    if use_cutmix:
        H, W = images.shape[-2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = rng.randint(H), rng.randint(W)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        images = lam * images + (1 - lam) * images[perm]
    return images, lam * targets + (1 - lam) * targets[perm]


@torch.inference_mode()
def predict(model, loader, device, tta=False):
    """Returns (softmax probabilities [N, C], labels [N]). TTA = average with horizontal flip."""
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        images = images.to(device)
        p = model(images).float().softmax(1)
        if tta:
            p = 0.5 * (p + model(torch.flip(images, dims=[3])).float().softmax(1))
        probs.append(p.cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def evaluate(model, loader, device, tta=False):
    probs, labels = predict(model, loader, device, tta)
    loss = F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels).item()
    acc = (probs.argmax(1) == labels).float().mean().item()
    return loss, acc


def train(model, train_loader, val_loader, cfg, device, num_classes, head_params=None, log=print):
    """Trains `model`; keeps the weights of the epoch with the best validation accuracy.

    cfg keys: epochs, optimizer{...}, schedule, warmup_epochs, label_smoothing,
    mixup_alpha, cutmix_alpha, mix_prob, grad_clip.
    """
    model.to(device)
    opt = build_optimizer(model, cfg["optimizer"], head_params)
    epochs = cfg["epochs"]
    steps_per_epoch = len(train_loader)
    total = epochs * steps_per_epoch
    warmup = int(cfg.get("warmup_epochs", 0) * steps_per_epoch)
    schedule = cfg.get("schedule", "constant")
    ls = cfg.get("label_smoothing", 0.0)
    a_mix, a_cut = cfg.get("mixup_alpha", 0.0), cfg.get("cutmix_alpha", 0.0)
    mix_prob = cfg.get("mix_prob", 1.0) if (a_mix > 0 or a_cut > 0) else 0.0
    rng = np.random.RandomState(cfg.get("seed", 0))

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"]}
    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}
    best_state = copy.deepcopy(model.state_dict())
    step, t0 = 0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, tot_correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            f = lr_factor(step, total, warmup, schedule)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            if mix_prob > 0 and rng.rand() < mix_prob:
                images_in, soft = mix_batch(images, labels, num_classes, a_mix, a_cut, rng)
                soft = soft * (1 - ls) + ls / num_classes
                logits = model(images_in)
                loss = torch.sum(-soft * F.log_softmax(logits, 1), 1).mean()
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, labels, label_smoothing=ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            step += 1
            tot_loss += loss.item() * labels.size(0)
            tot_correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device)
        hist["train_loss"].append(tot_loss / seen)
        hist["train_acc"].append(tot_correct / seen)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["time"].append(time.time() - t0)
        # model selection: best val accuracy, ties broken by lower val loss
        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
        log(f"epoch {epoch:03d}/{epochs} | train loss {tot_loss / seen:.4f} acc {tot_correct / seen:.4f}"
            f" | val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - t0:.0f}s")

    final = {"val_acc": hist["val_acc"][-1], "val_loss": hist["val_loss"][-1], "epoch": epochs}
    model.load_state_dict(best_state)
    return model, hist, best, final
