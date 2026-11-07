"""Paste this complete file into a Kaggle/Colab code cell and run.

CNN baselines + progressive fine-tuning + frozen DINOv2/logistic regression.
The final path remains /kaggle/working/plant_option_c_v3_optimized/submission.csv.
Internet or cached pretrained weights are required. No test labels are used.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ARCHES = ("efficientnet_b2", "convnext_tiny", "resnet50")
VARIANTS = ("base", "no_crop", "no_mixup", "no_ema")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BASE_LR = {"efficientnet_b2": (8e-5, 6e-4),
           "convnext_tiny": (5e-5, 5e-4),
           "resnet50": (7e-5, 6e-4)}
BATCH = {"efficientnet_b2": 16, "convnext_tiny": 8, "resnet50": 12}

# For a pasted Notebook cell, edit values here; command-line arguments are optional.
# Paths below default to the original Kaggle paths. For Colab, supply actual paths.
NOTEBOOK_CONFIG = {
    # "train": "/content/data/train", "test": "/content/data/test",
    # "template": "/content/data/submission-for-task2.csv",
    # "architectures": "efficientnet_b2",  # quicker single-CNN experiment
    # "metric": "macro_f1",               # use the competition's official metric
}


def args_parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    base = Path("/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2")
    p.add_argument("--train", type=Path, default=base / "train", help="ImageFolder train directory")
    p.add_argument("--test", type=Path, default=base / "test")
    p.add_argument("--template", type=Path,
                   default=Path("/kaggle/input/competitions/2026task1/submission-for-task2.csv"))
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/plant_option_c_v3_optimized"))
    p.add_argument("--architectures", default=",".join(ARCHES))
    p.add_argument("--variants", default="base", help="Comma list; use all four for ablation")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--size", type=int, default=288)
    p.add_argument("--highres-size", type=int, default=320)
    p.add_argument("--highres", action=argparse.BooleanOptionalAction, default=False,
                   help="Optional extra high-resolution inference; progressive training is separate")
    p.add_argument("--progressive-epochs", type=int, default=3)
    p.add_argument("--progressive-size", type=int, default=384)
    p.add_argument("--progressive-lr-scale", type=float, default=.1)
    p.add_argument("--resize-mode", choices=("pad", "stretch"), default="pad")
    p.add_argument("--compare-resize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tta-views", choices=(1, 2, 4, 8), type=int, default=4)
    p.add_argument("--dino", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dino-model", default="dinov2_vits14")
    p.add_argument("--dino-size", type=int, default=336)
    p.add_argument("--dino-batch-size", type=int, default=16)
    p.add_argument("--dino-inner-folds", type=int, default=3)
    p.add_argument("--groups-csv", type=Path, default=None)
    p.add_argument("--lr-scale", type=float, default=1.0)
    p.add_argument("--mixup-prob", type=float, default=0.35)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--metric", choices=("accuracy", "macro_f1"), default="accuracy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tta-chunk", type=int, default=2)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    # A pasted Notebook cell inherits kernel flags (Colab adds
    # --HistoryManager.hist_file=:memory:). They are not script options.
    kernel_name = Path(sys.argv[0]).name
    in_notebook = (kernel_name in {"colab_kernel_launcher.py", "ipykernel_launcher.py"}
                   or "ipykernel" in sys.modules or "google.colab" in sys.modules)
    unknown = set(NOTEBOOK_CONFIG) - {action.dest for action in p._actions}
    if unknown:
        p.error(f"Unknown NOTEBOOK_CONFIG keys: {sorted(unknown)}")
    p.set_defaults(**NOTEBOOK_CONFIG)
    if argv is None:
        argv = [] if in_notebook else sys.argv[1:]
    a = p.parse_args(argv)
    for name in ("train", "test", "template", "out", "groups_csv"):
        if getattr(a, name) is not None:
            setattr(a, name, Path(getattr(a, name)))
    a.architectures = tuple(x.strip() for x in a.architectures.split(",") if x.strip())
    a.variants = tuple(x.strip() for x in a.variants.split(",") if x.strip())
    if not a.architectures or any(x not in ARCHES for x in a.architectures):
        p.error(f"architectures must be from {ARCHES}")
    if not a.variants or any(x not in VARIANTS for x in a.variants):
        p.error(f"variants must be from {VARIANTS}")
    if a.folds < 2 or a.epochs < 1 or a.freeze_epochs >= a.epochs:
        p.error("Need folds >= 2 and 0 <= freeze-epochs < epochs")
    if a.freeze_epochs < 0 or a.size < 96 or a.tta_chunk < 1:
        p.error("Invalid freeze-epochs, size or tta-chunk")
    if len(set(a.architectures)) != len(a.architectures) or len(set(a.variants)) != len(a.variants):
        p.error("Duplicate architectures or variants")
    if not 0 <= a.mixup_prob <= 1 or a.mixup_alpha <= 0 or a.lr_scale <= 0:
        p.error("MixUp probability must be in [0,1]; alpha and lr-scale must be positive")
    if a.progressive_epochs < 0 or a.progressive_size < a.size or a.progressive_lr_scale <= 0:
        p.error("Invalid progressive fine-tuning configuration")
    if a.dino_size < 14 or a.dino_size % 14 or a.dino_batch_size < 1 or a.dino_inner_folds < 2:
        p.error("DINO size must be a positive multiple of 14; batch >=1 and inner folds >=2")
    if a.workers < 0 or a.patience < 1:
        p.error("workers >=0 and patience >=1 are required")
    return a


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GreenCrop:
    def __call__(self, image):
        hsv = np.asarray(image.convert("HSV"))
        mask = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 125)
                & (hsv[..., 1] >= 28) & (hsv[..., 2] >= 25))
        if mask.mean() < 0.015:
            return image
        yy, xx = np.where(mask)
        h, w = mask.shape
        x1, x2, y1, y2 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
        dx, dy = int((x2 - x1) * .18), int((y2 - y1) * .18)
        x1, x2 = max(0, x1 - dx), min(w, x2 + dx)
        y1, y2 = max(0, y1 - dy), min(h, y2 + dy)
        if (x2 - x1) * (y2 - y1) < .08 * w * h:
            return image
        return image.crop((int(x1), int(y1), int(x2), int(y2)))


class LetterboxResize:
    """Resize without changing aspect ratio; center-pad with the ImageNet mean."""
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, image):
        w, h = image.size
        ratio = self.size / max(w, h)
        new_size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), tuple(round(x * 255) for x in MEAN))
        canvas.paste(image, ((self.size - new_size[0]) // 2, (self.size - new_size[1]) // 2))
        return canvas


class RandomCropResize:
    def __init__(self, size, resize_mode="pad", scale=(.75, 1.)):
        self.size, self.resize_mode, self.scale = size, resize_mode, scale

    def __call__(self, image):
        top, left, h, w = T.RandomResizedCrop.get_params(image, self.scale, (.88, 1.12))
        image = TF.crop(image, top, left, h, w)
        if self.resize_mode == "pad":
            return LetterboxResize(self.size)(image)
        return TF.resize(image, [self.size, self.size], InterpolationMode.BICUBIC)


def train_tf(size, variant, resize_mode="pad", progressive=False):
    ops = [] if variant == "no_crop" else [T.RandomApply([GreenCrop()], p=.45)]
    strength = .10 if progressive else .15
    ops += [RandomCropResize(size, resize_mode, scale=(.85 if progressive else .75, 1.)),
            T.RandomHorizontalFlip(), T.RandomRotation(15 if progressive else 20,
                                                        interpolation=InterpolationMode.BILINEAR),
            T.ColorJitter(strength, strength, strength, .03), T.ToTensor(), T.Normalize(MEAN, STD),
            T.RandomErasing(p=.10, scale=(.02, .08), value="random")]
    return T.Compose(ops)


def eval_tf(size, crop, resize_mode="pad"):
    ops = [GreenCrop()] if crop else []
    resize = LetterboxResize(size) if resize_mode == "pad" else T.Resize(
        (size, size), interpolation=InterpolationMode.BICUBIC)
    return T.Compose(ops + [resize,
                            T.ToTensor(), T.Normalize(MEAN, STD)])


class PathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = self.transform(im.convert("RGB"))
        return image, int(self.labels[i])


def get_loader(paths, labels, tf, batch, shuffle, workers, device):
    return DataLoader(PathDataset(paths, labels, tf), batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=(device.type == "cuda"))


def build_model(arch, n_classes, pretrained, device):
    try:
        if arch == "efficientnet_b2":
            model = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
            dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(nn.Dropout(.20), nn.Linear(dim, n_classes))
        elif arch == "convnext_tiny":
            model = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(nn.Dropout(.30), nn.Linear(dim, n_classes))
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            dim = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(.30), nn.Linear(dim, n_classes))
    except Exception as exc:
        raise RuntimeError(f"Could not load {arch} pretrained weights; enable Kaggle Internet or attach weights") from exc
    return model.to(device)


def is_head(name, arch):
    return name.startswith("fc") if arch == "resnet50" else name.startswith("classifier")


def set_trainable(model, arch, backbone):
    for name, param in model.named_parameters():
        param.requires_grad_(backbone or is_head(name, arch))


def model_score(y, probs, metric):
    pred = probs.argmax(axis=1)
    if metric == "accuracy":
        return float(np.mean(pred == y))
    return float(f1_score(y, pred, average="macro", zero_division=0))


@torch.no_grad()
def ema_update(ema, model, step, decay=.995):
    d = min(decay, (1 + step) / (10 + step))
    for ep, p in zip(ema.parameters(), model.parameters()):
        ep.mul_(d).add_(p.detach(), alpha=1 - d)
    for eb, b in zip(ema.buffers(), model.buffers()):
        if eb.is_floating_point():
            eb.mul_(d).add_(b.detach(), alpha=1 - d)
        else:
            eb.copy_(b)


def scaled_batch(a, arch, size):
    return max(2, int(BATCH[arch] * min(1., (a.size / size) ** 2)))


def fit_fold(a, arch, variant, fold, train_idx, valid_idx, paths, y, n_classes, device,
             initial_state=None):
    started = time.perf_counter()
    progressive = initial_state is not None
    size = a.progressive_size if progressive else a.size
    epochs = a.progressive_epochs if progressive else a.epochs
    freeze_epochs = 0 if progressive else a.freeze_epochs
    stage = "progressive" if progressive else "base"
    seed_all(a.seed + fold)
    model = build_model(arch, n_classes, a.pretrained and not progressive, device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    ema = copy.deepcopy(model).eval() if variant != "no_ema" else None
    if ema is not None:
        ema.requires_grad_(False)
    set_trainable(model, arch, False)
    backbone, head = [], []
    for name, param in model.named_parameters():
        (head if is_head(name, arch) else backbone).append(param)
    lr_b, lr_h = BASE_LR[arch]
    lr_scale = a.lr_scale * (a.progressive_lr_scale if progressive else 1.)
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr_b * lr_scale},
                             {"params": head, "lr": lr_h * lr_scale}], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - freeze_epochs), eta_min=1e-7 if progressive else 1e-6)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    batch = scaled_batch(a, arch, size)
    tr = get_loader([paths[i] for i in train_idx], y[train_idx],
                    train_tf(size, variant, a.resize_mode, progressive),
                    batch, True, a.workers, device)
    va = get_loader([paths[i] for i in valid_idx], y[valid_idx], eval_tf(size, False, a.resize_mode),
                    batch, False, a.workers, device)
    best, best_loss, stale, step, state = -1., float("inf"), 0, 0, None
    for epoch in range(epochs):
        if epoch == freeze_epochs:
            set_trainable(model, arch, True)
        model.train()
        if epoch < freeze_epochs:
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            mix = variant != "no_mixup" and random.random() < a.mixup_prob * (.5 if progressive else 1.)
            if mix:
                lam = np.random.beta(a.mixup_alpha, a.mixup_alpha)
                perm = torch.randperm(len(yb), device=device)
                mixed = lam * xb + (1 - lam) * xb[perm]
                onehot = F.one_hot(yb, n_classes).float() * .95 + .05 / n_classes
                target = lam * onehot + (1 - lam) * onehot[perm]
            else:
                mixed = xb
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(mixed)
                loss = (-(target * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
                        if mix else F.cross_entropy(logits, yb, label_smoothing=.05))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                step += 1
                if ema is not None:
                    ema_update(ema, model, step)
        if epoch >= freeze_epochs:
            schedule.step()
        evaluated = ema if ema is not None else model
        evaluated.eval()
        probs, losses = [], []
        with torch.inference_mode():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = evaluated(xb)
                losses.append(float(F.cross_entropy(logits.float(), yb).item()) * len(yb))
                probs.append(F.softmax(logits.float(), 1).cpu().numpy())
        probabilities = np.concatenate(probs)
        if not np.isfinite(probabilities).all():
            raise FloatingPointError(f"Non-finite validation prediction: {arch}/{stage}/fold{fold}")
        score = model_score(y[valid_idx], probabilities, a.metric)
        val_loss = sum(losses) / len(valid_idx)
        print(f"{variant}/{arch}/{stage}@{size}/fold{fold} epoch{epoch+1}: {a.metric}={score:.4f} loss={val_loss:.4f}", flush=True)
        if score > best or (score == best and val_loss < best_loss):
            best, best_loss, stale = score, val_loss, 0
            state = {k: v.detach().cpu().clone() for k, v in evaluated.state_dict().items()}
        else:
            stale += 1
            if stale >= a.patience:
                break
    model.load_state_dict(state)
    del ema, opt, schedule, scaler, tr, va
    return model.eval(), {"best_score": best, "best_loss": best_loss, "epochs_run": epoch + 1,
                          "stage": stage, "size": size, "batch_size": batch,
                          "train_seconds": time.perf_counter() - started}


def views(x, tta, num_views=4):
    if not tta:
        return [x]
    if num_views == 1:
        return [x]
    if num_views == 2:
        return [x, x.flip(-1)]
    rotations = [x.rot90(k, (-2, -1)) for k in range(4)]
    return rotations if num_views == 4 else rotations + [v.flip(-1) for v in rotations]


@torch.inference_mode()
def predict(model, paths, size, crop, tta, batch, a, device, resize_mode=None, return_plain=False):
    ld = get_loader(paths, np.zeros(len(paths), dtype=int), eval_tf(size, crop, resize_mode or a.resize_mode),
                    batch, False, a.workers, device)
    output, plain_output = [], []
    for xb, _ in ld:
        xb = xb.to(device)
        vv = views(xb, tta, a.tta_views)
        total = None
        for j in range(0, len(vv), a.tta_chunk):
            chunk = vv[j:j + a.tta_chunk]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(torch.cat(chunk, dim=0))
            chunk_probs = F.softmax(logits.float(), dim=1).reshape(len(chunk), len(xb), -1)
            if j == 0 and return_plain:
                plain_output.append(chunk_probs[0].cpu().numpy())
            prob = chunk_probs.sum(0)
            total = prob if total is None else total + prob
        output.append((total / len(vv)).cpu().numpy())
    result = np.concatenate(output).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite inference probability")
    return (result, np.concatenate(plain_output)) if return_plain else result


def infer_modes(model, paths, a, arch, device, size=None):
    size = a.size if size is None else size
    batch = scaled_batch(a, arch, size)
    tta, plain = predict(model, paths, size, False, True, batch, a, device, return_plain=True)
    crop = predict(model, paths, size, True, True, batch, a, device)
    result = {"plain": plain, "tta": tta, "tta_crop": .5 * (tta + crop)}
    if a.compare_resize:
        alternate = "stretch" if a.resize_mode == "pad" else "pad"
        result[f"plain_{alternate}"] = predict(model, paths, size, False, False, batch, a, device,
                                                resize_mode=alternate)
    if a.highres:
        extra_size = max(size, a.highres_size)
        extra_batch = scaled_batch(a, arch, extra_size)
        hi = predict(model, paths, extra_size, False, False, extra_batch, a, device)
        hic = predict(model, paths, extra_size, True, False, extra_batch, a, device)
        result["tta_crop_highres"] = .8 * result["tta_crop"] + .1 * (hi + hic)
    return result


def weight_grid(n):
    if n < 1:
        raise ValueError("Need at least one ensemble member")
    def compositions(remaining, dimensions):
        if dimensions == 1:
            yield (remaining,)
        else:
            for count in range(remaining + 1):
                for tail in compositions(remaining - count, dimensions - 1):
                    yield (count,) + tail
    for counts in compositions(10, n):
        yield np.asarray(counts, dtype=float) / 10.
    if 10 % n:
        yield np.full(n, 1. / n)


def best_weights(probs, y, metric):
    best, answer = -1., None
    for w in weight_grid(len(probs)):
        p = sum(float(w[i]) * probs[i] for i in range(len(probs)))
        score = model_score(y, p, metric)
        if score > best + 1e-12:
            best, answer = score, w
    return answer, best


def select_recipe(predictions, y, folds, arches, metric):
    # Exploratory meta-selection: held labels are excluded from direct recipe
    # fitting, but base-model/checkpoint selection is NOT fully nested CV.
    n_classes = next(iter(predictions.values())).shape[1]
    cross_single = np.zeros((len(y), n_classes), np.float32)
    cross_ensemble = np.zeros_like(cross_single)
    fold_decisions = []
    for held in folds:
        train = np.setdiff1d(np.arange(len(y)), held)
        chosen = {}
        for arch in arches:
            keys = [k for k in predictions if k.split("|")[1] == arch]
            chosen[arch] = max(keys, key=lambda k: (model_score(y[train], predictions[k][train], metric), k))
        single_key = max(chosen.values(), key=lambda k: model_score(y[train], predictions[k][train], metric))
        cross_single[held] = predictions[single_key][held]
        probs = [predictions[chosen[arch]] for arch in arches]
        weights, _ = best_weights([p[train] for p in probs], y[train], metric)
        cross_ensemble[held] = sum(float(weights[i]) * probs[i][held] for i in range(len(probs)))
        fold_decisions.append({"single": single_key, "chosen": chosen,
                               "weights": weights.round(3).tolist()})
    single_score = model_score(y, cross_single, metric)
    ensemble_score = model_score(y, cross_ensemble, metric)
    chosen = {}
    for arch in arches:
        keys = [k for k in predictions if k.split("|")[1] == arch]
        chosen[arch] = max(keys, key=lambda k: (model_score(y, predictions[k], metric), k))
    full_single = max(chosen.values(), key=lambda k: model_score(y, predictions[k], metric))
    full_weights, _ = best_weights([predictions[chosen[arch]] for arch in arches], y, metric)
    use_ensemble = ensemble_score > single_score + 1e-12
    validated = cross_ensemble if use_ensemble else cross_single
    return {"crossfit_single": single_score, "crossfit_ensemble": ensemble_score,
            "use_ensemble": use_ensemble, "chosen": chosen, "single": full_single,
            "weights": full_weights.tolist(), "fold_decisions": fold_decisions}, validated


def make_splits(a, paths, y, classes):
    """Keep identical decoded RGB images and optional plant groups in one fold.

    groups_csv must contain path,group columns and cover every training image.
    Paths are relative to a.train. This detects exact decoded-pixel duplicates,
    not near duplicates or different photographs of the same plant; use group
    metadata for those cases. Image decoding matches PathDataset (no EXIF rotate).
    """
    import csv
    import hashlib
    from pathlib import Path, PureWindowsPath

    import numpy as np
    from PIL import Image
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

    paths = [Path(p).resolve() for p in paths]
    labels = np.asarray(y, dtype=np.int64)
    n = len(paths)
    expected_labels = set(range(len(classes)))
    if n != len(labels) or not n:
        raise ValueError("Training paths and labels must be nonempty and have equal length.")
    if set(labels.tolist()) != expected_labels:
        raise ValueError("Training labels must cover every class in classes, using contiguous IDs.")
    if a.folds < 2:
        raise ValueError("Need at least two folds.")
    if len(set(paths)) != n:
        raise ValueError("Training image list contains the same path more than once.")
    train_root = Path(a.train).resolve()
    try:
        relative_paths = [p.relative_to(train_root).as_posix() for p in paths]
    except ValueError as exc:
        raise ValueError("Every training image must be inside the train directory.") from exc
    index_by_path = {p: i for i, p in enumerate(paths)}
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    digest_first = {}
    digests = []
    print(f"Checking exact RGB duplicates in {n} training images...", flush=True)
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                h = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii"))
                h.update(rgb.tobytes())
                digest = h.hexdigest()
        except Exception as exc:
            raise ValueError(f"Cannot decode training image: {path}") from exc
        digests.append(digest)
        if digest in digest_first:
            j = digest_first[digest]
            if labels[i] != labels[j]:
                raise ValueError(
                    "Identical RGB images have conflicting labels: "
                    f"{relative_paths[j]} ({classes[labels[j]]}) and "
                    f"{relative_paths[i]} ({classes[labels[i]]}). Correct these labels first.")
            union(i, j)
        else:
            digest_first[digest] = i
        if (i + 1) % 500 == 0:
            print(f"  hashed {i + 1}/{n}", flush=True)

    supplied_groups = [""] * n
    groups_csv = getattr(a, "groups_csv", None)
    if groups_csv is not None:
        group_first = {}
        seen = set()
        with Path(groups_csv).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            if not {"path", "group"}.issubset(rows.fieldnames or []):
                raise ValueError("groups_csv must have path and group columns.")
            for line, row in enumerate(rows, start=2):
                raw_path = (row.get("path") or "").strip().replace("\\", "/")
                group = (row.get("group") or "").strip()
                if (not raw_path or not group or Path(raw_path).is_absolute()
                        or PureWindowsPath(raw_path).drive):
                    raise ValueError(f"groups_csv line {line}: need a relative path and nonempty group.")
                candidate = (train_root / raw_path).resolve()
                if candidate not in index_by_path:
                    raise ValueError(f"groups_csv line {line}: unknown training path {raw_path!r}.")
                i = index_by_path[candidate]
                if i in seen:
                    raise ValueError(f"groups_csv line {line}: duplicate training path {raw_path!r}.")
                seen.add(i)
                supplied_groups[i] = group
                if group in group_first:
                    union(i, group_first[group])
                else:
                    group_first[group] = i
        if len(seen) != n:
            missing = [relative_paths[i] for i in range(n) if i not in seen]
            raise ValueError(f"groups_csv is missing {len(missing)} training paths; examples: {missing[:5]}")

    roots = [root(i) for i in range(n)]
    group_lookup = {value: i for i, value in enumerate(sorted(set(roots)))}
    groups = np.array([group_lookup[value] for value in roots], dtype=np.int64)
    unique_groups = len(group_lookup)
    grouped = unique_groups < n
    for label, name in enumerate(classes):
        count = len(np.unique(groups[labels == label]))
        if count < a.folds:
            raise ValueError(f"Class {name!r} has only {count} independent groups for {a.folds} folds. "
                             "Reduce --folds or add independent labeled images.")
    dummy = np.zeros((n, 1), dtype=np.uint8)
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        splits = list(splitter.split(dummy, labels, groups))
    else:
        splitter = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        splits = list(splitter.split(dummy, labels))
    fold_ids = np.full(n, -1, dtype=np.int64)
    fold_reports = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        for split_name, idx in (("train", train_idx), ("validation", val_idx)):
            missing = expected_labels - set(labels[idx].tolist())
            if missing:
                names = [classes[label] for label in sorted(missing)]
                raise ValueError(f"Fold {fold} {split_name} is missing classes {names}. "
                                 "Reduce --folds or revise plant grouping; no training was started.")
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise RuntimeError(f"Internal split error: a group crosses fold {fold}.")
        fold_ids[val_idx] = fold
        fold_reports.append({
            "fold": fold,
            "train_count": int(len(train_idx)), "val_count": int(len(val_idx)),
            "train_class_counts": {str(name): int(np.sum(labels[train_idx] == i))
                                   for i, name in enumerate(classes)},
            "val_class_counts": {str(name): int(np.sum(labels[val_idx] == i))
                                 for i, name in enumerate(classes)}})
    if np.any(fold_ids < 0):
        raise RuntimeError("Internal split error: some images have no validation fold.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "class", "fold", "group", "rgb_sha256", "supplied_group"])
        writer.writerows((relative_paths[i], int(labels[i]), classes[labels[i]], int(fold_ids[i]),
                          int(groups[i]), digests[i], supplied_groups[i]) for i in range(n))
    split_report = {
        "strategy": type(splitter).__name__, "seed": int(a.seed), "folds": int(a.folds),
        "image_count": n, "exact_duplicate_count": n - len(digest_first),
        "unique_groups": unique_groups, "groups_csv": str(groups_csv) if groups_csv else None,
        "limitation": "RGB hashes identify exact pixel duplicates only. Provide group metadata "
                      "for near duplicates and multiple photos of the same plant.",
        "fold_details": fold_reports}
    print(f"Splits: {split_report['strategy']}; {unique_groups} independent groups; "
          f"{split_report['exact_duplicate_count']} extra exact duplicates.", flush=True)
    return splits, groups if grouped else None, split_report


def _dino_aligned_proba(classifier, features, n_classes):
    """Align sklearn probability columns with the ImageFolder class IDs."""
    probabilities = np.zeros((len(features), n_classes), dtype=np.float32)
    class_ids = np.asarray(classifier.classes_, dtype=int)
    if len(class_ids) != n_classes or not np.array_equal(
            np.sort(class_ids), np.arange(n_classes)):
        raise ValueError("DINO classifier training fold does not contain every class")
    probabilities[:, class_ids] = classifier.predict_proba(features)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("DINO classifier produced non-finite probabilities")
    return probabilities


def _dino_score(y, probabilities, metric, n_classes):
    from sklearn.metrics import f1_score
    predictions = probabilities.argmax(axis=1)
    if metric == "accuracy":
        return float(np.mean(predictions == y))
    return float(f1_score(y, predictions, labels=np.arange(n_classes),
                          average="macro", zero_division=0))


def _dino_inner_splits(y, groups, requested_folds, seed, n_classes):
    """Return indices local to the outer training set, or an explicit skip."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    min_count = int(np.bincount(y, minlength=n_classes).min())
    if groups is None:
        possible = min(requested_folds, min_count)
    else:
        independent_per_class = [len(np.unique(groups[y == label]))
                                 for label in range(n_classes)]
        possible = min(requested_folds, min(independent_per_class))
    # Some group layouts cannot retain every class in all training folds.
    # Decrease fold count, never fall back to splitting members of a group.
    for n_folds in range(possible, 1, -1):
        if groups is None:
            splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
            inner = list(splitter.split(np.zeros(len(y)), y))
        else:
            splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
            inner = list(splitter.split(np.zeros(len(y)), y, groups))
        complete = all(len(tr) and len(va)
                       and len(np.unique(y[tr])) == n_classes for tr, va in inner)
        disjoint = groups is None or all(
            not np.intersect1d(groups[tr], groups[va]).size for tr, va in inner)
        if complete and disjoint:
            return inner, None
    return [], ("Insufficient independent examples/groups for inner CV with "
                "all classes in each training fold; used a prespecified C.")


def _dino_new_classifier(c_value, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=2000,
                           random_state=seed),
    )


def _dino_fit_classifiers(features, y, test_features, splits, classes, *,
                          groups=None, c_values=(.01, .1, 1., 10.),
                          inner_folds=3, metric="accuracy", seed=42,
                          model_dir=None, model_name="dinov2_vits14", metadata=None):
    """Nested tuning on fixed frozen features; intentionally independently testable."""
    import time
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    import joblib

    n_classes = len(classes)
    y = np.asarray(y, dtype=int)
    groups = None if groups is None else np.asarray(groups)
    c_values = tuple(sorted(set(float(c) for c in c_values)))
    if not c_values or any(c <= 0 or not np.isfinite(c) for c in c_values):
        raise ValueError("DINO logistic C values must be finite and positive")
    if inner_folds < 2:
        raise ValueError("DINO inner folds must be >= 2")
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    test_mean = np.zeros((len(test_features), n_classes), dtype=np.float32)
    seen = np.zeros(len(y), dtype=int)
    fold_reports = []
    for fold, (tr, va) in enumerate(splits, 1):
        started = time.perf_counter()
        train_groups = None if groups is None else groups[tr]
        inner, skip_reason = _dino_inner_splits(
            y[tr], train_groups, inner_folds, seed + fold, n_classes)
        tuning = []
        for c_value in c_values if inner else ():
            scores, convergence_warnings = [], 0
            for inner_train, inner_valid in inner:
                # Pipeline.fit ensures the scaler never observes validation rows.
                candidate = _dino_new_classifier(c_value, seed + fold)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    candidate.fit(features[tr[inner_train]], y[tr[inner_train]])
                convergence_warnings += sum(issubclass(w.category, ConvergenceWarning)
                                            for w in caught)
                probabilities = _dino_aligned_proba(
                    candidate, features[tr[inner_valid]], n_classes)
                scores.append(_dino_score(y[tr[inner_valid]], probabilities,
                                          metric, n_classes))
            tuning.append({"C": c_value, "mean_score": float(np.mean(scores)),
                           "std_score": float(np.std(scores)), "fold_scores": scores,
                           "convergence_warnings": convergence_warnings})
        if tuning:
            chosen_c = max(tuning, key=lambda row: (row["mean_score"], -row["C"]))["C"]
        else:
            # Fixed fallback independent of outer validation labels.
            chosen_c = min(c_values, key=lambda c: (abs(np.log10(c)), c))
        classifier = _dino_new_classifier(chosen_c, seed + fold)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(features[tr], y[tr])
        convergence_warnings = sum(issubclass(w.category, ConvergenceWarning)
                                   for w in caught)
        valid_probabilities = _dino_aligned_proba(classifier, features[va], n_classes)
        oof[va] = valid_probabilities
        seen[va] += 1
        test_mean += _dino_aligned_proba(classifier, test_features, n_classes) / len(splits)
        checkpoint = None
        if model_dir is not None:
            checkpoint = Path(model_dir) / f"{model_name}_linear_fold{fold}.joblib"
            joblib.dump({"pipeline": classifier, "classes": list(classes),
                         "class_ids": list(range(n_classes)), "fold": fold,
                         "chosen_C": chosen_c, "backbone": model_name,
                         "metadata": metadata or {}}, checkpoint)
        item = {"fold": fold, "chosen_C": chosen_c, "inner_folds": len(inner),
                "tuning_status": "complete" if inner else "skipped",
                "tuning_skip_reason": skip_reason, "C_search": tuning,
                "train_count": len(tr), "valid_count": len(va),
                "accuracy": _dino_score(y[va], valid_probabilities, "accuracy", n_classes),
                "macro_f1": _dino_score(y[va], valid_probabilities, "macro_f1", n_classes),
                "convergence_warnings": convergence_warnings,
                "seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint) if checkpoint is not None else None}
        fold_reports.append(item)
        print(f"DINO linear fold {fold}/{len(splits)} C={chosen_c:g} "
              f"{metric}={item[metric]:.5f} ({item['seconds']:.1f}s)", flush=True)
        if skip_reason:
            print("DINO tuning note:", skip_reason, flush=True)
        if convergence_warnings or any(r["convergence_warnings"] for r in tuning):
            print("DINO logistic convergence warning; details saved in dino_analysis.json", flush=True)
    if not np.all(seen == 1):
        raise ValueError("DINO outer validation folds must cover each training row exactly once")
    report = {"model": model_name, "feature_extractor": "frozen pretrained DINOv2",
              "classifier": "StandardScaler + LogisticRegression (lbfgs)",
              "C_values": list(c_values), "metric": metric, "inner_folds_requested": inner_folds,
              "group_aware": groups is not None, "folds": fold_reports,
              "oof_accuracy": _dino_score(y, oof, "accuracy", n_classes),
              "oof_macro_f1": _dino_score(y, oof, "macro_f1", n_classes),
              "note": "Scaler and C tuning use only each outer training fold. "
                      "Frozen feature extraction observes no task labels. "
                      "Test probabilities are the mean of all outer-fold classifiers."}
    return oof, test_mean, report


def run_dino_branch(a, paths, y, test_paths, splits, classes, device, groups=None):
    """Extract once in FP32, then fit fold-specific linear classification heads.

    Call before CNN training so unavailable DINO weights fail immediately.
    Optional args: dino_model, dino_size, dino_batch_size, dino_inner_folds,
    dino_c_values. Defaults are ViT-S/14, 336, 32, 3, (.01,.1,1.,10.).
    """
    import time

    model_name = getattr(a, "dino_model", "dinov2_vits14")
    size = int(getattr(a, "dino_size", 336))
    batch = int(getattr(a, "dino_batch_size", 32))
    inner_folds = int(getattr(a, "dino_inner_folds", 3))
    c_values = getattr(a, "dino_c_values", (.01, .1, 1., 10.))
    if isinstance(c_values, str):
        c_values = tuple(float(c.strip()) for c in c_values.split(",") if c.strip())
    if size < 14 or size % 14 or batch < 1:
        raise ValueError("DINO size must be a positive multiple of 14 and batch >= 1")
    if model_name not in {"dinov2_vits14", "dinov2_vitb14"}:
        raise ValueError("Supported DINO backbones: dinov2_vits14, dinov2_vitb14")
    y = np.asarray(y, dtype=int)
    groups = None if groups is None else np.asarray(groups)
    n_classes = len(classes)
    if (len(paths) != len(y) or not len(test_paths)
            or not np.array_equal(np.unique(y), np.arange(n_classes))):
        raise ValueError("DINO paths/labels/classes are inconsistent or test set is empty")
    if groups is not None and len(groups) != len(y):
        raise ValueError("DINO groups must have one group ID per training image")
    splits = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in splits]
    seen = np.zeros(len(y), dtype=int)
    if len(splits) < 2:
        raise ValueError("DINO needs at least two outer folds")
    for tr, va in splits:
        if (not len(tr) or not len(va) or np.min(tr) < 0 or np.min(va) < 0
                or np.max(tr) >= len(y) or np.max(va) >= len(y)
                or len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va)
                or np.intersect1d(tr, va).size
                or len(np.unique(y[tr])) != n_classes):
            raise ValueError("DINO outer split has invalid indices or a missing training class")
        if groups is not None and np.intersect1d(groups[tr], groups[va]).size:
            raise ValueError("DINO outer split leaks a group between training and validation")
        seen[va] += 1
    if not np.all(seen == 1):
        raise ValueError("DINO outer folds must validate every image exactly once")

    seed_all(a.seed)
    model_dir = Path(a.out) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading official pretrained {model_name}; Internet or a populated Torch Hub cache is required.",
          flush=True)
    try:
        # Official interface: https://github.com/facebookresearch/dinov2
        model = torch.hub.load("facebookresearch/dinov2", model_name,
                               pretrained=True, trust_repo=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load official pretrained {model_name}. Enable Kaggle Internet/Colab network "
            "or provide a populated Torch Hub cache containing the official repository and weights. "
            "No random-weight replacement was used. Disable the DINO branch explicitly to run CNN only."
        ) from exc
    model = model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    resize_mode = getattr(a, "resize_mode", "pad")
    transform = eval_tf(size, False, resize_mode=resize_mode)
    extraction_start = time.perf_counter()

    def extract(image_paths, title):
        loader = get_loader(image_paths, np.zeros(len(image_paths), dtype=int),
                            transform, batch, False, a.workers, device)
        chunks = []
        count = 0
        with torch.inference_mode():
            for step, (xb, _) in enumerate(loader, 1):
                # FP32 by default; no autocast or training-mode updates.
                with torch.autocast(device_type=device.type, enabled=False):
                    features = model(xb.to(device=device, dtype=torch.float32, non_blocking=True))
                if features.ndim != 2 or not torch.isfinite(features).all().item():
                    raise FloatingPointError("DINO returned invalid frozen features")
                chunks.append(features.cpu().numpy())
                count += len(xb)
                if step == 1 or step % 20 == 0 or step == len(loader):
                    print(f"DINO {title} features {count}/{len(image_paths)}", flush=True)
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    try:
        features = extract(paths, "train")
        test_features = extract(test_paths, "test")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    extraction_seconds = time.perf_counter() - extraction_start
    metadata = {"repository": "facebookresearch/dinov2", "pretrained": True,
                "image_size": size, "resize_mode": resize_mode, "precision": "float32",
                "feature_dimension": int(features.shape[1]), "cache": "none",
                "extraction_seconds": extraction_seconds}
    oof, test_mean, report = _dino_fit_classifiers(
        features, y, test_features, splits, classes, groups=groups, c_values=c_values,
        inner_folds=inner_folds, metric=a.metric, seed=a.seed, model_dir=model_dir,
        model_name=model_name, metadata=metadata)
    report.update(metadata)
    (Path(a.out) / "dino_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return oof, test_mean, report


def main():
    started = time.perf_counter()
    a = args_parse()
    for label, path in (("train", a.train), ("test", a.test), ("submission template", a.template)):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}. In Colab, mount or copy the Kaggle dataset "
                "and pass --train/--test/--template when running as a script, "
                "or change their defaults in args_parse() before pasting the code."
            )
    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    print("Device:", device, "| output:", a.out, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    (a.out / "configuration.json").write_text(
        json.dumps(vars(a), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = datasets.ImageFolder(str(a.train))
    classes = raw.classes
    paths = [p for p, _ in raw.samples]
    y = np.asarray(raw.targets)
    if len(classes) != 5 or np.min(np.bincount(y)) < a.folds:
        raise ValueError("Expected five classes and at least folds images per class")
    template = pd.read_csv(a.template, dtype={"ID": str})
    if template.columns.tolist() != ["ID", "Category"] or not template.ID.is_unique:
        raise ValueError("Template must have unique ID and Category columns")
    test_paths = [str(a.test / image_id) for image_id in template.ID]
    if not all(Path(p).is_file() for p in test_paths):
        raise FileNotFoundError("A template ID is absent from test/")
    if not test_paths:
        raise ValueError("Submission template is empty")
    splits, groups, split_report = make_splits(a, paths, y, classes)
    held_folds = [va for _, va in splits]
    results, test_results, fit_log = {}, {}, []
    model_dir = a.out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    families = list(a.architectures)
    dino_report = None
    if a.dino:
        dino_oof, dino_test, dino_report = run_dino_branch(
            a, paths, y, test_paths, splits, classes, device, groups=groups)
        key = "frozen|dinov2|linear"
        results[key], test_results[key] = dino_oof, dino_test
        families.append("dinov2")
    for variant in a.variants:
        for arch in a.architectures:
            oof = {}
            test_sum = {}
            for fold, (tr, va) in enumerate(splits, 1):
                stages = ("base", "progressive") if a.progressive_epochs else ("base",)
                initial_state = None
                for stage in stages:
                    model, fit = fit_fold(a, arch, variant, fold, tr, va, paths, y, len(classes),
                                          device, initial_state=initial_state)
                    size = fit["size"]
                    suffix = "" if variant == "base" else f"_{variant}"
                    if stage == "progressive":
                        suffix += "_progressive"
                    checkpoint = model_dir / f"{arch}{suffix}_fold{fold}.pth"
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(state, checkpoint)
                    # Keep every contributing fold so the exported predictions are reproducible.
                    initial_state = state if stage == "base" else None
                    infer_start = time.perf_counter()
                    val_modes = infer_modes(model, [paths[i] for i in va], a, arch, device, size)
                    test_modes = infer_modes(model, test_paths, a, arch, device, size)
                    fit_log.append({"variant": variant, "architecture": arch, "fold": fold,
                                    "checkpoint": str(checkpoint), **fit,
                                    "inference_seconds": time.perf_counter() - infer_start})
                    for mode, p in val_modes.items():
                        name = f"{stage}_{mode}"
                        if name not in oof:
                            oof[name] = np.zeros((len(y), len(classes)), np.float32)
                            test_sum[name] = np.zeros((len(test_paths), len(classes)), np.float32)
                        oof[name][va] = p
                        test_sum[name] += test_modes[mode] / a.folds
                    del model, state
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            for mode in oof:
                key = f"{variant}|{arch}|{mode}"
                results[key], test_results[key] = oof[mode], test_sum[mode]
                print(key, "OOF", model_score(y, results[key], a.metric), flush=True)
    report = pd.DataFrame([{"configuration": k,
                            "accuracy": model_score(y, p, "accuracy"),
                            "macro_f1": model_score(y, p, "macro_f1")}
                           for k, p in results.items()]).sort_values(a.metric, ascending=False)
    report.to_csv(a.out / "ablation_scores.csv", index=False)
    pd.DataFrame(fit_log).to_csv(a.out / "fold_training.csv", index=False)
    fold_scores = []
    for key, probs in results.items():
        for fold, (_, va) in enumerate(splits, 1):
            fold_scores.append({"configuration": key, "fold": fold,
                                "accuracy": model_score(y[va], probs[va], "accuracy"),
                                "macro_f1": model_score(y[va], probs[va], "macro_f1")})
    pd.DataFrame(fold_scores).to_csv(a.out / "fold_scores.csv", index=False)
    np.savez_compressed(a.out / "candidate_oof.npz", **results)
    np.savez_compressed(a.out / "candidate_test.npz", **test_results)
    recipe, validated = select_recipe(results, y, held_folds, families, a.metric)
    if recipe["use_ensemble"]:
        selected = sum(recipe["weights"][i] * test_results[recipe["chosen"][arch]]
                       for i, arch in enumerate(families))
    else:
        selected = test_results[recipe["single"]]
    if selected.shape != (len(template), len(classes)) or not np.isfinite(selected).all():
        raise ValueError("Invalid final probability shape or non-finite value")
    if np.any(selected < 0) or not np.allclose(selected.sum(1), 1., atol=1e-5):
        raise ValueError("Final probabilities must be nonnegative and sum to one")
    submission = template.copy()
    submission["Category"] = [classes[i] for i in selected.argmax(1)]
    if submission.ID.tolist() != template.ID.tolist() or submission.Category.isna().any():
        raise ValueError("Final submission ID order or categories are invalid")
    submission.to_csv(a.out / "submission.csv.tmp", index=False)
    os.replace(a.out / "submission.csv.tmp", a.out / "submission.csv")
    np.savez_compressed(a.out / "crossfit_oof.npz", y=y, probabilities=validated)
    np.save(a.out / "final_test_probabilities.npy", selected)
    class_report = classification_report(y, validated.argmax(1), labels=np.arange(len(classes)),
                                         target_names=classes, output_dict=True, zero_division=0)
    (a.out / "class_metrics.json").write_text(json.dumps(class_report, indent=2), encoding="utf-8")
    overlaps = []
    chosen_keys = list(recipe["chosen"].values())
    for i, left in enumerate(chosen_keys):
        for right in chosen_keys[i + 1:]:
            e1, e2 = results[left].argmax(1) != y, results[right].argmax(1) != y
            union = int(np.count_nonzero(e1 | e2))
            overlaps.append({"model_a": left, "model_b": right,
                             "a_errors": int(e1.sum()), "b_errors": int(e2.sum()),
                             "shared_errors": int(np.count_nonzero(e1 & e2)),
                             "error_jaccard": float(np.count_nonzero(e1 & e2) / union) if union else 0.})
    pd.DataFrame(overlaps, columns=["model_a", "model_b", "a_errors", "b_errors", "shared_errors", "error_jaccard"]
                 ).to_csv(a.out / "error_overlap.csv", index=False)
    summary = {"metric": a.metric, "classes": classes, "class_counts": np.bincount(y).tolist(),
               "folds": a.folds, "size": a.size, "highres": a.highres,
               "architectures": a.architectures, "variants": a.variants,
               "families": families, "resize_mode": a.resize_mode, "tta_views": a.tta_views,
               "progressive_epochs": a.progressive_epochs, "progressive_size": a.progressive_size,
               "split_report": split_report, "dino_report": dino_report,
               "lr_scale": a.lr_scale, "mixup_prob": a.mixup_prob,
               "crossfit_accuracy": model_score(y, validated, "accuracy"),
               "crossfit_macro_f1": model_score(y, validated, "macro_f1"),
               "confusion_matrix": confusion_matrix(y, validated.argmax(1)).tolist(),
               "recipe": recipe,
               "total_minutes": (time.perf_counter() - started) / 60.,
               "note": "Exploratory CV, not an unbiased external test or fully nested CNN CV. "
                       "CNN epochs use fold validation; other-fold base models may have trained on held labels; "
                       "single versus ensemble is selected using CV scores. Progressive training adds compute "
                       "and also weakens augmentation, so its comparison does not isolate image size alone. "
                       "DINO C is tuned only within outer training folds. Test labels were never read."}
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved", a.out / "submission.csv")


if __name__ == "__main__":
    main()
