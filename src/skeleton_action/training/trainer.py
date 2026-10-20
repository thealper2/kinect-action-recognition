"""Configuration-driven training loop with checkpointing, early stopping and resume."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from skeleton_action.constants import CLASS_NAMES, DATASET_ID, NUM_ACTIONS, NUM_JOINTS
from skeleton_action.data.dataset import SkeletonDataset, cached_preprocess
from skeleton_action.data.kard import KardCollection
from skeleton_action.data.loading import get_collection
from skeleton_action.data.splits import SubjectSplit
from skeleton_action.evaluation.metrics import classification_metrics
from skeleton_action.models import build_model, count_parameters
from skeleton_action.preprocessing.augment import AugmentConfig
from skeleton_action.preprocessing.transforms import PreprocessConfig
from skeleton_action.training.checkpoint import FORMAT_VERSION, load_checkpoint, save_checkpoint
from skeleton_action.utils.config import dump_yaml
from skeleton_action.utils.logging import JsonlWriter, get_logger
from skeleton_action.utils.seed import (
    rng_states,
    seed_everything,
    select_device,
    set_rng_states,
    software_versions,
)

log = get_logger()


# --------------------------------------------------------------------------- builders


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=wd,
            momentum=float(cfg.get("momentum", 0.9)),
            nesterov=bool(cfg.get("nesterov", True)),
        )
    raise ValueError(f"unknown optimizer {name!r}")


class WarmupCosine:
    """Epoch-level LR multiplier: linear warmup then cosine decay to ``min_factor``."""

    def __init__(self, epochs: int, warmup: int, min_factor: float) -> None:
        self.epochs, self.warmup, self.min_factor = epochs, warmup, min_factor

    def __call__(self, epoch: int) -> float:
        if self.warmup > 0 and epoch < self.warmup:
            return (epoch + 1) / self.warmup
        span = max(1, self.epochs - self.warmup)
        progress = min(1.0, (epoch - self.warmup) / span)
        return self.min_factor + (1 - self.min_factor) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(opt: torch.optim.Optimizer, cfg: dict, epochs: int):
    name = (cfg or {}).get("name", "cosine").lower()
    if name == "none":
        return None
    if name == "cosine":
        base = opt.param_groups[0]["lr"]
        min_factor = float(cfg.get("min_lr", 1e-5)) / base
        return torch.optim.lr_scheduler.LambdaLR(
            opt, WarmupCosine(epochs, int(cfg.get("warmup_epochs", 5)), min_factor)
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(cfg.get("step_size", 50)), gamma=float(cfg.get("gamma", 0.1))
        )
    raise ValueError(f"unknown scheduler {name!r}")


# --------------------------------------------------------------------------- epochs


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = None,
    grad_clip: float | None = None,
) -> dict:
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates under
    ``torch.no_grad`` in eval mode. Metrics are aggregated over *all* samples."""
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    preds, labels, probs, idxs = [], [], [], []
    max_grad = 0.0
    t0 = time.perf_counter()
    with torch.set_grad_enabled(train):
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
                loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip if grad_clip else float("inf")
                )
                if not torch.isfinite(gn):
                    raise FloatingPointError(f"non-finite gradient norm {gn.item()}")
                max_grad = max(max_grad, float(gn))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            bs = y.shape[0]
            total_loss += float(loss.detach()) * bs
            n += bs
            p = torch.softmax(logits.detach().float(), dim=-1)
            probs.append(p.cpu())
            preds.append(p.argmax(-1).cpu())
            labels.append(y.cpu())
            idxs.append(idx)
    dt = time.perf_counter() - t0
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    m = classification_metrics(y_true, y_pred, NUM_ACTIONS)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "n": n,
        "seconds": dt,
        "samples_per_s": n / dt if dt > 0 else float("nan"),
        "max_grad_norm": max_grad,
        "_y_true": y_true,
        "_y_pred": y_pred,
        "_probs": torch.cat(probs).numpy(),
        "_idx": torch.cat(idxs).numpy(),
    }


# --------------------------------------------------------------------------- runner


def run_paths(cfg: dict, output_root: str | Path = "outputs") -> dict[str, Path]:
    seed = cfg["training"].get("seed", 0)
    run_id = f"{cfg['name']}/seed{seed}"
    root = Path(output_root)
    return {
        "run_id": Path(run_id),
        "ckpt": root / "checkpoints" / run_id,
        "metrics": root / "metrics" / run_id,
        "figures": root / "figures" / run_id,
    }


def _data_stats(col: KardCollection, idx: np.ndarray) -> dict:
    lengths = col.lengths[idx]
    return {
        "train_seq_len_median": float(np.median(lengths)),
        "train_seq_len_p05": float(np.percentile(lengths, 5)),
        "train_seq_len_p95": float(np.percentile(lengths, 95)),
    }


def train(
    cfg: dict,
    output_root: str | Path = "outputs",
    resume: str | Path | None = None,
    overwrite: bool = False,
    evaluate_test: bool = True,
) -> dict:
    """Train one model as described by a resolved config. Returns the run summary dict.

    Model selection uses validation macro-F1 only. The test split (if any) is evaluated
    once, after training, on the best-validation checkpoint.
    """
    tcfg = cfg["training"]
    seed = int(tcfg.get("seed", 0))
    seed_everything(seed, bool(tcfg.get("deterministic", True)))
    device = select_device(tcfg.get("device", "auto"))
    paths = run_paths(cfg, output_root)
    last_path = paths["ckpt"] / "last.pt"
    best_path = paths["ckpt"] / "best.pt"
    if resume is None and last_path.exists() and not overwrite:
        raise FileExistsError(f"{last_path} exists. Use --resume {last_path} or --overwrite.")
    if overwrite and resume is None:
        for p in (paths["ckpt"] / "train_log.jsonl",):
            p.unlink(missing_ok=True)

    # ---- data
    dcfg = cfg["dataset"]
    pre = PreprocessConfig.from_dict(dcfg.get("preprocess"))
    aug = AugmentConfig.from_dict(dcfg.get("augment"))
    split = SubjectSplit.from_dict(cfg["split"])
    col = get_collection(dcfg)
    parts = split.indices(col.subjects)
    if len(parts["train"]) == 0 or len(parts["val"]) == 0:
        raise ValueError(f"empty train/val partition for split {split.to_dict()}")
    fixed = cached_preprocess(col, pre, dcfg.get("processed_dir"))
    ds_train = SkeletonDataset(col, parts["train"], pre, aug, seed=seed, precomputed=fixed)
    ds_val = SkeletonDataset(col, parts["val"], pre, None, precomputed=fixed)
    bs = int(tcfg.get("batch_size", 32))
    nw = int(tcfg.get("num_workers", 0))
    gen = torch.Generator().manual_seed(seed)
    dl_train = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        generator=gen,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    dl_val = DataLoader(ds_val, batch_size=bs * 2, shuffle=False, num_workers=nw)

    # ---- model / optim
    mcfg = cfg["model"]
    model_params = dict(mcfg.get("params") or {})
    model = build_model(
        mcfg["name"], NUM_ACTIONS, pre.num_frames, NUM_JOINTS, pre.num_channels, **model_params
    ).to(device)
    n_params = count_parameters(model)
    opt = build_optimizer(model.parameters(), tcfg.get("optimizer", {}))
    epochs = int(tcfg.get("epochs", 100))
    sched = build_scheduler(opt, tcfg.get("scheduler", {}), epochs)
    amp_dtype = None
    scaler = None
    if tcfg.get("amp", False) and device.type == "cuda":
        amp_dtype = (
            torch.float16 if tcfg.get("amp_dtype", "bfloat16") == "float16" else (torch.bfloat16)
        )
        if amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")
    weight = None
    if tcfg.get("class_weighting", False):
        counts = np.bincount(col.labels[parts["train"]], minlength=NUM_ACTIONS).astype(float)
        weight = torch.tensor(
            counts.sum() / (NUM_ACTIONS * np.maximum(counts, 1)), dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(weight=weight)
    grad_clip = tcfg.get("grad_clip")
    es = tcfg.get("early_stopping", {}) or {}
    patience = int(es.get("patience", 0))  # 0 disables
    min_delta = float(es.get("min_delta", 0.0))
    ckpt_every = max(1, int(tcfg.get("checkpoint_every", 1)))

    # ---- state
    start_epoch, best_metric, best_epoch, bad_epochs = 1, -math.inf, 0, 0
    history: list[dict] = []
    if resume is not None:
        ck = load_checkpoint(resume, map_location=device)
        if ck["model_name"] != mcfg["name"] or ck["preprocess"] != pre.to_dict():
            raise ValueError("resume checkpoint does not match the current configuration")
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        if sched is not None and ck.get("scheduler_state"):
            sched.load_state_dict(ck["scheduler_state"])
        if scaler is not None and ck.get("scaler_state"):
            scaler.load_state_dict(ck["scaler_state"])
        start_epoch = int(ck["epoch"]) + 1
        best_metric = float(ck["best_metric"])
        best_epoch = int(ck["best_epoch"])
        bad_epochs = int(ck.get("bad_epochs", 0))
        history = list(ck.get("history", []))
        if ck.get("rng"):
            set_rng_states(ck["rng"])
        if ck.get("dataset_rng"):
            ds_train.set_rng_state(ck["dataset_rng"])
        if ck.get("loader_generator") is not None:
            gen.set_state(ck["loader_generator"])
        log.info(
            "resumed from %s at epoch %d (best %.4f @ %d)",
            resume,
            start_epoch - 1,
            best_metric,
            best_epoch,
        )

    paths["ckpt"].mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, paths["ckpt"] / "config.yaml")
    jlog = JsonlWriter(paths["ckpt"] / "train_log.jsonl")
    jlog.write(
        "start",
        run_id=str(paths["run_id"]),
        device=str(device),
        params=n_params,
        n_train=len(ds_train),
        n_val=len(ds_val),
        resume=str(resume) if resume else None,
    )
    log.info(
        "run %s | model %s (%d params) | device %s | train %d / val %d / test %d",
        paths["run_id"],
        mcfg["name"],
        n_params,
        device,
        len(ds_train),
        len(ds_val),
        len(parts["test"]),
    )

    static_meta = {
        "format_version": FORMAT_VERSION,
        "model_name": mcfg["name"],
        "model_params": model_params,
        "num_classes": NUM_ACTIONS,
        "num_frames": pre.num_frames,
        "num_joints": NUM_JOINTS,
        "in_channels": pre.num_channels,
        "class_names": list(CLASS_NAMES),
        "action_ids": list(range(1, NUM_ACTIONS + 1)),
        "dataset_id": DATASET_ID,
        "dataset_kind": col.kind,
        "train_seq_ids": [col.seq_ids[i] for i in parts["train"]],
        "preprocess": pre.to_dict(),
        "augment": aug.to_dict(),
        "split": split.to_dict(),
        "seed": seed,
        "config": cfg,
        "versions": software_versions(),
        "num_parameters": n_params,
        "data_stats": _data_stats(col, parts["train"]),
    }

    def snapshot(epoch: int) -> dict:
        return {
            **static_meta,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict() if sched is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
            "rng": rng_states(),
            "dataset_rng": ds_train.rng_state(),
            "loader_generator": gen.get_state(),
        }

    stopped_early = False
    for epoch in range(start_epoch, epochs + 1):
        lr = opt.param_groups[0]["lr"]
        tr = run_epoch(model, dl_train, device, criterion, opt, scaler, amp_dtype, grad_clip)
        va = run_epoch(model, dl_val, device, criterion)
        if sched is not None:
            sched.step()
        rec = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": tr["loss"],
            "train_acc": tr["accuracy"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": va["loss"],
            "val_acc": va["accuracy"],
            "val_macro_f1": va["macro_f1"],
            "train_seconds": tr["seconds"],
            "val_seconds": va["seconds"],
            "train_samples_per_s": tr["samples_per_s"],
            "max_grad_norm": tr["max_grad_norm"],
        }
        history.append(rec)
        improved = va["macro_f1"] > best_metric + min_delta
        if improved:
            best_metric, best_epoch, bad_epochs = va["macro_f1"], epoch, 0
        else:
            bad_epochs += 1
        # last.pt is written every `checkpoint_every` epochs and whenever the run stops
        # (writing ~6 MB per epoch is a large share of the epoch time on /mnt/* drives).
        stop_after = int(tcfg.get("stop_after_epoch") or 0)
        save_last = (
            epoch % ckpt_every == 0
            or epoch == epochs
            or bool(patience and bad_epochs >= patience)
            or bool(stop_after and epoch >= stop_after)
        )
        if improved or save_last:
            snap = snapshot(epoch)
            if improved:
                save_checkpoint(best_path, snap)
            if save_last:
                save_checkpoint(last_path, snap)
        jlog.write("epoch", **rec, best_val_macro_f1=best_metric, best_epoch=best_epoch)
        log.info(
            "ep %3d | lr %.2e | train loss %.4f acc %.3f f1 %.3f | val loss %.4f acc %.3f "
            "f1 %.3f | best %.3f@%d | %.1fs %.0f samp/s%s",
            epoch,
            lr,
            tr["loss"],
            tr["accuracy"],
            tr["macro_f1"],
            va["loss"],
            va["accuracy"],
            va["macro_f1"],
            best_metric,
            best_epoch,
            tr["seconds"] + va["seconds"],
            tr["samples_per_s"],
            " *" if improved else "",
        )
        if patience and bad_epochs >= patience:
            log.info("early stopping: no val macro-F1 improvement for %d epochs", patience)
            stopped_early = True
            break
        if tcfg.get("stop_after_epoch") and epoch >= int(tcfg["stop_after_epoch"]):
            # Time-boxed session / interruption simulation: resume later from last.pt.
            log.info("stop_after_epoch=%s reached; resume with --resume %s", epoch, last_path)
            jlog.write("paused", epoch=epoch)
            return {
                "run_id": str(paths["run_id"]),
                "paused_at_epoch": epoch,
                "checkpoint": str(last_path),
            }

    jlog.write(
        "end", best_val_macro_f1=best_metric, best_epoch=best_epoch, stopped_early=stopped_early
    )
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(paths["metrics"] / "history.csv", index=False)
    from skeleton_action.evaluation.plots import plot_history

    plot_history(history, paths["figures"] / "curves.png", title=str(paths["run_id"]))

    summary = {
        "run_id": str(paths["run_id"]),
        "model": mcfg["name"],
        "params": n_params,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "checkpoint": str(best_path),
        "split": split.to_dict(),
        "seed": seed,
    }
    from skeleton_action.evaluation.evaluator import evaluate_checkpoint

    parts_to_eval = ["val"] + (["test"] if evaluate_test and len(parts["test"]) else [])
    for part in parts_to_eval:
        res = evaluate_checkpoint(
            best_path, part, paths["metrics"], paths["figures"], device=device, collection=col
        )
        summary[f"{part}_metrics"] = {
            k: res[k]
            for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
        }
    (paths["metrics"] / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "done: %s",
        json.dumps({k: summary[k] for k in ("run_id", "best_epoch", "best_val_macro_f1")}),
    )
    return summary
