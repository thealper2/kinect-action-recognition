"""Evaluate a checkpoint on a subject-independent partition and write all artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skeleton_action.data.dataset import SkeletonDataset
from skeleton_action.data.kard import KardCollection
from skeleton_action.data.loading import get_collection
from skeleton_action.data.splits import SubjectSplit
from skeleton_action.evaluation.benchmark import measure_latency
from skeleton_action.evaluation.metrics import classification_metrics, top_confusions
from skeleton_action.evaluation.plots import plot_confusion
from skeleton_action.models import count_parameters
from skeleton_action.preprocessing.transforms import PreprocessConfig
from skeleton_action.training.checkpoint import load_checkpoint, model_from_checkpoint


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, probs [N, K], sample indices) with the model in eval mode."""
    model.eval()
    ys, ps, ix = [], [], []
    for x, y, idx in loader:
        ps.append(torch.softmax(model(x.to(device)).float(), -1).cpu())
        ys.append(y)
        ix.append(idx)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(ix).numpy()


def evaluate_checkpoint(
    ckpt_path: str | Path,
    partition: str = "test",
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    collection: KardCollection | None = None,
    split: SubjectSplit | None = None,
    latency: bool = True,
) -> dict:
    """Compute metrics of a checkpoint on ``partition`` (train/val/test) of its split.

    Writes ``<partition>_metrics.json``, ``<partition>_predictions.csv``,
    ``<partition>_per_class.csv`` and ``confusion_<partition>.png`` when output
    directories are given.
    """
    device = torch.device(device)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = model_from_checkpoint(ckpt, device)
    pre = PreprocessConfig.from_dict(ckpt["preprocess"])
    split = split or SubjectSplit.from_dict(ckpt["split"])
    col = collection or get_collection(ckpt["config"]["dataset"])
    idx = split.indices(col.subjects)[partition]
    if len(idx) == 0:
        raise ValueError(f"partition {partition!r} of split {split.name} is empty")
    ds = SkeletonDataset(col, idx, pre, None)
    y_true, probs, sidx = predict(model, DataLoader(ds, batch_size=64), device)
    y_pred = probs.argmax(1)
    names = ckpt["class_names"]
    m = classification_metrics(y_true, y_pred, ckpt["num_classes"], names)
    m["top_confusions"] = top_confusions(m["confusion_matrix"], names)
    m.update(
        {
            "checkpoint": str(ckpt_path),
            "partition": partition,
            "split": split.to_dict(),
            "subjects": sorted({int(s) for s in col.subjects[idx]}),
            "protocol": "KARD subject-independent evaluation",
            "model": ckpt["model_name"],
            "num_parameters": count_parameters(model),
            "checkpoint_size_bytes": Path(ckpt_path).stat().st_size,
            "model_weights_bytes": int(
                sum(t.numel() * t.element_size() for t in ckpt["model_state"].values())
            ),
            "epoch": ckpt.get("epoch"),
            "seed": ckpt.get("seed"),
        }
    )
    if latency:
        shape = (1, pre.num_frames, ckpt["num_joints"], ckpt["in_channels"])
        m["latency_cpu_1thread"] = measure_latency(model, shape, "cpu", num_threads=1)
        if torch.cuda.is_available():
            m["latency_cuda"] = measure_latency(model, shape, "cuda")
        model.to(device)
    if metrics_dir is not None:
        out = Path(metrics_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{partition}_metrics.json").write_text(json.dumps(m, indent=2))
        pd.DataFrame(
            {
                "seq_id": [col.seq_ids[i] for i in sidx],
                "subject": col.subjects[sidx],
                "true": [names[i] for i in y_true],
                "pred": [names[i] for i in y_pred],
                "confidence": probs.max(1),
                "correct": y_true == y_pred,
            }
        ).to_csv(out / f"{partition}_predictions.csv", index=False)
        pd.DataFrame(m["per_class"]).T.rename_axis("class").to_csv(
            out / f"{partition}_per_class.csv"
        )
    if figures_dir is not None:
        plot_confusion(
            m["confusion_matrix"],
            names,
            Path(figures_dir) / f"confusion_{partition}.png",
            title=f"{ckpt['model_name']} | {partition} subjects {m['subjects']} | "
            f"acc {m['accuracy']:.3f} macro-F1 {m['macro_f1']:.3f}",
        )
    return m
