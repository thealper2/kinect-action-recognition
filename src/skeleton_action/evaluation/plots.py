"""Figures: confusion matrices and training curves (matplotlib, non-interactive)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(cm, class_names: list[str], path: str | Path, title: str = "") -> None:
    """Row-normalised confusion matrix with raw counts annotated."""
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.5 * n + 2.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), class_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.04, label="row-normalised rate")
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history: list[dict], path: str | Path, title: str = "") -> None:
    """Loss and macro-F1 curves for train/val."""
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_ylabel("cross-entropy")
    axes[1].plot(ep, [h["train_macro_f1"] for h in history], label="train")
    axes[1].plot(ep, [h["val_macro_f1"] for h in history], label="val")
    axes[1].set_ylabel("macro F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
