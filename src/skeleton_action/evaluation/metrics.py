"""Classification metrics aggregated over all samples of a split."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, class_names: list[str] | None = None
) -> dict:
    """Accuracy, macro P/R/F1, weighted F1, per-class P/R/F1/support and confusion matrix.

    All classes ``0..num_classes-1`` are always reported (zero_division=0), so metrics
    on a split missing some class are still well defined and comparable.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.arange(num_classes)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    names = class_names or [str(i) for i in labels]
    return {
        "num_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0,
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def top_confusions(cm: np.ndarray | list, class_names: list[str], k: int = 10) -> list[dict]:
    """Largest off-diagonal confusion counts, as (true, predicted, count, rate)."""
    cm = np.asarray(cm)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
    out = []
    for i, j in order[:k]:
        if off[i, j] == 0:
            break
        out.append(
            {
                "true": class_names[i],
                "pred": class_names[j],
                "count": int(off[i, j]),
                "rate_of_true": float(off[i, j] / max(cm[i].sum(), 1)),
            }
        )
    return out
