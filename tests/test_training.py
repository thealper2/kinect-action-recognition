from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from skeleton_action.training.checkpoint import (
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
)
from skeleton_action.training.trainer import train
from skeleton_action.utils.config import resolve_config


def test_train_end_to_end(tmp_path: Path, tiny_cfg: dict) -> None:
    summary = train(tiny_cfg, tmp_path / "out")
    assert summary["epochs_run"] == 3
    ck = load_checkpoint(summary["checkpoint"])
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "class_names",
        "preprocess",
        "split",
        "seed",
        "versions",
        "config",
        "dataset_id",
        "best_metric",
        "epoch",
    ):
        assert key in ck
    assert len(ck["class_names"]) == 18 and ck["split"]["test"] == [4]
    m = tmp_path / "out" / "metrics" / "tiny" / "seed1"
    assert (m / "test_metrics.json").exists() and (m / "history.csv").exists()
    tm = json.loads((m / "test_metrics.json").read_text())
    assert tm["subjects"] == [4] and len(tm["confusion_matrix"]) == 18
    assert tm["num_samples"] == 36
    hist = ck["history"]
    assert all(torch.isfinite(torch.tensor(h["train_loss"])) for h in hist)
    with pytest.raises(FileExistsError):
        train(tiny_cfg, tmp_path / "out")


def test_resume_is_exact(tmp_path: Path, tiny_cfg: dict) -> None:
    full = train(copy.deepcopy(tiny_cfg), tmp_path / "a", evaluate_test=False)
    paused_cfg = copy.deepcopy(tiny_cfg)
    paused_cfg["training"]["stop_after_epoch"] = 1
    paused = train(paused_cfg, tmp_path / "b", evaluate_test=False)
    assert paused["paused_at_epoch"] == 1
    resumed = train(
        copy.deepcopy(tiny_cfg), tmp_path / "b", resume=paused["checkpoint"], evaluate_test=False
    )
    assert resumed["epochs_run"] == full["epochs_run"]
    a = load_checkpoint(tmp_path / "a" / "checkpoints" / "tiny" / "seed1" / "last.pt")
    b = load_checkpoint(tmp_path / "b" / "checkpoints" / "tiny" / "seed1" / "last.pt")
    for k, v in a["model_state"].items():
        assert torch.allclose(v.float(), b["model_state"][k].float(), atol=1e-6), k
    assert a["history"][-1]["val_loss"] == pytest.approx(b["history"][-1]["val_loss"], abs=1e-6)


def test_checkpoint_roundtrip(tmp_path: Path, tiny_cfg: dict) -> None:
    summary = train(tiny_cfg, tmp_path / "o", evaluate_test=False)
    ck = load_checkpoint(summary["checkpoint"])
    model = model_from_checkpoint(ck)
    x = torch.randn(2, 16, 15, 3)
    with torch.no_grad():
        y1 = model(x)
    save_checkpoint(tmp_path / "copy.pt", ck)
    model2 = model_from_checkpoint(load_checkpoint(tmp_path / "copy.pt"))
    with torch.no_grad():
        torch.testing.assert_close(y1, model2(x))
    bad = dict(ck)
    bad.pop("preprocess")
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "bad.pt", bad)


def test_config_composition() -> None:
    cfg = resolve_config(
        "configs/experiments/e4_stgcn_xyz_vel.yaml",
        overrides=["training.epochs=7", "model.params.dropout=0.5"],
    )
    assert cfg["dataset"]["preprocess"]["velocity"] is True
    assert cfg["training"]["epochs"] == 7 and cfg["model"]["params"]["dropout"] == 0.5
    assert cfg["name"] == "e4_stgcn_xyz_vel"
    for exp in (
        "e1_mlp_xyz",
        "e2_lstm_xyz",
        "e3_stgcn_xyz",
        "e5_stgcn_no_norm",
        "e6_stgcn_T32",
        "e6_stgcn_T96",
        "e7_stgcn_perframe_center",
    ):
        c = resolve_config(f"configs/experiments/{exp}.yaml")
        assert c["split"]["test"] == [9, 10]


def test_suite_entries() -> None:
    from skeleton_action.utils.config import load_yaml, suite_entries

    short = suite_entries(load_yaml("configs/experiments/suite.yaml"))
    full = suite_entries(load_yaml("configs/experiments/suite_full.yaml"))
    assert sum(len(s) for _, s in short) == 14
    assert sum(len(s) for _, s in full) == 24
    assert {p for p, _ in short} == {p for p, _ in full}
    assert all(42 in s for _, s in short)  # ablations share seed 42 with e3
    assert all(s == [7] for _, s in suite_entries(load_yaml("configs/experiments/suite.yaml"), [7]))


def test_checkpoint_every(tmp_path: Path, tiny_cfg: dict) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    cfg["training"].update({"epochs": 5, "checkpoint_every": 3})
    train(cfg, tmp_path, evaluate_test=False)
    ck = tmp_path / "checkpoints" / "tiny" / "seed1"
    assert load_checkpoint(ck / "last.pt")["epoch"] == 5  # final epoch always saved
    assert (ck / "best.pt").exists()
