"""YAML experiment configuration: composition from component files and dotted overrides.

An experiment file looks like::

    name: stgcn_xyz
    dataset: configs/dataset/kard_realworld.yaml   # path or inline mapping
    split: configs/dataset/split_subject_fixed.yaml
    model: configs/model/stgcn.yaml
    training: configs/training/default.yaml
    overrides:                                     # optional dotted keys
      dataset.preprocess.velocity: true

The resolved configuration is a plain dict with keys
``name, dataset, split, model, training`` and is stored verbatim in every checkpoint.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

SECTIONS = ("dataset", "split", "model", "training")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """``a.b.c=value`` with the value parsed as YAML (numbers, bools, lists...)."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like key=value")
    k, v = text.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def resolve_config(
    experiment: str | Path | dict | None = None,
    *,
    dataset: str | Path | None = None,
    split: str | Path | None = None,
    model: str | Path | None = None,
    training: str | Path | None = None,
    overrides: list[str] | None = None,
    name: str | None = None,
) -> dict:
    """Compose the full configuration.

    Component paths given explicitly take precedence over the experiment file;
    command-line ``overrides`` are applied last.
    """
    exp: dict = {}
    if isinstance(experiment, dict):
        exp = copy.deepcopy(experiment)
    elif experiment is not None:
        exp = load_yaml(experiment)
    explicit = {"dataset": dataset, "split": split, "model": model, "training": training}
    cfg: dict = {"name": name or exp.get("name", "experiment")}
    for sec in SECTIONS:
        src = explicit[sec] if explicit[sec] is not None else exp.get(sec)
        if src is None:
            raise ValueError(f"missing configuration section {sec!r}")
        cfg[sec] = load_yaml(src) if isinstance(src, str | Path) else copy.deepcopy(src)
    for k, v in (exp.get("overrides") or {}).items():
        set_dotted(cfg, k, v)
    for text in overrides or []:
        k, v = parse_override(text)
        set_dotted(cfg, k, v)
    if name:
        cfg["name"] = name
    return cfg


def suite_entries(suite: dict, seeds_override: list[int] | None = None) -> list[tuple[str, list]]:
    """Expand a suite file into ``(experiment path, seeds)`` pairs.

    Entries are either a path (uses the suite-level ``seeds``) or
    ``{config: path, seeds: [...]}``. ``seeds_override`` replaces every seed list.
    """
    default = list(suite.get("seeds", [42]))
    out = []
    for e in suite.get("experiments", []):
        if isinstance(e, dict):
            path, seeds = e["config"], list(e.get("seeds", default))
        else:
            path, seeds = e, default
        out.append((path, list(seeds_override) if seeds_override else seeds))
    return out


def dump_yaml(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
