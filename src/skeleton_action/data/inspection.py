"""Dataset inspection report: inventory, completeness, validity and statistics.

All numbers are computed from the files actually present on disk.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from skeleton_action.constants import (
    ACTION_NAMES,
    EXPECTED_SEQUENCES,
    JOINT_NAMES,
    NUM_ACTIONS,
    NUM_REPETITIONS,
    NUM_SUBJECTS,
    J,
)
from skeleton_action.data.kard import (
    KardParseError,
    KardSampleId,
    discover,
    parse_skeleton_file,
)


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    return {
        "min": float(x.min()),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
        "std": float(x.std()),
    }


def build_report(root: str | Path, kind: str = "realworld") -> dict:
    """Inspect every skeleton file under ``root`` and return a JSON-serialisable report."""
    root = Path(root)
    disc = discover(root)
    files = disc.of_kind(kind)
    other_kind = "screen" if kind == "realworld" else "realworld"
    other_files = disc.of_kind(other_kind)

    # Inventory per directory.
    inventory: dict[str, Counter] = {}
    for (_sid, k), f in disc.files.items():
        d = str(f.path.parent.relative_to(root)) if f.path.parent != root else "."
        inventory.setdefault(d, Counter())[k] += 1

    expected = {
        KardSampleId(a, s, e)
        for a in range(1, NUM_ACTIONS + 1)
        for s in range(1, NUM_SUBJECTS + 1)
        for e in range(1, NUM_REPETITIONS + 1)
    }
    missing = sorted(expected - set(files))

    valid: dict[KardSampleId, np.ndarray] = {}
    malformed: list[dict] = []
    nonfinite_total = 0
    zero_joints_total = 0
    zero_joints_per_joint = np.zeros(len(JOINT_NAMES), dtype=np.int64)
    for sid, f in files.items():
        try:
            parsed = parse_skeleton_file(f.path)
        except KardParseError as exc:
            malformed.append({"seq_id": sid.seq_id, "path": str(f.path), "reason": exc.reason})
            continue
        nonfinite_total += parsed.n_nonfinite
        zero_joints_total += parsed.n_zero_joints
        zero_joints_per_joint += np.all(parsed.coords == 0, axis=-1).sum(axis=0)
        if parsed.n_nonfinite:
            malformed.append(
                {
                    "seq_id": sid.seq_id,
                    "path": str(f.path),
                    "reason": f"{parsed.n_nonfinite} non-finite values",
                }
            )
            continue
        valid[sid] = parsed.coords

    per_action = Counter(sid.action for sid in valid)
    per_subject = Counter(sid.subject for sid in valid)
    lengths = np.array([v.shape[0] for v in valid.values()])
    lengths_by_action = {
        ACTION_NAMES[a]: _stats([v.shape[0] for s, v in valid.items() if s.action == a])
        for a in range(1, NUM_ACTIONS + 1)
        if per_action.get(a)
    }

    report: dict = {
        "root": str(root),
        "coordinate_kind": kind,
        "expected_sequences": EXPECTED_SEQUENCES,
        "files_found": len(files),
        "valid_sequences": len(valid),
        "malformed_sequences": malformed,
        "missing_sequences": [s.seq_id for s in missing],
        "duplicates": disc.duplicates,
        "unrecognized_files": disc.unrecognized,
        "bad_filenames": disc.bad_names,
        "inventory": {d: dict(c) for d, c in sorted(inventory.items())},
        "examples_per_action": {
            f"{a:02d} {ACTION_NAMES[a]}": per_action.get(a, 0) for a in range(1, NUM_ACTIONS + 1)
        },
        "examples_per_subject": {
            f"s{s:02d}": per_subject.get(s, 0) for s in range(1, NUM_SUBJECTS + 1)
        },
        "frames": {
            "total": int(lengths.sum()) if lengths.size else 0,
            "per_sequence": _stats(lengths),
            "per_action": lengths_by_action,
        },
        "invalid_coordinates": {
            "nonfinite_values": int(nonfinite_total),
            "all_zero_joint_entries": int(zero_joints_total),
            "all_zero_entries_per_joint": dict(
                zip(JOINT_NAMES, zero_joints_per_joint.tolist(), strict=True)
            ),
        },
    }
    if valid:
        allc = np.concatenate(list(valid.values()), axis=0)  # [sumT, 15, 3]
        report["coordinate_ranges"] = {
            ax: _stats(allc[..., i].ravel()) for i, ax in enumerate("xyz")
        }
        report["coordinate_ranges_per_joint"] = {
            name: {
                ax: [float(allc[:, j, i].min()), float(allc[:, j, i].max())]
                for i, ax in enumerate("xyz")
            }
            for j, name in enumerate(JOINT_NAMES)
        }
        if kind == "realworld":
            report["geometry_checks"] = _geometry_checks(allc)
        # Cross-check frame counts with the other coordinate kind where available.
        mismatch = []
        for sid, arr in valid.items():
            if sid in other_files:
                try:
                    o = parse_skeleton_file(other_files[sid].path).coords
                except KardParseError:
                    continue
                if o.shape[0] != arr.shape[0]:
                    mismatch.append(
                        {"seq_id": sid.seq_id, kind: arr.shape[0], other_kind: o.shape[0]}
                    )
        report[f"frame_count_mismatch_vs_{other_kind}"] = mismatch
        report[f"{other_kind}_files_found"] = len(other_files)
    return report


def _geometry_checks(allc: np.ndarray) -> dict:
    """Sanity statistics used to document coordinate conventions (real-world, mm)."""
    ok = np.any(allc != 0, axis=-1)  # [N, 15]

    def frac(cond: np.ndarray, *joints: int) -> float:
        m = np.all(ok[:, list(joints)], axis=1)
        return float(cond[m].mean()) if m.any() else float("nan")

    def bone(a: int, b: int) -> np.ndarray:
        m = ok[:, a] & ok[:, b]
        return np.linalg.norm(allc[m, a] - allc[m, b], axis=-1)

    thigh = np.concatenate([bone(J.R_HIP, J.R_KNEE), bone(J.L_HIP, J.L_KNEE)])
    shank = np.concatenate([bone(J.R_KNEE, J.R_FOOT), bone(J.L_KNEE, J.L_FOOT)])
    upper = np.concatenate([bone(J.R_SHOULDER, J.R_ELBOW), bone(J.L_SHOULDER, J.L_ELBOW)])
    fore = np.concatenate([bone(J.R_ELBOW, J.R_HAND), bone(J.L_ELBOW, J.L_HAND)])
    return {
        "frac_head_above_torso_y": frac(allc[:, J.HEAD, 1] > allc[:, J.TORSO, 1], J.HEAD, J.TORSO),
        "frac_right_shoulder_x_less_than_left": frac(
            allc[:, J.R_SHOULDER, 0] < allc[:, J.L_SHOULDER, 0], J.R_SHOULDER, J.L_SHOULDER
        ),
        "frac_right_hip_x_less_than_left": frac(
            allc[:, J.R_HIP, 0] < allc[:, J.L_HIP, 0], J.R_HIP, J.L_HIP
        ),
        "torso_depth_z_mm": _stats(allc[ok[:, J.TORSO], J.TORSO, 2]),
        "bone_length_mm": {
            "neck_torso": _stats(bone(J.NECK, J.TORSO)),
            "shoulder_width": _stats(bone(J.R_SHOULDER, J.L_SHOULDER)),
            "hip_width": _stats(bone(J.R_HIP, J.L_HIP)),
            "upper_arm": _stats(upper),
            "forearm_elbow_to_hand": _stats(fore),
            "thigh_hip_to_knee": _stats(thigh),
            "shank_knee_to_foot": _stats(shank),
        },
    }


def report_to_markdown(r: dict) -> str:
    """Human-readable summary of :func:`build_report` output."""
    lines = [
        f"# KARD inspection report ({r['coordinate_kind']})",
        "",
        f"- Root: `{r['root']}`",
        f"- Files found: **{r['files_found']}** (expected {r['expected_sequences']})",
        f"- Valid sequences: **{r['valid_sequences']}**",
        f"- Malformed: {len(r['malformed_sequences'])}, missing: "
        f"{len(r['missing_sequences'])}, duplicates: {len(r['duplicates'])}, "
        f"bad filenames: {len(r['bad_filenames'])}",
        "",
        "## Inventory",
        "",
        "| directory | realworld | screen |",
        "|---|---:|---:|",
    ]
    for d, c in r["inventory"].items():
        lines.append(f"| {d} | {c.get('realworld', 0)} | {c.get('screen', 0)} |")
    lines += ["", "## Examples per action", "", "| action | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in r["examples_per_action"].items()]
    lines += ["", "## Examples per subject", "", "| subject | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in r["examples_per_subject"].items()]
    f = r["frames"]
    if f["per_sequence"]:
        ps = f["per_sequence"]
        lines += [
            "",
            "## Sequence length (frames)",
            "",
            f"Total frames: {f['total']}. min {ps['min']:.0f}, p05 {ps['p05']:.0f}, "
            f"median {ps['median']:.0f}, mean {ps['mean']:.1f}, p95 {ps['p95']:.0f}, "
            f"max {ps['max']:.0f}",
            "",
            "| action | min | median | max |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| {a} | {s['min']:.0f} | {s['median']:.0f} | {s['max']:.0f} |"
            for a, s in f["per_action"].items()
        ]
    inv = r["invalid_coordinates"]
    lines += [
        "",
        "## Invalid coordinates",
        "",
        f"- Non-finite values: {inv['nonfinite_values']}",
        f"- All-zero (untracked) joint entries: {inv['all_zero_joint_entries']}",
    ]
    if "coordinate_ranges" in r:
        lines += [
            "",
            "## Coordinate ranges",
            "",
            "| axis | min | p05 | median | p95 | max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for ax, s in r["coordinate_ranges"].items():
            lines.append(
                f"| {ax} | {s['min']:.1f} | {s['p05']:.1f} | {s['median']:.1f} | "
                f"{s['p95']:.1f} | {s['max']:.1f} |"
            )
    if "geometry_checks" in r:
        g = r["geometry_checks"]
        lines += ["", "## Geometry checks (real-world)", ""]
        for k, v in g.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
        lines.append(
            f"- torso depth z (mm): median {g['torso_depth_z_mm']['median']:.0f}, "
            f"range {g['torso_depth_z_mm']['min']:.0f}..{g['torso_depth_z_mm']['max']:.0f}"
        )
        lines += ["", "| bone | median mm | p05 | p95 |", "|---|---:|---:|---:|"]
        for k, s in g["bone_length_mm"].items():
            lines.append(f"| {k} | {s['median']:.0f} | {s['p05']:.0f} | {s['p95']:.0f} |")
    for key in ("malformed_sequences", "missing_sequences", "duplicates", "bad_filenames"):
        if r[key]:
            lines += ["", f"## {key}", ""] + [f"- {x}" for x in r[key][:200]]
    mm = [k for k in r if k.startswith("frame_count_mismatch")]
    for k in mm:
        lines += ["", f"- {k}: {len(r[k])} sequences"]
    return "\n".join(lines) + "\n"
