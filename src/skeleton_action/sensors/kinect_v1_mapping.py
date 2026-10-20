"""Kinect for Windows SDK v1.x (Xbox 360 Kinect) skeleton -> KARD 15-joint representation.

Source skeleton: ``NUI_SKELETON_POSITION_INDEX`` (20 joints) from NuiSensor.h / NuiApi.h
of the Kinect for Windows SDK 1.8. Source coordinates ("skeleton space"): metres,
right-handed, origin at the IR camera, +Y up, +Z pointing out of the sensor into the
scene, +X towards the *sensor's left* (= towards the right hand of a user facing it).

Target: KARD real-world coordinates (OpenNI-style): millimetres, +X to the right of the
sensor image, +Y up, +Z = depth away from the sensor. Verified on the KARD files: the
"Right Shoulder"/"Right Hip" have a smaller X than their left counterparts in 98.3 % of
frames, and the "Right" hand/foot is the moving limb in every one-sided action except
"Toss paper" (see docs/kinect_setup.md). Hence::

    x_kard = -1000 * x_sdk     y_kard = 1000 * y_sdk     z_kard = 1000 * z_sdk

and SDK *Right joints map to KARD Right joints (both describe the anatomical side).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skeleton_action.constants import NUM_JOINTS, J

# NUI_SKELETON_POSITION_INDEX (SDK v1.x)
SDK_JOINTS: list[str] = [
    "HipCenter",
    "Spine",
    "ShoulderCenter",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
]
SDK = {name: i for i, name in enumerate(SDK_JOINTS)}
NUM_SDK_JOINTS = 20

# NUI_SKELETON_POSITION_TRACKING_STATE
POS_NOT_TRACKED, POS_INFERRED, POS_TRACKED = 0, 1, 2


@dataclass(frozen=True)
class JointMapping:
    target: int
    sources: tuple[str, ...]  # averaged when several
    exact: bool
    note: str


def build_mapping(
    hand: str = "hand", foot: str = "ankle", torso: str = "spine"
) -> list[JointMapping]:
    """Mapping table. Options choose between anatomically close SDK joints.

    Args:
        hand: ``hand`` (HandX, default) or ``wrist`` (WristX).
        foot: ``ankle`` (AnkleX, default) or ``foot`` (FootX). KARD shank length
            knee->foot (median 412 mm) is close to the thigh length (440 mm), i.e. the KARD
            "Foot" joint sits near the ankle, not at the toes.
        torso: ``spine`` (Spine, default) or ``spine_shoulder_mid`` (mean of Spine and
            ShoulderCenter).
    """
    if hand not in ("hand", "wrist") or foot not in ("ankle", "foot"):
        raise ValueError("invalid hand/foot option")
    if torso not in ("spine", "spine_shoulder_mid"):
        raise ValueError("invalid torso option")
    h = "Hand" if hand == "hand" else "Wrist"
    f = "Ankle" if foot == "ankle" else "Foot"
    torso_src = ("Spine",) if torso == "spine" else ("Spine", "ShoulderCenter")
    return [
        JointMapping(J.HEAD, ("Head",), True, "same semantic joint"),
        JointMapping(
            J.NECK,
            ("ShoulderCenter",),
            False,
            "SDK ShoulderCenter (between the shoulders) approximates OpenNI Neck",
        ),
        JointMapping(J.R_SHOULDER, ("ShoulderRight",), True, ""),
        JointMapping(J.R_ELBOW, ("ElbowRight",), True, ""),
        JointMapping(J.R_HAND, (f"{h}Right",), hand == "hand", f"SDK {h}Right"),
        JointMapping(J.L_SHOULDER, ("ShoulderLeft",), True, ""),
        JointMapping(J.L_ELBOW, ("ElbowLeft",), True, ""),
        JointMapping(J.L_HAND, (f"{h}Left",), hand == "hand", f"SDK {h}Left"),
        JointMapping(
            J.TORSO,
            torso_src,
            False,
            "OpenNI Torso is the torso centre; SDK Spine sits lower (mid-back)",
        ),
        JointMapping(J.R_HIP, ("HipRight",), True, ""),
        JointMapping(J.R_KNEE, ("KneeRight",), True, ""),
        JointMapping(J.R_FOOT, (f"{f}Right",), False, f"KARD Foot ~ ankle level; SDK {f}Right"),
        JointMapping(J.L_HIP, ("HipLeft",), True, ""),
        JointMapping(J.L_KNEE, ("KneeLeft",), True, ""),
        JointMapping(J.L_FOOT, (f"{f}Left",), False, f"KARD Foot ~ ankle level; SDK {f}Left"),
    ]


@dataclass
class KinectV1Mapper:
    """Convert one SDK v1 skeleton to KARD joints.

    Args:
        accept_inferred: treat INFERRED SDK joints as valid (they are SDK estimates of
            occluded joints). If False they are marked missing and later interpolated.
        flip_x: apply the SDK -> KARD X-axis flip (keep True unless the source already
            uses the OpenNI convention).
        swap_left_right: mirror the person (swap L/R joints and negate X) at test time,
            e.g. for left-handed users; KARD subjects used their right hand.
    """

    hand: str = "hand"
    foot: str = "ankle"
    torso: str = "spine"
    accept_inferred: bool = True
    flip_x: bool = True
    swap_left_right: bool = False

    def __post_init__(self) -> None:
        self.table = build_mapping(self.hand, self.foot, self.torso)
        if sorted(m.target for m in self.table) != list(range(NUM_JOINTS)):
            raise AssertionError("mapping must cover every KARD joint exactly once")

    def convert(
        self, positions_m: np.ndarray, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Map SDK joints to KARD.

        Args:
            positions_m: ``[20, 3]`` SDK skeleton-space positions in metres.
            states: ``[20]`` NUI_SKELETON_POSITION_TRACKING_STATE values.

        Returns:
            (joints ``[15, 3]`` float32 mm KARD convention, valid ``[15]`` bool, info dict)
        """
        positions_m = np.asarray(positions_m, dtype=np.float64)
        states = np.asarray(states)
        if positions_m.shape != (NUM_SDK_JOINTS, 3) or states.shape != (NUM_SDK_JOINTS,):
            raise ValueError("expected positions [20, 3] and states [20]")
        ok_state = (states == POS_TRACKED) | (self.accept_inferred & (states == POS_INFERRED))
        ok = ok_state & np.isfinite(positions_m).all(axis=1)
        out = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
        valid = np.zeros(NUM_JOINTS, dtype=bool)
        n_inferred = 0
        for m in self.table:
            idx = [SDK[s] for s in m.sources]
            if all(ok[i] for i in idx):
                out[m.target] = positions_m[idx].mean(axis=0)
                valid[m.target] = True
                n_inferred += int(any(states[i] == POS_INFERRED for i in idx))
        out *= 1000.0
        if self.flip_x:
            out[:, 0] *= -1.0
        if self.swap_left_right:
            out, valid = mirror_kard(out, valid)
        out[~valid] = 0.0
        return out.astype(np.float32), valid, {"inferred_mapped": n_inferred}


_LR = [
    (J.R_SHOULDER, J.L_SHOULDER),
    (J.R_ELBOW, J.L_ELBOW),
    (J.R_HAND, J.L_HAND),
    (J.R_HIP, J.L_HIP),
    (J.R_KNEE, J.L_KNEE),
    (J.R_FOOT, J.L_FOOT),
]


def mirror_kard(joints: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a KARD skeleton about the sagittal plane: negate X and swap L/R labels."""
    perm = np.arange(NUM_JOINTS)
    for r, left in _LR:
        perm[r], perm[left] = left, r
    j = joints[perm].copy()
    j[:, 0] *= -1.0
    return j, valid[perm].copy()


def mapping_table_markdown() -> str:
    rows = ["| KARD joint | SDK v1 source | exact | note |", "|---|---|---|---|"]
    from skeleton_action.constants import JOINT_NAMES

    for m in sorted(build_mapping(), key=lambda m: m.target):
        rows.append(
            f"| {JOINT_NAMES[m.target]} | {' + '.join(m.sources)} | "
            f"{'yes' if m.exact else 'approx.'} | {m.note} |"
        )
    return "\n".join(rows)
