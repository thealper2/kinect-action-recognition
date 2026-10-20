"""Official KARD facts: action IDs, joint order and skeleton topology.

Sources: the KARD v2 Mendeley Data page (https://doi.org/10.17632/k28dtm7tr6.2) and the
``loadJoints.m`` script distributed with the dataset. Everything in this module is a
dataset fact, except ``KARD_EDGES`` (an implementation choice, see docs/architecture.md).
"""

from __future__ import annotations

NUM_ACTIONS = 18
NUM_SUBJECTS = 10
NUM_REPETITIONS = 3
NUM_JOINTS = 15
NUM_COORDS = 3
EXPECTED_SEQUENCES = NUM_ACTIONS * NUM_SUBJECTS * NUM_REPETITIONS  # 540

# Official action IDs (1-based, as in the filenames aXX) -> human-readable label.
ACTION_NAMES: dict[int, str] = {
    1: "Horizontal arm wave",
    2: "High arm wave",
    3: "Two hand wave",
    4: "Catch cap",
    5: "High throw",
    6: "Draw X",
    7: "Draw tick",
    8: "Toss paper",
    9: "Forward kick",
    10: "Side kick",
    11: "Take umbrella",
    12: "Bend",
    13: "Hand clap",
    14: "Walk",
    15: "Phone call",
    16: "Drink",
    17: "Sit down",
    18: "Stand up",
}

# Model class index c (0-based) <-> official action id c + 1.
CLASS_NAMES: list[str] = [ACTION_NAMES[i] for i in range(1, NUM_ACTIONS + 1)]


def action_id_to_class(action_id: int) -> int:
    """Map an official 1-based action id to a 0-based class index."""
    if action_id not in ACTION_NAMES:
        raise ValueError(f"Unknown KARD action id: {action_id}")
    return action_id - 1


def class_to_action_id(class_index: int) -> int:
    """Map a 0-based class index back to the official 1-based action id."""
    if not 0 <= class_index < NUM_ACTIONS:
        raise ValueError(f"Class index out of range: {class_index}")
    return class_index + 1


# Official joint order of every 15-row frame block in *_realworld.txt / *_screen.txt.
JOINT_NAMES: list[str] = [
    "Head",
    "Neck",
    "Right Shoulder",
    "Right Elbow",
    "Right Hand",
    "Left Shoulder",
    "Left Elbow",
    "Left Hand",
    "Torso",
    "Right Hip",
    "Right Knee",
    "Right Foot",
    "Left Hip",
    "Left Knee",
    "Left Foot",
]


class J:
    """0-based joint indices (KARD order)."""

    HEAD = 0
    NECK = 1
    R_SHOULDER = 2
    R_ELBOW = 3
    R_HAND = 4
    L_SHOULDER = 5
    L_ELBOW = 6
    L_HAND = 7
    TORSO = 8
    R_HIP = 9
    R_KNEE = 10
    R_FOOT = 11
    L_HIP = 12
    L_KNEE = 13
    L_FOOT = 14


# Anatomical bones of the 15-joint OpenNI/NiTE-style skeleton (undirected tree, 14 edges).
# Implementation choice: we keep the kinematic tree only (no hip-hip / shoulder-torso
# shortcut edges); see docs/architecture.md.
KARD_EDGES: list[tuple[int, int]] = [
    (J.HEAD, J.NECK),
    (J.NECK, J.R_SHOULDER),
    (J.R_SHOULDER, J.R_ELBOW),
    (J.R_ELBOW, J.R_HAND),
    (J.NECK, J.L_SHOULDER),
    (J.L_SHOULDER, J.L_ELBOW),
    (J.L_ELBOW, J.L_HAND),
    (J.NECK, J.TORSO),
    (J.TORSO, J.R_HIP),
    (J.R_HIP, J.R_KNEE),
    (J.R_KNEE, J.R_FOOT),
    (J.TORSO, J.L_HIP),
    (J.L_HIP, J.L_KNEE),
    (J.L_KNEE, J.L_FOOT),
]

# Left/right joint pairs (used e.g. for handedness diagnostics).
LEFT_RIGHT_PAIRS: list[tuple[int, int]] = [
    (J.L_SHOULDER, J.R_SHOULDER),
    (J.L_ELBOW, J.R_ELBOW),
    (J.L_HAND, J.R_HAND),
    (J.L_HIP, J.R_HIP),
    (J.L_KNEE, J.R_KNEE),
    (J.L_FOOT, J.R_FOOT),
]

DATASET_ID = "KARD-v2 (Mendeley Data, doi:10.17632/k28dtm7tr6.2)"
DATASET_CITATION = (
    "Morana, M., Lo Re, G., & Gaglio, S. (2022). KARD - Kinect Activity Recognition "
    "Dataset, Version 2. Mendeley Data. https://doi.org/10.17632/k28dtm7tr6.2"
)
DATASET_LICENSE = "CC BY 4.0"
