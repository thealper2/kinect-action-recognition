from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skeleton_action.constants import CLASS_NAMES, NUM_JOINTS, action_id_to_class
from skeleton_action.data.kard import (
    KardCollection,
    KardParseError,
    KardSampleId,
    discover,
    load_collection,
    parse_filename,
    parse_skeleton_file,
    parse_skeleton_text,
)


def encoded_text(frames: int) -> str:
    """Row r = (frame f, joint j) holds values (1000f + 10j + c) for c = 0, 1, 2."""
    lines = []
    for f in range(frames):
        for j in range(NUM_JOINTS):
            lines.append(" ".join(str(1000 * f + 10 * j + c) for c in range(3)))
    return "\n".join(lines) + "\n"


def test_filename_parsing() -> None:
    sid, kind = parse_filename("a07_s03_e02_realworld.txt")
    assert (sid.action, sid.subject, sid.repetition, kind) == (7, 3, 2, "realworld")
    assert sid.seq_id == "a07_s03_e02"
    assert sid.label == 6
    sid, kind = parse_filename(Path("x/y/a18_s10_e03_screen.txt"))
    assert (sid.action, sid.subject, sid.repetition, kind) == (18, 10, 3, "screen")


@pytest.mark.parametrize(
    "name",
    [
        "a19_s01_e01_realworld.txt",
        "a00_s01_e01_realworld.txt",
        "a01_s11_e01_realworld.txt",
        "a01_s01_e04_realworld.txt",
        "a1_s01_e01_realworld.txt",
        "a01_s01_e01_depth.txt",
        "a01_s01_e01.mp4",
    ],
)
def test_filename_rejects(name: str) -> None:
    with pytest.raises(KardParseError):
        parse_filename(name)


def test_class_mapping() -> None:
    assert len(CLASS_NAMES) == 18
    assert CLASS_NAMES[0] == "Horizontal arm wave"
    assert CLASS_NAMES[16] == "Sit down" and CLASS_NAMES[17] == "Stand up"
    assert action_id_to_class(14) == 13 and CLASS_NAMES[13] == "Walk"
    with pytest.raises(ValueError):
        action_id_to_class(19)


def test_frame_joint_ordering() -> None:
    p = parse_skeleton_text(encoded_text(4))
    assert p.coords.shape == (4, NUM_JOINTS, 3)
    assert p.coords.dtype == np.float32
    for f in range(4):
        for j in range(NUM_JOINTS):
            np.testing.assert_array_equal(p.coords[f, j], [1000 * f + 10 * j + c for c in range(3)])


def test_rejects_partial_frame() -> None:
    text = encoded_text(2) + "1 2 3\n"
    with pytest.raises(KardParseError, match="multiple of 15"):
        parse_skeleton_text(text)


def test_rejects_non_numeric_and_bad_columns() -> None:
    lines = encoded_text(1).splitlines()
    bad = lines.copy()
    bad[3] = "1.0 abc 2.0"
    with pytest.raises(KardParseError, match="non-numeric"):
        parse_skeleton_text("\n".join(bad))
    bad = lines.copy()
    bad[5] = "1.0 2.0"
    with pytest.raises(KardParseError, match="expected 3 values"):
        parse_skeleton_text("\n".join(bad))
    with pytest.raises(KardParseError, match="no coordinate rows"):
        parse_skeleton_text("\n\n")


def test_nonfinite_and_zero_joints_counted() -> None:
    lines = encoded_text(1).splitlines()
    lines[2] = "nan 1 2"
    lines[4] = "0 0 0"
    p = parse_skeleton_text("\n".join(lines))
    assert p.n_nonfinite == 1
    assert p.n_zero_joints == 1
    assert not p.valid_mask[0, 2] and not p.valid_mask[0, 4] and p.valid_mask[0, 3]


def test_discover_and_load_collection(tmp_path: Path) -> None:
    d1 = tmp_path / "KARD_a01_a02"
    d2 = tmp_path / "nested" / "copy"
    d1.mkdir()
    d2.mkdir(parents=True)
    (d1 / "a01_s01_e01_realworld.txt").write_text(encoded_text(3))
    (d1 / "a02_s02_e01_realworld.txt").write_text(encoded_text(5))
    (d1 / "a02_s02_e01_screen.txt").write_text(encoded_text(5))
    (d2 / "a01_s01_e01_realworld.txt").write_text(encoded_text(3))  # identical duplicate
    (d1 / "a03_s01_e01_realworld.txt").write_text(encoded_text(1) + "1 2 3\n")  # corrupt
    (d1 / "a99_s01_e01_realworld.txt").write_text(encoded_text(1))  # bad id
    disc = discover(tmp_path)
    assert len(disc.of_kind("realworld")) == 3
    assert len(disc.duplicates) == 1 and disc.duplicates[0]["identical"]
    assert len(disc.bad_names) == 1
    col, rejected = load_collection(tmp_path, "realworld")
    assert col.seq_ids == ["a01_s01_e01", "a02_s02_e01"]
    assert [r["seq_id"] for r in rejected] == ["a03_s01_e01"]
    assert col.lengths.tolist() == [3, 5]
    np.testing.assert_array_equal(col.labels, [0, 1])
    np.testing.assert_array_equal(col.sequence(1)[2, 7], [2070, 2071, 2072])


def test_conflicting_duplicates_dropped(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "a01_s01_e01_realworld.txt").write_text(encoded_text(2))
    (tmp_path / "b" / "a01_s01_e01_realworld.txt").write_text(encoded_text(3))
    disc = discover(tmp_path)
    assert not disc.of_kind("realworld")
    assert disc.duplicates and not disc.duplicates[0]["identical"]


def test_collection_roundtrip(tmp_path: Path, tiny_collection: KardCollection) -> None:
    path = tmp_path / "c.npz"
    tiny_collection.save(path)
    c2 = KardCollection.load(path)
    assert c2.seq_ids == tiny_collection.seq_ids
    np.testing.assert_array_equal(c2.coords, tiny_collection.coords)
    np.testing.assert_array_equal(c2.offsets, tiny_collection.offsets)
    assert c2.kind == "realworld"
    assert c2.index_of("a05_s02_e01") == tiny_collection.index_of("a05_s02_e01")


def test_parse_file(tmp_path: Path) -> None:
    f = tmp_path / "a01_s01_e01_realworld.txt"
    f.write_text(encoded_text(2))
    assert parse_skeleton_file(f).num_frames == 2
    assert KardSampleId(1, 1, 1).seq_id == "a01_s01_e01"
