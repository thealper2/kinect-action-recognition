"""KARD file discovery, filename parsing and skeleton-file parsing.

Parsing convention (verified against ``loadJoints.m`` shipped with KARD v2 and by
inspecting the files, see docs/dataset_format.md):

* A skeleton file is a whitespace-separated text file with one ``x y z`` triple per line.
* Lines are grouped in consecutive blocks of 15 lines; block ``f`` is frame ``f`` and
  line ``j`` inside the block is joint ``j`` in the official KARD joint order.
  i.e. row ``r`` -> (frame ``r // 15``, joint ``r % 15``), so a C-order reshape of the
  ``[15 * F, 3]`` row matrix to ``[F, 15, 3]`` is exact.
* ``*_realworld.txt``: millimetres, X to the right of the sensor image, Y up, Z = depth.
* ``*_screen.txt``: pixel u, pixel v (down), depth in millimetres.

Corrupted files are rejected with an explicit reason, never silently reshaped.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skeleton_action.constants import (
    NUM_ACTIONS,
    NUM_COORDS,
    NUM_JOINTS,
    NUM_REPETITIONS,
    NUM_SUBJECTS,
    action_id_to_class,
)

FILENAME_RE = re.compile(r"^a(\d{2})_s(\d{2})_e(\d{2})_(realworld|screen)\.txt$")
COORD_KINDS = ("realworld", "screen")


class KardParseError(ValueError):
    """Raised when a KARD skeleton file or filename is malformed."""

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = str(path)
        self.reason = reason
        super().__init__(f"{self.path}: {reason}")


@dataclass(frozen=True, order=True)
class KardSampleId:
    """Identifier of one KARD recording: action, subject and repetition (all 1-based)."""

    action: int
    subject: int
    repetition: int

    @property
    def seq_id(self) -> str:
        return f"a{self.action:02d}_s{self.subject:02d}_e{self.repetition:02d}"

    @property
    def label(self) -> int:
        """0-based class index."""
        return action_id_to_class(self.action)


def parse_filename(name: str | Path) -> tuple[KardSampleId, str]:
    """Parse ``aAA_sSS_eEE_<kind>.txt`` and validate the ID ranges.

    Returns:
        (sample id, coordinate kind) where kind is ``"realworld"`` or ``"screen"``.

    Raises:
        KardParseError: if the name does not match or an ID is out of range.
    """
    base = Path(name).name
    m = FILENAME_RE.match(base)
    if m is None:
        raise KardParseError(base, "filename does not match aAA_sSS_eEE_<realworld|screen>.txt")
    a, s, e = (int(m.group(i)) for i in (1, 2, 3))
    if not 1 <= a <= NUM_ACTIONS:
        raise KardParseError(base, f"action id {a} outside 1..{NUM_ACTIONS}")
    if not 1 <= s <= NUM_SUBJECTS:
        raise KardParseError(base, f"subject id {s} outside 1..{NUM_SUBJECTS}")
    if not 1 <= e <= NUM_REPETITIONS:
        raise KardParseError(base, f"repetition {e} outside 1..{NUM_REPETITIONS}")
    return KardSampleId(a, s, e), m.group(4)


@dataclass
class ParsedSkeleton:
    """A parsed skeleton file.

    Attributes:
        coords: float32 array ``[T, 15, 3]`` exactly as stored in the file.
        n_nonfinite: number of non-finite coordinate values (NaN/inf) in the file.
        n_zero_joints: number of (frame, joint) entries equal to (0, 0, 0), which
            OpenNI/NiTE emits for untracked joints.
    """

    coords: np.ndarray
    n_nonfinite: int
    n_zero_joints: int

    @property
    def num_frames(self) -> int:
        return int(self.coords.shape[0])

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean ``[T, 15]``: joint is finite and not the all-zero placeholder."""
        finite = np.isfinite(self.coords).all(axis=-1)
        nonzero = np.any(self.coords != 0.0, axis=-1)
        return finite & nonzero


def parse_skeleton_text(text: str, source: str = "<string>") -> ParsedSkeleton:
    """Parse the content of a KARD skeleton file into ``[T, 15, 3]``.

    Args:
        text: file content.
        source: name used in error messages.

    Raises:
        KardParseError: on non-numeric tokens, rows without exactly 3 values, an empty
            file, or a row count that is not a multiple of 15.
    """
    rows: list[list[float]] = []
    n_nonfinite = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != NUM_COORDS:
            raise KardParseError(
                source, f"line {lineno}: expected {NUM_COORDS} values, got {len(parts)}"
            )
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            raise KardParseError(source, f"line {lineno}: non-numeric value {parts!r}") from None
        n_nonfinite += sum(1 for v in vals if not math.isfinite(v))
        rows.append(vals)

    if not rows:
        raise KardParseError(source, "file contains no coordinate rows")
    if len(rows) % NUM_JOINTS != 0:
        raise KardParseError(
            source,
            f"{len(rows)} coordinate rows is not a multiple of {NUM_JOINTS} joints",
        )
    arr = np.asarray(rows, dtype=np.float64)
    # Row r -> frame r // 15, joint r % 15  (frame-major, joint-minor). C-order reshape.
    coords = arr.reshape(len(rows) // NUM_JOINTS, NUM_JOINTS, NUM_COORDS).astype(np.float32)
    n_zero = int(np.all(coords == 0.0, axis=-1).sum())
    return ParsedSkeleton(coords=coords, n_nonfinite=n_nonfinite, n_zero_joints=n_zero)


def parse_skeleton_file(path: str | Path) -> ParsedSkeleton:
    """Read and parse one KARD skeleton text file (see :func:`parse_skeleton_text`)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError:
        raise KardParseError(path, "file is not ASCII text") from None
    return parse_skeleton_text(text, source=str(path))


@dataclass
class DiscoveredFile:
    path: Path
    sample_id: KardSampleId
    kind: str
    sha256: str


@dataclass
class Discovery:
    """Result of scanning a raw-data directory."""

    files: dict[tuple[KardSampleId, str], DiscoveredFile] = field(default_factory=dict)
    duplicates: list[dict] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)
    bad_names: list[dict] = field(default_factory=list)

    def of_kind(self, kind: str) -> dict[KardSampleId, DiscoveredFile]:
        return {sid: f for (sid, k), f in sorted(self.files.items()) if k == kind}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(root: str | Path) -> Discovery:
    """Recursively find KARD skeleton files under ``root``.

    Duplicates (same action/subject/repetition/kind in several places) are reported. The
    lexicographically first path is kept when the copies are byte-identical; when they
    differ, the sample is dropped and reported as a conflict.
    """
    root = Path(root)
    disc = Discovery()
    if not root.exists():
        return disc
    conflicts: set[tuple[KardSampleId, str]] = set()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = path.name
        if not name.endswith(".txt") or not ("_realworld" in name or "_screen" in name):
            if name.startswith("a") and name.endswith(".txt"):
                disc.unrecognized.append(str(path.relative_to(root)))
            continue
        try:
            sid, kind = parse_filename(name)
        except KardParseError as exc:
            disc.bad_names.append({"path": str(path.relative_to(root)), "reason": exc.reason})
            continue
        key = (sid, kind)
        digest = _sha256(path)
        if key in disc.files:
            first = disc.files[key]
            identical = first.sha256 == digest
            disc.duplicates.append(
                {
                    "seq_id": sid.seq_id,
                    "kind": kind,
                    "kept": str(first.path.relative_to(root)),
                    "duplicate": str(path.relative_to(root)),
                    "identical": identical,
                }
            )
            if not identical:
                conflicts.add(key)
            continue
        disc.files[key] = DiscoveredFile(path=path, sample_id=sid, kind=kind, sha256=digest)
    for key in conflicts:
        disc.files.pop(key, None)
    return disc


@dataclass
class KardCollection:
    """All parsed sequences of one coordinate kind, stored contiguously.

    Attributes:
        coords: ``[sum(T_i), 15, 3]`` float32, sequences concatenated along time.
        offsets: ``[N + 1]`` int64; sequence i is ``coords[offsets[i]:offsets[i+1]]``.
        actions, subjects, repetitions: ``[N]`` int64, official 1-based ids.
        seq_ids: ``[N]`` original sequence identifiers (``aAA_sSS_eEE``).
        kind: ``"realworld"`` or ``"screen"``.
    """

    coords: np.ndarray
    offsets: np.ndarray
    actions: np.ndarray
    subjects: np.ndarray
    repetitions: np.ndarray
    seq_ids: list[str]
    kind: str

    def __len__(self) -> int:
        return len(self.seq_ids)

    @property
    def labels(self) -> np.ndarray:
        return self.actions - 1

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def sequence(self, i: int) -> np.ndarray:
        return self.coords[self.offsets[i] : self.offsets[i + 1]]

    def index_of(self, seq_id: str) -> int:
        try:
            return self.seq_ids.index(seq_id)
        except ValueError:
            raise KeyError(f"sequence {seq_id!r} not in collection") from None

    @classmethod
    def from_sequences(
        cls, items: Iterable[tuple[KardSampleId, np.ndarray]], kind: str
    ) -> KardCollection:
        items = sorted(items, key=lambda x: x[0])
        if not items:
            raise ValueError("cannot build an empty KardCollection")
        lengths = [a.shape[0] for _, a in items]
        return cls(
            coords=np.concatenate([a for _, a in items], axis=0).astype(np.float32),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            actions=np.array([s.action for s, _ in items], dtype=np.int64),
            subjects=np.array([s.subject for s, _ in items], dtype=np.int64),
            repetitions=np.array([s.repetition for s, _ in items], dtype=np.int64),
            seq_ids=[s.seq_id for s, _ in items],
            kind=kind,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coords=self.coords,
            offsets=self.offsets,
            actions=self.actions,
            subjects=self.subjects,
            repetitions=self.repetitions,
            seq_ids=np.array(self.seq_ids, dtype="U16"),
            kind=np.array(self.kind),
        )

    @classmethod
    def load(cls, path: str | Path) -> KardCollection:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                coords=z["coords"],
                offsets=z["offsets"],
                actions=z["actions"],
                subjects=z["subjects"],
                repetitions=z["repetitions"],
                seq_ids=[str(s) for s in z["seq_ids"]],
                kind=str(z["kind"]),
            )


def load_collection(root: str | Path, kind: str = "realworld") -> tuple[KardCollection, list]:
    """Discover and parse every file of ``kind`` under ``root``.

    Returns:
        (collection of valid sequences, list of ``{"seq_id", "path", "reason"}`` rejects)
    """
    if kind not in COORD_KINDS:
        raise ValueError(f"kind must be one of {COORD_KINDS}")
    disc = discover(root)
    items: list[tuple[KardSampleId, np.ndarray]] = []
    rejected: list[dict] = []
    for sid, f in disc.of_kind(kind).items():
        try:
            parsed = parse_skeleton_file(f.path)
        except KardParseError as exc:
            rejected.append({"seq_id": sid.seq_id, "path": str(f.path), "reason": exc.reason})
            continue
        if parsed.n_nonfinite:
            rejected.append(
                {
                    "seq_id": sid.seq_id,
                    "path": str(f.path),
                    "reason": f"{parsed.n_nonfinite} non-finite coordinate values",
                }
            )
            continue
        items.append((sid, parsed.coords))
    return KardCollection.from_sequences(items, kind), rejected
