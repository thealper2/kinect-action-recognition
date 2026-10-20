"""Sensor-agnostic skeleton source interface.

Every source yields :class:`SkeletonFrame` objects whose joints are already expressed in
the KARD real-world convention: 15 joints in KARD order, millimetres, X to the right of
the sensor image, Y up, Z = distance from the sensor. Conversion from a sensor's own
skeleton (e.g. the 20-joint Kinect v1 SDK skeleton) happens inside the adapter, so the
model and the real-time pipeline never see sensor-specific layouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

TRACKED = "tracked"
NO_PERSON = "no_person"
LOST = "lost"  # source alive but data missing/late


@dataclass
class SkeletonFrame:
    """One skeleton observation.

    Attributes:
        timestamp: seconds (monotonic clock of the producer).
        joints: ``[15, 3]`` float32 in KARD real-world convention, or None when no person
            is tracked.
        valid: ``[15]`` bool per-joint validity (False = not tracked / unmapped).
        tracking: ``tracked`` | ``no_person`` | ``lost``.
        source: human-readable source name.
        frame_number: producer frame counter.
        meta: extra info (tracking id, inferred-joint count, ...).
    """

    timestamp: float
    joints: np.ndarray | None
    valid: np.ndarray | None
    tracking: str
    source: str
    frame_number: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def num_valid(self) -> int:
        return 0 if self.valid is None else int(self.valid.sum())


class SkeletonSource(ABC):
    """Abstract skeleton stream. Use as a context manager and iterate over frames."""

    name: str = "source"

    def open(self) -> None:  # noqa: B027 - optional hook
        """Acquire the device/stream."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release the device/stream."""

    @abstractmethod
    def read(self) -> SkeletonFrame | None:
        """Block until the next frame; ``None`` signals the end of the stream."""

    def __enter__(self) -> SkeletonSource:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[SkeletonFrame]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f
