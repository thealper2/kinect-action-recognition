"""Concrete skeleton sources: Kinect v1 (in-process or via TCP bridge), KARD replay, synthetic."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import numpy as np

from skeleton_action.constants import NUM_JOINTS, J
from skeleton_action.sensors.base import LOST, NO_PERSON, TRACKED, SkeletonFrame, SkeletonSource
from skeleton_action.sensors.kinect_v1_mapping import KinectV1Mapper


class PersonSelector:
    """Choose one tracked skeleton per frame.

    Keeps following the same SDK tracking id while it is present; otherwise picks the
    skeleton closest to the sensor (smallest z).
    """

    def __init__(self) -> None:
        self.current_id: int | None = None

    def select(self, skeletons: list[dict]) -> dict | None:
        if not skeletons:
            self.current_id = None
            return None
        for s in skeletons:
            if s["tracking_id"] == self.current_id:
                return s
        best = min(skeletons, key=lambda s: s["position"][2])
        self.current_id = best["tracking_id"]
        return best


class ClockSync:
    """Map sensor timestamps (ms) onto this process's ``time.monotonic()`` clock.

    Frame spacing comes from the sensor clock (exact), the offset from the first frame
    received; the offset is re-estimated when the two clocks diverge by more than
    ``max_drift_s`` (e.g. after the bridge reconnects).
    """

    def __init__(self, max_drift_s: float = 1.0) -> None:
        self.offset: float | None = None
        self.max_drift_s = max_drift_s

    def __call__(self, sensor_ms: float | None) -> float:
        now = time.monotonic()
        if sensor_ms is None:
            return now
        s = sensor_ms / 1000.0
        if self.offset is None or abs(s + self.offset - now) > self.max_drift_s:
            self.offset = now - s
        return s + self.offset


def sdk_frame_to_skeleton_frame(
    raw: dict,
    mapper: KinectV1Mapper,
    selector: PersonSelector,
    source: str,
    clock: ClockSync | None = None,
) -> SkeletonFrame:
    """Convert one bridge/reader frame dict to a KARD-convention :class:`SkeletonFrame`."""
    ts = raw.get("sdk_timestamp_ms")
    t = clock(ts) if clock is not None else (ts / 1000.0 if ts is not None else time.monotonic())
    person = selector.select(raw.get("skeletons", []))
    if person is None:
        return SkeletonFrame(
            t,
            None,
            None,
            NO_PERSON,
            source,
            int(raw.get("frame_number", 0)),
            {"t_send": raw.get("t_send")},
        )
    joints, valid, info = mapper.convert(
        np.asarray(person["joints"], dtype=np.float64), np.asarray(person["joint_states"])
    )
    meta = {
        "tracking_id": person["tracking_id"],
        "t_send": raw.get("t_send"),
        **info,
        "n_people": len(raw.get("skeletons", [])),
    }
    return SkeletonFrame(t, joints, valid, TRACKED, source, int(raw.get("frame_number", 0)), meta)


class KinectV1SdkSource(SkeletonSource):
    """In-process Kinect v1 via the SDK 1.8 DLL (Windows Python with numpy/torch only)."""

    name = "kinect_v1_sdk"

    def __init__(self, mapper: KinectV1Mapper | None = None, smoothing: bool = True) -> None:
        from skeleton_action.sensors.kinect_sdk_ctypes import KinectV1SdkReader

        self.reader = KinectV1SdkReader(
            smoothing=KinectV1SdkReader.DEFAULT_SMOOTHING if smoothing else None
        )
        self.mapper = mapper or KinectV1Mapper()
        self.selector = PersonSelector()
        self.clock = ClockSync()

    def open(self) -> None:
        self.reader.open()

    def close(self) -> None:
        self.reader.close()

    def read(self) -> SkeletonFrame | None:
        raw = self.reader.read()
        if raw is None:
            return SkeletonFrame(time.monotonic(), None, None, LOST, self.name)
        return sdk_frame_to_skeleton_frame(raw, self.mapper, self.selector, self.name, self.clock)


class BridgeSource(SkeletonSource):
    """TCP listener for ``scripts/kinect_bridge_windows.py`` (newline-delimited JSON).

    Listens on ``host:port`` (default all interfaces, so the Windows host can reach a
    WSL2 listener through localhost forwarding) and accepts one bridge connection at a
    time. ``read`` returns a LOST frame when no data arrives within ``timeout`` seconds
    and keeps waiting for reconnections; it never ends the stream by itself.
    """

    name = "kinect_v1_bridge"

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        timeout: float = 0.5,
        mapper: KinectV1Mapper | None = None,
        verbose: bool = True,
    ) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.mapper = mapper or KinectV1Mapper()
        self.selector = PersonSelector()
        self.clock = ClockSync()
        self.verbose = verbose
        self._srv: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._buf = b""
        self.hello: dict | None = None

    def open(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        srv.settimeout(self.timeout)
        self._srv = srv
        if self.verbose:
            print(
                f"[bridge-source] listening on {self.host}:{self.port}; start "
                "scripts/kinect_bridge_windows.py on Windows",
                flush=True,
            )

    def close(self) -> None:
        for s in (self._conn, self._srv):
            if s is not None:
                s.close()
        self._conn = self._srv = None

    def _lost(self) -> SkeletonFrame:
        return SkeletonFrame(time.monotonic(), None, None, LOST, self.name)

    def read(self) -> SkeletonFrame | None:
        if self._srv is None:
            raise RuntimeError("source not opened")
        if self._conn is None:
            try:
                conn, addr = self._srv.accept()
            except TimeoutError:
                return self._lost()
            conn.settimeout(self.timeout)
            self._conn, self._buf = conn, b""
            if self.verbose:
                print(f"[bridge-source] bridge connected from {addr}", flush=True)
        while b"\n" not in self._buf:
            try:
                chunk = self._conn.recv(65536)
            except TimeoutError:
                return self._lost()
            except OSError:
                chunk = b""
            if not chunk:
                if self.verbose:
                    print("[bridge-source] bridge disconnected", flush=True)
                self._conn.close()
                self._conn = None
                self.selector = PersonSelector()
                return self._lost()
            self._buf += chunk
        # Drain to the newest complete line: stale frames are useless for real time.
        *lines, self._buf = self._buf.split(b"\n")
        msg = None
        for line in reversed(lines):
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("type") == "hello":
                self.hello = m
                continue
            msg = m
            break
        if msg is None:
            return self._lost()
        frame = sdk_frame_to_skeleton_frame(msg, self.mapper, self.selector, self.name, self.clock)
        frame.meta["dropped_stale"] = max(0, len([x for x in lines if x.strip()]) - 1)
        return frame


class KardReplaySource(SkeletonSource):
    """Replay KARD real-world sequences as a live stream (mock sensor, no hardware).

    Sequences are played back-to-back at ``fps`` with ``gap_frames`` of "no person"
    frames between them, optionally in real time. Each frame's ``meta`` carries the
    ground-truth ``seq_id``/``label`` so the pipeline can be checked end to end.
    """

    name = "kard_replay"

    def __init__(
        self,
        sequences: list[tuple[str, int, np.ndarray]],
        fps: float = 30.0,
        gap_frames: int = 30,
        realtime: bool = False,
        loop: bool = False,
        noise_mm: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not sequences:
            raise ValueError("no sequences to replay")
        self.sequences, self.fps, self.gap = sequences, fps, gap_frames
        self.realtime, self.loop, self.noise = realtime, loop, noise_mm
        self.rng = np.random.default_rng(seed)
        self._plan = self._make_plan()
        self._i = 0
        self._t0: float | None = None

    def _make_plan(self) -> list[tuple[int, int]]:
        plan = []
        for s in range(len(self.sequences)):
            plan += [(s, k) for k in range(self.sequences[s][2].shape[0])]
            plan += [(-1, k) for k in range(self.gap)]
        return plan

    def read(self) -> SkeletonFrame | None:
        if self._i >= len(self._plan):
            if not self.loop:
                return None
            self._i = 0
        if self._t0 is None:
            self._t0 = time.monotonic()
        t = self._i / self.fps
        if self.realtime:
            delay = self._t0 + t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        s, k = self._plan[self._i]
        self._i += 1
        if s < 0:
            return SkeletonFrame(t, None, None, NO_PERSON, self.name, self._i)
        seq_id, label, arr = self.sequences[s]
        joints = arr[k].astype(np.float32).copy()
        if self.noise > 0:
            joints += self.rng.normal(0, self.noise, joints.shape).astype(np.float32)
        return SkeletonFrame(
            t,
            joints,
            np.ones(NUM_JOINTS, bool),
            TRACKED,
            self.name,
            self._i,
            {"seq_id": seq_id, "label": label, "frame_in_seq": k},
        )

    @classmethod
    def from_collection(
        cls,
        path: str | Path,
        seq_ids: list[str] | None = None,
        subjects: list[int] | None = None,
        limit: int | None = None,
        **kw,
    ) -> KardReplaySource:
        from skeleton_action.data.kard import KardCollection

        col = KardCollection.load(path)
        idx = range(len(col))
        if seq_ids:
            idx = [col.index_of(s) for s in seq_ids]
        elif subjects:
            idx = [i for i in idx if int(col.subjects[i]) in subjects]
        idx = list(idx)[:limit] if limit else list(idx)
        seqs = [(col.seq_ids[i], int(col.labels[i]), col.sequence(i)) for i in idx]
        return cls(seqs, **kw)


class SyntheticSource(SkeletonSource):
    """Procedural standing skeleton waving its right hand (smoke tests only).

    Synthetic data verifies that the pipeline runs; it says nothing about accuracy.
    """

    name = "synthetic"

    def __init__(
        self,
        num_frames: int = 300,
        fps: float = 30.0,
        realtime: bool = False,
        dropout_every: int = 0,
    ) -> None:
        self.n, self.fps, self.realtime = num_frames, fps, realtime
        self.dropout_every = dropout_every
        self._i = 0
        self._t0: float | None = None

    @staticmethod
    def base_pose() -> np.ndarray:
        """A plausible KARD-convention standing pose (mm, ~2.8 m from the sensor)."""
        p = np.zeros((NUM_JOINTS, 3))
        p[J.HEAD] = [0, 560, 2700]
        p[J.NECK] = [0, 350, 2740]
        p[J.R_SHOULDER] = [-175, 350, 2745]
        p[J.R_ELBOW] = [-240, 105, 2800]
        p[J.R_HAND] = [-280, -160, 2760]
        p[J.L_SHOULDER] = [175, 350, 2745]
        p[J.L_ELBOW] = [240, 105, 2800]
        p[J.L_HAND] = [280, -160, 2760]
        p[J.TORSO] = [0, 135, 2780]
        p[J.R_HIP] = [-100, -80, 2820]
        p[J.R_KNEE] = [-110, -520, 2830]
        p[J.R_FOOT] = [-115, -930, 2830]
        p[J.L_HIP] = [100, -80, 2820]
        p[J.L_KNEE] = [110, -520, 2830]
        p[J.L_FOOT] = [115, -930, 2830]
        return p

    def frame_at(self, i: int) -> np.ndarray:
        p = self.base_pose()
        phase = 2 * np.pi * i / self.fps  # 1 Hz wave
        p[J.R_ELBOW] = [-300, 450, 2760]
        p[J.R_HAND] = [-300 + 200 * np.sin(phase), 700, 2740]
        return p.astype(np.float32)

    def read(self) -> SkeletonFrame | None:
        if self._i >= self.n:
            return None
        if self._t0 is None:
            self._t0 = time.monotonic()
        t = self._i / self.fps
        if self.realtime:
            delay = self._t0 + t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        i = self._i
        self._i += 1
        valid = np.ones(NUM_JOINTS, bool)
        if self.dropout_every and i % self.dropout_every == 0:
            valid[J.L_HAND] = False
        joints = self.frame_at(i)
        joints[~valid] = 0.0
        return SkeletonFrame(t, joints, valid, TRACKED, self.name, i)
