"""Real-time recognition over a skeleton stream.

Per incoming :class:`SkeletonFrame`:

1. tracking-loss handling: frames without a tracked person are not buffered; after
   ``lost_timeout_s`` without a tracked person the buffer and the smoothing state reset;
2. temporal buffer: tracked frames (KARD-convention joints + validity) with timestamps;
3. every ``stride_frames`` tracked frames, for each window length W the frames of the
   last W seconds are resampled onto a uniform time grid at ``source_fps`` (missing
   joints stay masked) and passed through the *training* preprocessing via the
   predictor; probabilities are averaged over window lengths;
4. smoothing: exponential moving average of probabilities (or majority vote);
5. decision: ``confident`` only if smoothed top-1 >= ``confidence_threshold`` and
   top1 - top2 >= ``margin_threshold``; otherwise ``uncertain``. With too little
   valid data the status is ``warming_up`` and no label is emitted.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from skeleton_action.constants import NUM_JOINTS
from skeleton_action.inference.predictor import Prediction, Predictor
from skeleton_action.sensors.base import TRACKED, SkeletonFrame

NO_PERSON_STATUS = "no_person"
WARMING_UP = "warming_up"
UNCERTAIN = "uncertain"
CONFIDENT = "confident"


@dataclass
class RealtimeConfig:
    window_seconds: list[float] | None = None  # None -> [p05, median] train length / fps
    source_fps: float = 30.0  # resampling grid (KARD assumed rate)
    stride_frames: int = 5
    min_window_coverage: float = 0.8  # tracked span / window length
    max_gap_s: float = 0.35  # larger holes in the window -> not ready
    lost_timeout_s: float = 0.5
    smoothing: str = "ema"  # ema | majority | none
    ema_alpha: float = 0.35
    majority_k: int = 5
    confidence_threshold: float = 0.6
    margin_threshold: float = 0.15
    topk: int = 3

    @classmethod
    def from_dict(cls, d: dict | None) -> RealtimeConfig:
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown realtime keys {sorted(unknown)}")
        return cls(**d)


@dataclass
class RecognizerOutput:
    status: str
    tracking: str
    label: str | None = None
    confidence: float = 0.0
    topk: list[tuple[str, float]] = field(default_factory=list)
    buffer_seconds: float = 0.0
    window_seconds: list[float] = field(default_factory=list)
    preprocess_ms: float = 0.0
    forward_ms: float = 0.0
    pipeline_ms: float = 0.0
    updated: bool = False
    reason: str = ""
    raw: Prediction | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "tracking": self.tracking,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "topk": [(n, round(p, 4)) for n, p in self.topk],
            "buffer_s": round(self.buffer_seconds, 2),
            "updated": self.updated,
            "preprocess_ms": round(self.preprocess_ms, 2),
            "forward_ms": round(self.forward_ms, 2),
            "pipeline_ms": round(self.pipeline_ms, 2),
            "reason": self.reason,
        }


def resample_window(
    times: np.ndarray,
    joints: np.ndarray,
    valid: np.ndarray,
    t_end: float,
    duration: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample buffered frames in ``[t_end - duration, t_end]`` onto a uniform grid.

    Each joint is linearly interpolated over time using only the frames where it was
    valid; grid points outside the joint's valid time span (or with no valid sample at
    all) are marked invalid.

    Returns:
        (sequence ``[N, 15, 3]``, mask ``[N, 15]``) with ``N = round(duration * fps) + 1``.
    """
    n = round(duration * fps) + 1
    grid = np.linspace(t_end - duration, t_end, n)
    seq = np.zeros((n, NUM_JOINTS, 3), dtype=np.float64)
    mask = np.zeros((n, NUM_JOINTS), dtype=bool)
    for j in range(NUM_JOINTS):
        m = valid[:, j]
        if m.sum() < 2:
            continue
        tj = times[m]
        inside = (grid >= tj[0]) & (grid <= tj[-1])
        for c in range(3):
            seq[:, j, c] = np.interp(grid, tj, joints[m, j, c])
        mask[:, j] = inside
    return seq.astype(np.float32), mask


class RealtimeRecognizer:
    def __init__(self, predictor: Predictor, cfg: RealtimeConfig | None = None) -> None:
        self.pred = predictor
        self.cfg = cfg or RealtimeConfig()
        if self.cfg.window_seconds:
            self.windows = sorted(float(w) for w in self.cfg.window_seconds)
        else:
            # Default: two scales from the training sequence-length distribution (5th
            # percentile and median), so short actions (e.g. Hand clap) are covered too.
            st = predictor.ckpt.get("data_stats", {})
            lens = (st.get("train_seq_len_p05", 63.0), st.get("train_seq_len_median", 112.0))
            self.windows = sorted({round(float(n) / self.cfg.source_fps, 2) for n in lens})
        self.max_window = max(self.windows)
        self.buf: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self.reset()

    def reset(self) -> None:
        self.buf.clear()
        self.ema: np.ndarray | None = None
        self.votes: deque[int] = deque(maxlen=self.cfg.majority_k)
        self.since_last = 0
        self.last_tracked_t: float | None = None
        self.last_output: RecognizerOutput | None = None

    # ---------------------------------------------------------------- frame handling

    def update(self, frame: SkeletonFrame) -> RecognizerOutput:
        t_start = time.perf_counter()
        if frame.tracking != TRACKED or frame.joints is None:
            lost_for = (
                frame.timestamp - self.last_tracked_t
                if self.last_tracked_t is not None
                else float("inf")
            )
            if lost_for > self.cfg.lost_timeout_s:
                self.reset()
                return RecognizerOutput(
                    NO_PERSON_STATUS, frame.tracking, reason="no tracked person"
                )
            prev = self.last_output
            return RecognizerOutput(
                prev.status if prev else WARMING_UP,
                frame.tracking,
                prev.label if prev else None,
                prev.confidence if prev else 0.0,
                prev.topk if prev else [],
                self._span(),
                self.windows,
                reason=f"tracking interrupted ({lost_for:.2f}s)",
            )
        self.last_tracked_t = frame.timestamp
        valid = frame.valid if frame.valid is not None else np.ones(NUM_JOINTS, bool)
        self.buf.append((frame.timestamp, np.asarray(frame.joints, np.float32), valid.copy()))
        while self.buf and self.buf[0][0] < frame.timestamp - self.max_window - 0.5:
            self.buf.popleft()
        self.since_last += 1
        if self.since_last < self.cfg.stride_frames and self.last_output is not None:
            out = self.last_output
            return RecognizerOutput(
                out.status,
                frame.tracking,
                out.label,
                out.confidence,
                out.topk,
                self._span(),
                self.windows,
                reason=out.reason,
            )
        self.since_last = 0
        out = self._infer(frame.timestamp)
        out.tracking = frame.tracking
        out.pipeline_ms = (time.perf_counter() - t_start) * 1e3
        self.last_output = out
        return out

    def _span(self) -> float:
        return self.buf[-1][0] - self.buf[0][0] if len(self.buf) > 1 else 0.0

    def _window_ready(self, times: np.ndarray, t_end: float, w: float) -> str:
        sel = times[times >= t_end - w - 1e-6]
        if sel.size < 2:
            return "not enough frames"
        cover = (sel[-1] - sel[0]) / w
        if cover < self.cfg.min_window_coverage:
            return f"window {w:.1f}s only {cover * 100:.0f}% filled"
        gap = float(np.diff(sel).max()) if sel.size > 1 else 0.0
        if gap > self.cfg.max_gap_s:
            return f"tracking gap of {gap:.2f}s in window"
        return ""

    def _infer(self, t_end: float) -> RecognizerOutput:
        times = np.array([b[0] for b in self.buf])
        joints = np.stack([b[1] for b in self.buf])
        valid = np.stack([b[2] for b in self.buf])
        probs_list, pre_ms, fwd_ms, reasons = [], 0.0, 0.0, []
        for w in self.windows:
            why = self._window_ready(times, t_end, w)
            if why:
                reasons.append(why)
                continue
            t0 = time.perf_counter()
            # Use the covered part of the window (>= min_window_coverage), like the
            # training-time temporal crops, instead of padding a masked prefix.
            first = float(times[times >= t_end - w - 1e-6][0])
            seq, mask = resample_window(
                times, joints, valid, t_end, t_end - first, self.cfg.source_fps
            )
            pre_ms += (time.perf_counter() - t0) * 1e3
            p = self.pred.predict_raw(seq, mask, topk=self.cfg.topk)
            if not p.valid:
                reasons.append(p.reason)
                continue
            probs_list.append(p.probs)
            pre_ms += p.preprocess_ms
            fwd_ms += p.forward_ms
        if not probs_list:
            return RecognizerOutput(
                WARMING_UP,
                TRACKED,
                buffer_seconds=self._span(),
                window_seconds=self.windows,
                preprocess_ms=pre_ms,
                reason="; ".join(reasons),
            )
        probs = np.mean(probs_list, axis=0)
        smoothed = self._smooth(probs)
        order = np.argsort(-smoothed)
        c1, c2 = int(order[0]), int(order[1])
        conf = float(smoothed[c1])
        margin = conf - float(smoothed[c2])
        names = self.pred.class_names
        topk = [(names[i], float(smoothed[i])) for i in order[: self.cfg.topk]]
        ok = conf >= self.cfg.confidence_threshold and margin >= self.cfg.margin_threshold
        return RecognizerOutput(
            CONFIDENT if ok else UNCERTAIN,
            TRACKED,
            names[c1] if ok else None,
            conf,
            topk,
            self._span(),
            self.windows,
            pre_ms,
            fwd_ms,
            updated=True,
            reason="" if ok else f"top1 {conf:.2f} / margin {margin:.2f} below threshold",
        )

    def _smooth(self, probs: np.ndarray) -> np.ndarray:
        mode = self.cfg.smoothing
        if mode == "none":
            return probs
        if mode == "ema":
            a = self.cfg.ema_alpha
            self.ema = probs if self.ema is None else a * probs + (1 - a) * self.ema
            return self.ema
        if mode == "majority":
            self.votes.append(int(np.argmax(probs)))
            counts = Counter(self.votes)
            out = np.zeros_like(probs)
            for k, v in counts.items():
                out[k] = v / len(self.votes)
            return out
        raise ValueError(f"unknown smoothing {mode!r}")
