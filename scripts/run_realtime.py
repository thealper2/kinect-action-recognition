#!/usr/bin/env python
"""Real-time action recognition from a skeleton stream.

Sources:
    bridge      Kinect v1 frames from scripts/kinect_bridge_windows.py (WSL2/Linux side)
    kinect-sdk  Kinect v1 in-process via Kinect10.dll (Windows Python with torch)
    replay      KARD sequences replayed as a live stream (mock sensor, ground truth shown)
    synthetic   procedural skeleton (smoke test only)

Examples:
    python scripts/run_realtime.py --checkpoint CKPT --source replay --subjects 9 10 --headless
    python scripts/run_realtime.py --checkpoint CKPT --source bridge          # + OpenCV window
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skeleton_action.inference.predictor import Predictor
from skeleton_action.inference.realtime import (
    CONFIDENT,
    RealtimeConfig,
    RealtimeRecognizer,
)
from skeleton_action.sensors.kinect_v1_mapping import KinectV1Mapper
from skeleton_action.sensors.sources import (
    BridgeSource,
    KardReplaySource,
    KinectV1SdkSource,
    SyntheticSource,
)
from skeleton_action.utils.config import load_yaml
from skeleton_action.utils.seed import select_device


def make_source(args, cfg: dict):
    mapper = KinectV1Mapper(**(cfg.get("kinect_v1") or {}))
    realtime = not args.fast
    if args.source == "bridge":
        b = cfg.get("bridge") or {}
        return BridgeSource(
            host=args.host or b.get("host", "0.0.0.0"),
            port=args.port or b.get("port", 8765),
            mapper=mapper,
        )
    if args.source == "kinect-sdk":
        return KinectV1SdkSource(mapper=mapper)
    if args.source == "replay":
        return KardReplaySource.from_collection(
            args.interim,
            seq_ids=args.seq_ids,
            subjects=args.subjects,
            limit=args.limit,
            fps=args.replay_fps,
            realtime=realtime,
            noise_mm=args.noise_mm,
        )
    return SyntheticSource(num_frames=args.limit * 30 if args.limit else 300, realtime=realtime)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--source", choices=["bridge", "kinect-sdk", "replay", "synthetic"], default="replay"
    )
    ap.add_argument("--config", type=Path, default=Path("configs/inference/realtime.yaml"))
    ap.add_argument("--headless", action="store_true", help="no window; JSON lines on stdout")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--interim", type=Path, default=Path("data/interim/kard_realworld.npz"))
    ap.add_argument("--seq-ids", nargs="*")
    ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--limit", type=int, help="max sequences (replay) / seconds (synthetic)")
    ap.add_argument("--replay-fps", type=float, default=30.0)
    ap.add_argument("--noise-mm", type=float, default=0.0)
    ap.add_argument("--fast", action="store_true", help="replay/synthetic as fast as possible")
    ap.add_argument("--log", type=Path, help="write every output as JSON lines here")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_yaml(args.config) if args.config.exists() else {}
    rt_cfg = RealtimeConfig.from_dict(cfg.get("realtime"))
    predictor = Predictor(args.checkpoint, select_device(args.device))
    rec = RealtimeRecognizer(predictor, rt_cfg)
    src = make_source(args, cfg)
    view = None
    if not args.headless:
        from skeleton_action.inference.visualize import SkeletonView, display_problem

        problem = display_problem()
        if not problem:
            try:
                view = SkeletonView()
            except Exception as exc:  # e.g. Qt/X11 failure at window creation
                problem = f"cannot open window: {exc}"
        if problem:
            print(f"[realtime] WARNING: {problem} -> continuing headless", file=sys.stderr)
    logf = args.log.open("w") if args.log else None
    print(
        f"[realtime] model {predictor.ckpt['model_name']} | windows {rec.windows} s | "
        f"source {src.name}",
        file=sys.stderr,
    )

    n, t0, fps = 0, time.monotonic(), 0.0
    eval_hits = eval_total = 0
    last_seq = None
    try:
        with src:
            for frame in src:
                out = rec.update(frame)
                n += 1
                el = time.monotonic() - t0
                fps = n / el if el > 0 else 0.0
                rec_d = {
                    "t": round(frame.timestamp, 3),
                    "frame": frame.frame_number,
                    **out.to_dict(),
                }
                if "seq_id" in frame.meta:
                    rec_d["gt"] = predictor.class_names[frame.meta["label"]]
                    rec_d["seq_id"] = frame.meta["seq_id"]
                    # Replay check: score the last decision taken at the end of each sequence.
                    if last_seq is not None and frame.meta["seq_id"] != last_seq[0]:
                        eval_total += 1
                        eval_hits += int(last_seq[1] == last_seq[2])
                    last_seq = (
                        frame.meta["seq_id"],
                        rec_d["gt"],
                        out.label if out.status == CONFIDENT else None,
                    )
                if logf:
                    logf.write(json.dumps(rec_d) + "\n")
                if view is not None:
                    if not view.draw(frame, out, src.name, fps):
                        break
                elif out.updated or n % 30 == 0:
                    print(json.dumps(rec_d), flush=True)
                if args.max_seconds and el > args.max_seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if view is not None:
            view.close()
        if logf:
            logf.close()
    if last_seq is not None:
        eval_total += 1
        eval_hits += int(last_seq[1] == last_seq[2])
        print(
            f"[realtime] replay sanity check: confident & correct at sequence end in "
            f"{eval_hits}/{eval_total} sequences (streaming windows, smoothing and "
            f"thresholds included; not the offline test metric)",
            file=sys.stderr,
        )
    print(f"[realtime] processed {n} frames at {fps:.1f} frames/s", file=sys.stderr)


if __name__ == "__main__":
    main()
