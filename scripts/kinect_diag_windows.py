#!/usr/bin/env python
"""Kinect v1 diagnostic viewer (Windows Python with numpy + opencv-python).

Shows what the sensor actually sees, to debug "no tracked person":

* left:  depth image (near = bright); pixels the SDK assigns to a *player* are coloured,
         tracked skeletons are drawn on top; black = no depth reading
* right: colour camera (optional)
* text:  tilt angle, depth coverage, median depth, player pixels, skeleton slot states

Usage (PowerShell / cmd in the project folder, or `make kinect-diag` from WSL):
    python scripts\\kinect_diag_windows.py                     # window, q/ESC quits
    python scripts\\kinect_diag_windows.py --headless --seconds 15
    python scripts\\kinect_diag_windows.py --snapshot depth.png --seconds 5 --no-color
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from skeleton_action.sensors.kinect_sdk_ctypes import (
    KinectError,
    KinectV1Diagnostics,
)

# SDK v1 bone list (20-joint skeleton) for drawing.
SDK_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (2, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (2, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
]
PLAYER_COLORS = [
    (0, 0, 0),
    (60, 60, 255),
    (60, 255, 60),
    (255, 120, 60),
    (0, 220, 255),
    (255, 60, 255),
    (255, 255, 60),
    (200, 200, 200),
]
STATE = {0: "-", 1: "POS_ONLY", 2: "TRACKED"}


def to_depth_px(x: float, y: float, z: float) -> tuple[int, int]:
    """Skeleton space (m) -> 320x240 depth pixel (NuiTransformSkeletonToDepthImage)."""
    if z <= 1e-3:
        return -1, -1
    return int(160 + x * 285.63 / z), int(120 - y * 285.63 / z)


def render_depth(raw: bytes, w: int, h: int) -> tuple[np.ndarray, dict]:
    px = np.frombuffer(raw, dtype="<u2", count=w * h).reshape(h, w)
    depth = (px >> 3).astype(np.float32)
    player = (px & 7).astype(np.int64)
    valid = depth > 0
    img = np.zeros((h, w, 3), np.uint8)
    if valid.any():
        near, far = 800.0, 4000.0
        g = np.clip(255 * (1 - (depth - near) / (far - near)), 30, 255).astype(np.uint8)
        img[valid] = g[valid, None]
    pl = player > 0
    if pl.any():
        img[pl] = (0.4 * img[pl] + 0.6 * np.array(PLAYER_COLORS)[player[pl]]).astype(np.uint8)
    stats = {
        "valid_pct": 100.0 * valid.mean(),
        "median_mm": float(np.median(depth[valid])) if valid.any() else 0.0,
        "player_px": int(pl.sum()),
        "players": sorted({int(p) for p in np.unique(player[pl])}),
    }
    return img, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="print stats only, no window")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N s (0 = until q)")
    ap.add_argument("--no-color", action="store_true", help="do not open the colour camera")
    ap.add_argument("--snapshot", type=Path, help="save the last depth visualisation (PNG)")
    ap.add_argument(
        "--tilt", type=int, help="set the tilt motor angle first (-27..27 deg, 0 = level)"
    )
    args = ap.parse_args()

    cv2 = None
    if not args.headless or args.snapshot:
        import cv2  # type: ignore[no-redef]

    kin = KinectV1Diagnostics(color=not args.no_color)
    try:
        kin.open()
    except KinectError as exc:
        raise SystemExit(f"[diag] {exc}") from None
    print(f"[diag] sensor opened, tilt angle = {kin.elevation_deg()} deg", flush=True)
    if args.tilt is not None:
        kin.set_elevation(args.tilt)
        time.sleep(2.5)  # the motor needs a moment; do not call it repeatedly
        print(f"[diag] tilt set to {args.tilt} deg -> now {kin.elevation_deg()} deg", flush=True)
    t0 = last_print = time.time()
    n_depth = n_skel = 0
    skel = None
    color_img = None
    vis = None
    try:
        while True:
            d = kin.read_depth(100)
            s = kin.read_skeleton_states(0)
            if s is not None:
                skel = s
                n_skel += 1
            c = kin.read_color(0)
            if c is not None and c[2]:
                cw, ch, cb = c
                color_img = np.frombuffer(cb, np.uint8, count=cw * ch * 4).reshape(ch, cw, 4)[
                    ..., :3
                ]
            if d is None or not d[2]:
                continue
            n_depth += 1
            vis, st = render_depth(d[2], d[0], d[1])
            tracked = [k for k, sl in enumerate(skel["slots"])] if skel else []
            if skel and cv2 is not None:
                for k in tracked:
                    joints = skel["joints"][k]
                    if joints is None:
                        continue
                    pts = [to_depth_px(*j) for j in joints]
                    for a, b in SDK_BONES:
                        cv2.line(vis, pts[a], pts[b], (255, 255, 255), 2)
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                slots = ", ".join(
                    f"{STATE.get(sl[0], sl[0])}@{sl[2][2]:.1f}m"
                    for sl in (skel["slots"] if skel else [])
                    if sl[0] != 0
                )
                fp = skel["floor_clip_plane"] if skel else [0, 0, 0, 0]
                print(
                    f"[diag] depth fps {n_depth / (now - t0):4.1f} | valid depth "
                    f"{st['valid_pct']:5.1f}% | median {st['median_mm']:5.0f} mm | player px "
                    f"{st['player_px']:6d} {st['players']} | skeletons: {slots or 'none'} | "
                    f"floor ({fp[0]:+.2f},{fp[1]:+.2f},{fp[2]:+.2f},{fp[3]:+.2f})",
                    flush=True,
                )
            if not args.headless:
                left = cv2.resize(vis, (640, 480), interpolation=cv2.INTER_NEAREST)
                txt = (
                    f"tilt {kin.elevation_deg()} deg | depth {st['valid_pct']:.0f}% | "
                    f"player px {st['player_px']} | skeleton slots "
                    f"{[STATE[sl[0]] for sl in skel['slots'] if sl[0]] if skel else []}"
                )
                cv2.putText(left, txt, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                view = left if color_img is None else np.hstack([left, color_img])
                cv2.imshow("Kinect v1 diagnostics (q quits)", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.seconds and now - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if args.snapshot and vis is not None:
            cv2.imwrite(
                str(args.snapshot), cv2.resize(vis, (640, 480), interpolation=cv2.INTER_NEAREST)
            )
            print(f"[diag] saved {args.snapshot}")
        kin.close()
        if cv2 is not None and not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
