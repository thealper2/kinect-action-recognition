#!/usr/bin/env python
"""Windows-side Kinect v1 bridge: reads SDK 1.8 skeletons and streams them over TCP.

Runs with a stock *Windows* Python 3.8+ (standard library only; no numpy/torch needed)
on the PC where the Xbox 360 Kinect and the Kinect for Windows SDK/Runtime 1.8 are
installed. The recogniser (WSL2 or any other machine) listens with
``scripts/run_realtime.py --source bridge``; this script connects to it.

WSL2 (default NAT networking) forwards ``localhost`` on Windows to WSL, so the default
``--host 127.0.0.1`` reaches a listener inside WSL.

Usage (PowerShell / cmd, from the project folder):
    python scripts\\kinect_bridge_windows.py --print          # local test, no network
    python scripts\\kinect_bridge_windows.py --host 127.0.0.1 --port 8765

Protocol: newline-delimited JSON. First line ``{"type": "hello", ...}``, then one
``{"type": "frame", "t_send": <unix s>, ...reader fields...}`` per skeleton frame
(SDK skeleton space, metres; mapping to KARD happens on the receiving side).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skeleton_action.sensors.kinect_sdk_ctypes import KinectError, KinectV1SdkReader

PROTOCOL_VERSION = 1


def stream(reader: KinectV1SdkReader, sock: socket.socket | None, print_every: int) -> None:
    n = 0
    t0 = time.time()
    while True:
        fr = reader.read()
        if fr is None:
            continue
        fr["type"] = "frame"
        fr["t_send"] = time.time()
        n += 1
        if sock is not None:
            sock.sendall((json.dumps(fr, separators=(",", ":")) + "\n").encode())
        if print_every and n % print_every == 0:
            rate = n / (time.time() - t0)
            s = fr["skeletons"]
            info = "no tracked person"
            if s:
                head = s[0]["joints"][3]
                states = s[0]["joint_states"]
                info = (
                    f"{len(s)} tracked | id {s[0]['tracking_id']} | head "
                    f"({head[0]:+.2f}, {head[1]:+.2f}, {head[2]:.2f}) m | joints tracked "
                    f"{sum(v == 2 for v in states)}/20 inferred {sum(v == 1 for v in states)}"
                )
            print(f"frame {fr['frame_number']:6d} | {rate:5.1f} fps | {info}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="only print frames locally (hardware test, no network)",
    )
    ap.add_argument("--print-every", type=int, default=30)
    ap.add_argument(
        "--no-smoothing", action="store_true", help="disable the SDK NuiTransformSmooth filter"
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop after this many seconds (0 = run until Ctrl+C)",
    )
    ap.add_argument(
        "--exit-on-disconnect",
        action="store_true",
        help="exit when an established connection drops (used by `make live`)",
    )
    ap.add_argument(
        "--give-up-after",
        type=float,
        default=0.0,
        help="exit if no receiver can be reached for this many seconds (0 = retry forever)",
    )
    args = ap.parse_args()

    reader = KinectV1SdkReader(
        smoothing=None if args.no_smoothing else KinectV1SdkReader.DEFAULT_SMOOTHING
    )
    try:
        reader.open()
    except KinectError as exc:
        raise SystemExit(f"[bridge] {exc}") from None
    print(f"[bridge] Kinect opened ({reader.sensor_count} sensor(s)).", flush=True)
    if args.max_seconds > 0:
        import threading

        threading.Timer(args.max_seconds, lambda: __import__("os")._exit(0)).start()
    try:
        if args.print_only:
            stream(reader, None, args.print_every)
            return
        connected_once = False
        unreachable_since: float | None = time.monotonic()
        while True:
            try:
                with socket.create_connection((args.host, args.port), timeout=5) as sock:
                    sock.settimeout(None)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    hello = {
                        "type": "hello",
                        "protocol": PROTOCOL_VERSION,
                        "sensor": "Kinect v1 (SDK 1.8)",
                        "units": "m",
                        "space": "NUI skeleton space",
                    }
                    sock.sendall((json.dumps(hello) + "\n").encode())
                    connected_once = True
                    unreachable_since = None
                    print(
                        f"[bridge] connected to {args.host}:{args.port}, streaming...", flush=True
                    )
                    stream(reader, sock, args.print_every)
            except (ConnectionError, TimeoutError, OSError) as exc:
                if args.exit_on_disconnect and connected_once:
                    print("[bridge] receiver closed the connection; exiting", flush=True)
                    return
                if unreachable_since is None:
                    unreachable_since = time.monotonic()
                waited = time.monotonic() - unreachable_since
                if args.give_up_after and waited > args.give_up_after:
                    print(f"[bridge] no receiver for {waited:.0f} s; exiting", flush=True)
                    return
                print(
                    f"[bridge] {args.host}:{args.port} unavailable ({exc}); retrying in 2 s",
                    flush=True,
                )
                time.sleep(2)
    except KeyboardInterrupt:
        print("[bridge] stopped")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
