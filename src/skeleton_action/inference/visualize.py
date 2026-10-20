"""Lightweight OpenCV overlay: front-view skeleton + prediction HUD.

OpenCV is optional; :func:`have_display` tells whether a window can be opened.
"""

from __future__ import annotations

import os

import numpy as np

from skeleton_action.constants import KARD_EDGES
from skeleton_action.inference.realtime import CONFIDENT, RecognizerOutput
from skeleton_action.sensors.base import SkeletonFrame

STATUS_COLORS = {  # BGR
    "confident": (60, 200, 60),
    "uncertain": (0, 190, 255),
    "warming_up": (200, 200, 200),
    "no_person": (120, 120, 120),
}


def display_problem() -> str:
    """Empty string if an OpenCV window can be opened, otherwise the reason.

    Checks for a display *and* for a GUI-enabled OpenCV build: ``opencv-python-headless``
    imports fine but fails on ``namedWindow``, and when several OpenCV wheels share one
    environment the headless one may be the ``cv2`` actually loaded.
    """
    if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no DISPLAY/WAYLAND_DISPLAY (WSLg not running?)"
    # The Qt build of OpenCV ships no fonts; point it at the system fonts to avoid warnings.
    if os.name != "nt" and "QT_QPA_FONTDIR" not in os.environ:
        for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts"):
            if os.path.isdir(d):
                os.environ["QT_QPA_FONTDIR"] = d
                break
    try:
        import cv2
    except ImportError:
        return "OpenCV is not installed"
    gui = [
        line.split(":", 1)[1].strip()
        for line in cv2.getBuildInformation().splitlines()
        if line.strip().startswith("GUI:")
    ]
    if gui and gui[0].upper() == "NONE":
        return (
            f"the loaded cv2 ({cv2.__file__}) is a headless build without GUI support; "
            "remove opencv-python-headless and reinstall opencv-python or "
            "opencv-contrib-python (docs/troubleshooting.md)"
        )
    return ""


def have_display() -> bool:
    return not display_problem()


class SkeletonView:
    def __init__(self, width: int = 960, height: int = 600, title: str = "KARD action") -> None:
        import cv2

        self.cv2 = cv2
        self.w, self.h, self.title = width, height, title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def _project(self, joints: np.ndarray) -> np.ndarray:
        """Front view: X (right in sensor image) -> u, Y (up) -> -v, 2.4 m tall field."""
        scale = (self.h * 0.9) / 2400.0
        cx = joints[:, 0].mean() if joints.size else 0.0
        u = self.w * 0.35 + (joints[:, 0] - cx) * scale
        v = self.h * 0.5 - joints[:, 1] * scale
        return np.stack([u, v], axis=1).astype(int)

    def draw(self, frame: SkeletonFrame, out: RecognizerOutput, source: str, fps: float) -> bool:
        """Render and return False when the user pressed q/ESC."""
        cv2 = self.cv2
        img = np.full((self.h, self.w, 3), 25, np.uint8)
        if frame.joints is not None and frame.valid is not None:
            pts = self._project(frame.joints)
            for a, b in KARD_EDGES:
                if frame.valid[a] and frame.valid[b]:
                    cv2.line(img, tuple(pts[a]), tuple(pts[b]), (230, 230, 230), 3)
            for j, p in enumerate(pts):
                color = (80, 220, 80) if frame.valid[j] else (60, 60, 230)
                cv2.circle(img, tuple(p), 6, color, -1)
        else:
            # This view draws the tracked skeleton only (no camera image), so say explicitly
            # why it is empty. Hershey fonts are ASCII-only.
            hint = [
                ("NO SKELETON TRACKED", 1.0, (80, 80, 255), 2),
                (
                    "Stand 1.5 - 3.5 m from the Kinect, whole body in view.",
                    0.55,
                    (220, 220, 220),
                    1,
                ),
                (
                    "Sensor must face you (tilt ~0 deg), nothing closer than 0.8 m.",
                    0.55,
                    (220, 220, 220),
                    1,
                ),
                ("See what the sensor sees:  make kinect-diag", 0.55, (0, 220, 255), 1),
            ]
            y = self.h // 2 - 40
            for text, sc, c, th in hint:
                cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)
                y += 40 if sc > 0.8 else 30
        x0 = int(self.w * 0.66)
        col = STATUS_COLORS.get(out.status, (200, 200, 200))
        label = out.label if out.status == CONFIDENT else f"({out.status})"
        lines = [
            (label, 0.9, col, 2),
            (f"confidence {out.confidence:.2f}", 0.6, (220, 220, 220), 1),
            (f"tracking: {out.tracking}", 0.55, (220, 220, 220), 1),
            (f"source: {source}  {fps:.1f} fps", 0.55, (220, 220, 220), 1),
            (
                f"latency: model {out.forward_ms:.1f} ms | pipeline {out.pipeline_ms:.1f} ms",
                0.5,
                (220, 220, 220),
                1,
            ),
            (
                f"buffer {out.buffer_seconds:.1f}s / windows {out.window_seconds}",
                0.5,
                (180, 180, 180),
                1,
            ),
        ]
        y = 40
        for text, sc, c, th in lines:
            cv2.putText(
                img,
                str(text),
                (x0 - 300, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                sc,
                c,
                th,
                cv2.LINE_AA,
            )
            y += 34 if sc > 0.8 else 26
        y += 10
        for name, p in out.topk:
            cv2.rectangle(
                img, (x0 - 300, y - 14), (x0 - 300 + int(250 * p), y + 2), (90, 140, 90), -1
            )
            cv2.putText(
                img,
                f"{name} {p:.2f}",
                (x0 - 295, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y += 24
        if out.reason:
            cv2.putText(
                img,
                out.reason[:90],
                (10, self.h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(self.title, img)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        self.cv2.destroyAllWindows()
