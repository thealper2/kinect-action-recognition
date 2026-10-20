"""Kinect for Windows SDK v1.8 skeleton reader via ctypes (Windows only, stdlib only).

This module deliberately depends on nothing but the Python standard library so that it
can run on a stock Windows Python next to the Kinect runtime, while training and model
inference run elsewhere (e.g. WSL2). It calls the flat C API exported by
``Kinect10.dll`` (installed by the Kinect for Windows SDK/Runtime 1.8):

    HRESULT NuiInitialize(DWORD dwFlags);
    void    NuiShutdown(void);
    HRESULT NuiSkeletonTrackingEnable(HANDLE hNextFrameEvent, DWORD dwFlags);
    HRESULT NuiSkeletonGetNextFrame(DWORD dwMillisecondsToWait, NUI_SKELETON_FRAME *pFrame);
    HRESULT NuiTransformSmooth(NUI_SKELETON_FRAME *pFrame,
                               const NUI_TRANSFORM_SMOOTH_PARAMETERS *pParams);
    HRESULT NuiGetSensorCount(int *pCount);

Struct layouts follow NuiSensor.h / NuiSkeleton.h of SDK 1.8.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import POINTER, Structure, byref, c_float, c_int, c_int32, c_int64, c_uint32, c_void_p

NUI_INITIALIZE_FLAG_USES_SKELETON = 0x00000008
NUI_SKELETON_COUNT = 6
NUI_SKELETON_POSITION_COUNT = 20
NUI_SKELETON_NOT_TRACKED, NUI_SKELETON_POSITION_ONLY, NUI_SKELETON_TRACKED = 0, 1, 2
E_NUI_FRAME_NO_DATA = 0x83010001
S_OK = 0


class Vector4(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float), ("w", c_float)]


class NUI_SKELETON_DATA(Structure):
    _fields_ = [
        ("eTrackingState", c_int32),
        ("dwTrackingID", c_uint32),
        ("dwEnrollmentIndex", c_uint32),
        ("dwUserIndex", c_uint32),
        ("Position", Vector4),
        ("SkeletonPositions", Vector4 * NUI_SKELETON_POSITION_COUNT),
        ("eSkeletonPositionTrackingState", c_int32 * NUI_SKELETON_POSITION_COUNT),
        ("dwQualityFlags", c_uint32),
    ]


class NUI_SKELETON_FRAME(Structure):
    _fields_ = [
        ("liTimeStamp", c_int64),
        ("dwFrameNumber", c_uint32),
        ("dwFlags", c_uint32),
        ("vFloorClipPlane", Vector4),
        ("vNormalToGravity", Vector4),
        ("SkeletonData", NUI_SKELETON_DATA * NUI_SKELETON_COUNT),
    ]


class NUI_TRANSFORM_SMOOTH_PARAMETERS(Structure):
    _fields_ = [
        ("fSmoothing", c_float),
        ("fCorrection", c_float),
        ("fPrediction", c_float),
        ("fJitterRadius", c_float),
        ("fMaxDeviationRadius", c_float),
    ]


# Sanity checks of the struct layout (sizes implied by the SDK headers).
assert ctypes.sizeof(NUI_SKELETON_DATA) == 436, ctypes.sizeof(NUI_SKELETON_DATA)
assert ctypes.sizeof(NUI_SKELETON_FRAME) == 2664, ctypes.sizeof(NUI_SKELETON_FRAME)


class KinectError(RuntimeError):
    pass


def _hr(value: int) -> int:
    return value & 0xFFFFFFFF


class KinectV1SdkReader:
    """Minimal skeleton-stream reader for one Kinect v1 sensor.

    Args:
        dll: DLL name or path (``Kinect10.dll`` from SDK/Runtime 1.8).
        smoothing: apply ``NuiTransformSmooth`` with ``smooth_params`` (SDK's
            Holt double-exponential filter). None disables.
        timeout_ms: wait per ``read`` call.
    """

    DEFAULT_SMOOTHING = (0.5, 0.5, 0.5, 0.05, 0.04)

    def __init__(
        self,
        dll: str = "Kinect10.dll",
        smoothing: tuple[float, float, float, float, float] | None = DEFAULT_SMOOTHING,
        timeout_ms: int = 100,
    ) -> None:
        if sys.platform != "win32":
            raise KinectError(
                "Kinect SDK v1 is Windows-only. Run scripts/kinect_bridge_windows.py with a "
                "Windows Python and use `--source bridge` on Linux/WSL."
            )
        self.dll_name = dll
        self.smoothing = smoothing
        self.timeout_ms = timeout_ms
        self._dll = None
        self._frame = NUI_SKELETON_FRAME()
        self._smooth = NUI_TRANSFORM_SMOOTH_PARAMETERS(*smoothing) if smoothing else None

    def open(self) -> None:
        try:
            dll = ctypes.WinDLL(self.dll_name)  # type: ignore[attr-defined]
        except OSError as exc:
            raise KinectError(
                f"cannot load {self.dll_name}: {exc}. Install the Kinect for "
                "Windows SDK or Runtime v1.8."
            ) from exc
        dll.NuiInitialize.argtypes = [c_uint32]
        dll.NuiInitialize.restype = c_int32
        dll.NuiShutdown.argtypes = []
        dll.NuiShutdown.restype = None
        dll.NuiSkeletonTrackingEnable.argtypes = [c_void_p, c_uint32]
        dll.NuiSkeletonTrackingEnable.restype = c_int32
        dll.NuiSkeletonGetNextFrame.argtypes = [c_uint32, POINTER(NUI_SKELETON_FRAME)]
        dll.NuiSkeletonGetNextFrame.restype = c_int32
        dll.NuiTransformSmooth.argtypes = [
            POINTER(NUI_SKELETON_FRAME),
            POINTER(NUI_TRANSFORM_SMOOTH_PARAMETERS),
        ]
        dll.NuiTransformSmooth.restype = c_int32
        dll.NuiGetSensorCount.argtypes = [POINTER(c_int)]
        dll.NuiGetSensorCount.restype = c_int32
        count = c_int(0)
        hr = dll.NuiGetSensorCount(byref(count))
        if hr != S_OK or count.value < 1:
            raise KinectError(
                f"no Kinect sensor found (NuiGetSensorCount hr=0x{_hr(hr):08X}, "
                f"count={count.value})"
            )
        hr = dll.NuiInitialize(NUI_INITIALIZE_FLAG_USES_SKELETON)
        if hr != S_OK:
            raise KinectError(
                f"NuiInitialize failed: hr=0x{_hr(hr):08X} (sensor in use by "
                "another application, not powered, or driver problem?)"
            )
        hr = dll.NuiSkeletonTrackingEnable(None, 0)  # default: full-body, standing mode
        if hr != S_OK:
            dll.NuiShutdown()
            raise KinectError(f"NuiSkeletonTrackingEnable failed: hr=0x{_hr(hr):08X}")
        self._dll = dll
        self.sensor_count = count.value

    def close(self) -> None:
        if self._dll is not None:
            self._dll.NuiShutdown()
            self._dll = None

    def __enter__(self) -> KinectV1SdkReader:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read(self) -> dict | None:
        """Next skeleton frame as plain Python data, or None on timeout.

        Returns:
            ``{"frame_number", "sdk_timestamp_ms", "floor_clip_plane": [a, b, c, d],
            "skeletons": [{"tracking_id", "state", "position": [x, y, z],
            "joints": [[x, y, z] * 20], "joint_states": [int * 20], "quality_flags"}]}``
            with only TRACKED skeletons listed (metres, SDK skeleton space).
        """
        if self._dll is None:
            raise KinectError("reader not opened")
        hr = self._dll.NuiSkeletonGetNextFrame(self.timeout_ms, byref(self._frame))
        if _hr(hr) == E_NUI_FRAME_NO_DATA:
            return None
        if hr != S_OK:
            raise KinectError(f"NuiSkeletonGetNextFrame failed: hr=0x{_hr(hr):08X}")
        if self._smooth is not None:
            self._dll.NuiTransformSmooth(byref(self._frame), byref(self._smooth))
        f = self._frame
        skels = []
        for s in f.SkeletonData:
            if s.eTrackingState != NUI_SKELETON_TRACKED:
                continue
            skels.append(
                {
                    "tracking_id": int(s.dwTrackingID),
                    "state": int(s.eTrackingState),
                    "position": [s.Position.x, s.Position.y, s.Position.z],
                    "joints": [[p.x, p.y, p.z] for p in s.SkeletonPositions],
                    "joint_states": [int(v) for v in s.eSkeletonPositionTrackingState],
                    "quality_flags": int(s.dwQualityFlags),
                }
            )
        fc = f.vFloorClipPlane
        return {
            "frame_number": int(f.dwFrameNumber),
            "sdk_timestamp_ms": int(f.liTimeStamp),
            "floor_clip_plane": [fc.x, fc.y, fc.z, fc.w],
            "skeletons": skels,
        }


# --------------------------------------------------------------------------- image streams
# Used by the diagnostic viewer (scripts/kinect_diag_windows.py), not by the bridge.

NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX = 0x00000001
NUI_INITIALIZE_FLAG_USES_COLOR = 0x00000002
NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX = 0
NUI_IMAGE_TYPE_COLOR = 1
NUI_IMAGE_RESOLUTION_320x240 = 1
NUI_IMAGE_RESOLUTION_640x480 = 2


class NUI_IMAGE_VIEW_AREA(Structure):
    _fields_ = [("eDigitalZoom", c_int32), ("lCenterX", c_int32), ("lCenterY", c_int32)]


class NUI_IMAGE_FRAME(Structure):
    _fields_ = [
        ("liTimeStamp", c_int64),
        ("dwFrameNumber", c_uint32),
        ("eImageType", c_int32),
        ("eResolution", c_int32),
        ("pFrameTexture", c_void_p),  # INuiFrameTexture*
        ("dwFrameFlags", c_uint32),
        ("ViewArea", NUI_IMAGE_VIEW_AREA),
    ]


class NUI_LOCKED_RECT(Structure):
    _fields_ = [("Pitch", c_int32), ("size", c_int32), ("pBits", c_void_p)]


def _vtable_method(obj: int, index: int, restype, *argtypes):
    """Bound COM method ``index`` of the interface pointer ``obj`` (this-call on x64)."""
    vtbl = ctypes.cast(obj, POINTER(c_void_p))[0]
    fn = ctypes.cast(vtbl, POINTER(c_void_p))[index]
    proto = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)  # type: ignore[attr-defined]
    f = proto(fn)
    return lambda *a: f(obj, *a)


class KinectV1Diagnostics:
    """Depth (+ player index), colour and skeleton streams of one sensor, for debugging.

    ``read_depth()`` returns ``(width, height, bytes)`` of little-endian uint16 pixels where
    ``value >> 3`` is the depth in mm and ``value & 7`` the player index (0 = no person).
    ``read_color()`` returns ``(640, 480, BGRX bytes)``.
    """

    def __init__(self, dll: str = "Kinect10.dll", color: bool = True) -> None:
        if sys.platform != "win32":
            raise KinectError("Windows only")
        self.dll_name, self.use_color = dll, color
        self._dll = None
        self._depth = c_void_p()
        self._color = c_void_p()
        self.frame = NUI_SKELETON_FRAME()

    def open(self) -> None:
        dll = ctypes.WinDLL(self.dll_name)  # type: ignore[attr-defined]
        dll.NuiInitialize.argtypes = [c_uint32]
        dll.NuiImageStreamOpen.argtypes = [
            c_int32,
            c_int32,
            c_uint32,
            c_uint32,
            c_void_p,
            POINTER(c_void_p),
        ]
        dll.NuiImageStreamGetNextFrame.argtypes = [
            c_void_p,
            c_uint32,
            POINTER(POINTER(NUI_IMAGE_FRAME)),
        ]
        dll.NuiImageStreamReleaseFrame.argtypes = [c_void_p, POINTER(NUI_IMAGE_FRAME)]
        dll.NuiSkeletonTrackingEnable.argtypes = [c_void_p, c_uint32]
        dll.NuiSkeletonGetNextFrame.argtypes = [c_uint32, POINTER(NUI_SKELETON_FRAME)]
        dll.NuiCameraElevationGetAngle.argtypes = [POINTER(c_int32)]
        for name in (
            "NuiInitialize",
            "NuiImageStreamOpen",
            "NuiImageStreamGetNextFrame",
            "NuiImageStreamReleaseFrame",
            "NuiSkeletonTrackingEnable",
            "NuiSkeletonGetNextFrame",
            "NuiCameraElevationGetAngle",
        ):
            getattr(dll, name).restype = c_int32
        flags = NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON
        if self.use_color:
            flags |= NUI_INITIALIZE_FLAG_USES_COLOR
        hr = dll.NuiInitialize(flags)
        if hr != S_OK:
            raise KinectError(f"NuiInitialize failed: hr=0x{_hr(hr):08X}")
        self._dll = dll
        hr = dll.NuiImageStreamOpen(
            NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX,
            NUI_IMAGE_RESOLUTION_320x240,
            0,
            2,
            None,
            byref(self._depth),
        )
        if hr != S_OK:
            raise KinectError(f"depth stream open failed: hr=0x{_hr(hr):08X}")
        if self.use_color:
            hr = dll.NuiImageStreamOpen(
                NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, None, byref(self._color)
            )
            if hr != S_OK:
                raise KinectError(f"colour stream open failed: hr=0x{_hr(hr):08X}")
        hr = dll.NuiSkeletonTrackingEnable(None, 0)
        if hr != S_OK:
            raise KinectError(f"NuiSkeletonTrackingEnable failed: hr=0x{_hr(hr):08X}")

    def close(self) -> None:
        if self._dll is not None:
            self._dll.NuiShutdown()
            self._dll = None

    def set_elevation(self, degrees: int) -> None:
        """Drive the tilt motor (SDK range -27..27 deg). The SDK asks callers not to move the
        motor more than once per second / 15 times in 20 s, to protect it."""
        degrees = max(-27, min(27, int(degrees)))
        self._dll.NuiCameraElevationSetAngle.argtypes = [c_int32]
        self._dll.NuiCameraElevationSetAngle.restype = c_int32
        hr = self._dll.NuiCameraElevationSetAngle(degrees)
        if hr != S_OK:
            raise KinectError(f"NuiCameraElevationSetAngle failed: hr=0x{_hr(hr):08X}")

    def elevation_deg(self) -> int | None:
        a = c_int32(0)
        return a.value if self._dll.NuiCameraElevationGetAngle(byref(a)) == S_OK else None

    def _read_image(self, stream: c_void_p, timeout_ms: int) -> tuple[int, int, bytes] | None:
        pf = POINTER(NUI_IMAGE_FRAME)()
        hr = self._dll.NuiImageStreamGetNextFrame(stream, timeout_ms, byref(pf))
        if hr != S_OK:
            return None
        try:
            tex = pf.contents.pFrameTexture
            lock = _vtable_method(
                tex, 5, c_int32, ctypes.c_uint, POINTER(NUI_LOCKED_RECT), c_void_p, c_uint32
            )
            unlock = _vtable_method(tex, 7, c_int32, ctypes.c_uint)
            rect = NUI_LOCKED_RECT()
            lock(0, byref(rect), None, 0)
            try:
                data = ctypes.string_at(rect.pBits, rect.size) if rect.Pitch else b""
            finally:
                unlock(0)
            res = pf.contents.eResolution
        finally:
            self._dll.NuiImageStreamReleaseFrame(stream, pf)
        w, h = {1: (320, 240), 2: (640, 480)}.get(res, (0, 0))
        return w, h, data

    def read_depth(self, timeout_ms: int = 100):
        return self._read_image(self._depth, timeout_ms)

    def read_color(self, timeout_ms: int = 0):
        return self._read_image(self._color, timeout_ms) if self.use_color else None

    def read_skeleton_states(self, timeout_ms: int = 0) -> dict | None:
        hr = self._dll.NuiSkeletonGetNextFrame(timeout_ms, byref(self.frame))
        if hr != S_OK:
            return None
        f = self.frame
        fc = f.vFloorClipPlane
        return {
            "frame_number": int(f.dwFrameNumber),
            "floor_clip_plane": [fc.x, fc.y, fc.z, fc.w],
            "slots": [
                (
                    int(s.eTrackingState),
                    int(s.dwTrackingID),
                    [round(s.Position.x, 2), round(s.Position.y, 2), round(s.Position.z, 2)],
                )
                for s in f.SkeletonData
            ],
            "joints": [
                [(p.x, p.y, p.z) for p in s.SkeletonPositions] if s.eTrackingState == 2 else None
                for s in f.SkeletonData
            ],
        }
