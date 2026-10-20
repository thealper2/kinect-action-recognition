# Xbox 360 Kinect (v1) setup and live inference

## Architecture used on this PC

```
 Windows 11                                   WSL2 Ubuntu (venv, torch + CUDA)
 ┌───────────────────────────────┐   TCP     ┌──────────────────────────────────────┐
 │ Kinect for Windows SDK 1.8    │  JSON     │ scripts/run_realtime.py --source bridge│
 │ Kinect10.dll (NUI skeleton)   │  lines    │  BridgeSource (listens 0.0.0.0:8765)   │
 │ scripts/kinect_bridge_windows │ ───────▶  │  KinectV1Mapper → KARD 15 joints (mm)  │
 │  (stock Python, stdlib only,  │ localhost │  RealtimeRecognizer → model → HUD     │
 │   ctypes → Kinect10.dll)      │  :8765    │  OpenCV window via WSLg / headless    │
 └───────────────────────────────┘           └──────────────────────────────────────┘
```

Why not run the sensor inside WSL?

* WSL2 has no access to USB devices unless they are forwarded with `usbipd-win`, and then
  the Kinect would be detached from the Windows driver.
* On Linux, `libfreenect` gives depth/RGB but **no skeleton tracking**. Skeleton tracking for
  Kinect v1 on Linux needs OpenNI (1.x/2) + PrimeSense NiTE, which is discontinued and no
  longer distributed officially. The model needs tracked joints, so the Microsoft SDK
  (Windows) is the only maintained skeleton tracker for this sensor.
* The Kinect SDK 1.8 is already installed and working on this PC (`Kinect10.dll` present,
  device "Kinect for Windows Device" OK in Device Manager).

The bridge is plain Python + `ctypes` (no numpy/torch on Windows needed). It calls the flat
C API of `Kinect10.dll`: `NuiGetSensorCount`, `NuiInitialize(USES_SKELETON)`,
`NuiSkeletonTrackingEnable(NULL, 0)` (standing, full-body), `NuiSkeletonGetNextFrame`,
optionally `NuiTransformSmooth` (SDK Holt filter; defaults 0.5/0.5/0.5/0.05/0.04), and
`NuiShutdown`. Struct layouts follow `NuiSensor.h` (`NUI_SKELETON_FRAME` = 2664 bytes,
asserted at import).

## Commands

```bash
make kinect-test        # Windows Python prints 10 s of skeleton frames (hardware check)
make live               # bridge (Windows, background) + recogniser window (WSL); q/ESC quits
# or in two terminals:
make realtime-kinect    # WSL: listener + recogniser
make kinect-bridge      # WSL calls powershell.exe → Windows Python bridge
```

From a Windows terminal directly: `python scripts\kinect_bridge_windows.py --print` (local
test) or `--host 127.0.0.1 --port 8765` (stream to WSL). WSL2's default NAT mode forwards
Windows `localhost` to listeners inside WSL; with mirrored networking this also works. If a
firewall/VPN blocks it, run the recogniser on Windows instead (`--source kinect-sdk`, needs
numpy + torch in the Windows Python) or pass `--host <WSL IP>` to the bridge.

## What has been tested (2026-10-20, this PC)

| item | status |
|---|---|
| `Kinect10.dll` load, `NuiGetSensorCount` = 1, `NuiInitialize`, skeleton stream | **tested**: ~29 fps skeleton frames |
| Windows bridge → WSL `BridgeSource` over localhost TCP | **tested**: frames received, `no_person` status reported |
| Skeleton content with a person in front of the sensor | **not tested** (nobody was in view during development) |
| Joint mapping + recognition on live data | **not tested** with a person; mapping verified by unit tests and the documented conventions |
| Bridge protocol / reconnect / tracking-loss logic | unit-tested with synthetic frames |

Placement (2026-10-21 diagnosis on this PC): the sensor was tilted to +22 deg (seeing the ceiling)
and most of its view at 0 deg was closer than 0.8 m (no depth), so no skeleton could be tracked.
Kinect v1 needs the person 1.5-3.5 m away with the whole body in view; `make kinect-diag` shows
the depth image, the SDK's player segmentation and the tilt angle (`TILT=0` levels it).

First live check to do: `make kinect-test`, stand 2–3 m from the sensor, confirm
"1 tracked | joints tracked 20/20"; then `make live`.

## Joint mapping (SDK v1 20 joints → KARD 15 joints)

| KARD joint | SDK v1 source | exact | note |
|---|---|---|---|
| Head | Head | yes | |
| Neck | ShoulderCenter | approx. | SDK ShoulderCenter (between the shoulders) approximates OpenNI Neck |
| Right Shoulder | ShoulderRight | yes | |
| Right Elbow | ElbowRight | yes | |
| Right Hand | HandRight | yes | option `hand: wrist` → WristRight |
| Left Shoulder | ShoulderLeft | yes | |
| Left Elbow | ElbowLeft | yes | |
| Left Hand | HandLeft | yes | option `hand: wrist` → WristLeft |
| Torso | Spine | approx. | OpenNI Torso is the torso centre; SDK Spine sits lower; option `torso: spine_shoulder_mid` = mean(Spine, ShoulderCenter) |
| Right Hip | HipRight | yes | |
| Right Knee | KneeRight | yes | |
| Right Foot | AnkleRight | approx. | KARD Foot is at ankle height (knee→foot 412 mm ≈ thigh 440 mm); option `foot: foot` → FootRight |
| Left Hip | HipLeft | yes | |
| Left Knee | KneeLeft | yes | |
| Left Foot | AnkleLeft | approx. | as above |

Unused SDK joints: HipCenter, WristLeft/Right (unless `hand: wrist`), FootLeft/Right
(unless `foot: foot`). "exact" means the same semantic joint; the two skeleton trackers
(Microsoft vs. OpenNI/NiTE) still place joints differently, so even "exact" joints are not
anatomically identical.

## Coordinate conversion

| | Kinect SDK v1 skeleton space | KARD real-world (OpenNI) |
|---|---|---|
| unit | metre | millimetre |
| origin | IR camera | depth sensor |
| +Y | up | up |
| +Z | out of the sensor into the scene | away from the sensor (depth) |
| +X | towards the sensor's **left** (right-handed frame) | towards the **right** of the image (left-handed frame) |
| "Right" joints | user's anatomical right | user's anatomical right (verified, dataset_format.md) |

Therefore `x_kard = −1000·x_sdk, y_kard = 1000·y_sdk, z_kard = 1000·z_sdk`, Right→Right
(`flip_x: true`). A user facing the sensor then has Right Shoulder X < Left Shoulder X, as in
98.3 % of KARD frames. The sign convention of the SDK X axis comes from the SDK
documentation of skeleton space; it should be confirmed in the first live session (raise the
right hand: the HUD skeleton's raised hand must be drawn on the image-left side, like KARD).
After centring/scale normalisation, the residual differences are camera height/tilt (the SDK
reports a floor plane, KARD does not) and the distance to the sensor.

## Missing joints and tracking loss

* SDK joint state NOT_TRACKED → joint invalid; INFERRED → valid by default
  (`accept_inferred: false` to mask them). Invalid joints are interpolated over time inside
  the window; windows where < 80 % of joint entries or < 90 % of frames with all core joints
  (neck, torso, shoulders, hips) are valid produce no prediction.
* Multiple people: the closest tracked skeleton is chosen and followed by tracking id.
* No tracked person for > 0.5 s → buffer and smoothing reset, status `no_person`.
  Shorter dropouts keep the last decision; a gap > 0.35 s inside a window blocks inference
  until it has left the window.

## Real-time pipeline parameters (`configs/inference/realtime.yaml`)

Two windows by default (5th percentile and median KARD training length / 30 fps ≈ 2.1 s and
3.7 s), resampled on a uniform 30 Hz grid, preprocessed exactly like training, probabilities
averaged across windows, EMA smoothing (α = 0.35), inference every 5 tracked frames,
decision only if smoothed top-1 ≥ 0.6 and top-1 − top-2 ≥ 0.15 (otherwise `uncertain`).
Latency is shown in the HUD (model forward and full pipeline).

## Limitations / domain mismatch

* KARD was recorded with a Kinect and an OpenNI/NiTE-style 15-joint tracker (the
  real-world/screen conventions match OpenNI); the exact sensor model, tracker version, frame
  rate, camera height and tilt are not documented. Kinect SDK v1 joints are a different
  tracker's estimates; Neck/Torso/Foot are approximate mappings.
* KARD clips are pre-segmented, start/end at rest and contain one action each. Live
  streams contain transitions and idle periods that the model never saw; there is no
  "no action" class, so the confidence/margin thresholds are the only rejection mechanism.
* Subjects performed one-sided actions with the right limb; left-handed performance is out
  of distribution (`swap_left_right: true` mirrors the live skeleton as a workaround).
* 10 subjects, 30 examples per class: offline subject-independent accuracy does not
  guarantee live accuracy. Expect a drop and calibrate thresholds on your own recordings.
