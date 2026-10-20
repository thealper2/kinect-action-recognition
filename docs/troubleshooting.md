# Troubleshooting

| symptom | cause / fix |
|---|---|
| `No skeleton .txt files under data/raw/kard` | run `make extract-data` (set `KARD_TAR_DIR` if the tars are elsewhere) |
| `make extract-data` slow | reading 3.4 GB of tars over `/mnt/d` (9P) takes ~30 s; only ~60 MB is written |
| `FileExistsError: .../last.pt exists` | the run already exists: `--resume <last.pt>` to continue, `--overwrite` to restart, or change `SEED`/name |
| resume refused "does not match" | the checkpoint was trained with a different model or preprocessing config |
| `CUDA requested but not available` | `make env-info`; use `DEVICE=cpu`, or install a torch build for your CUDA (RTX 50xx needs cu128) |
| UserWarning about deterministic algorithms | expected with `deterministic: true` (`warn_only`); set `training.deterministic=false` for speed |
| `Weights only load failed` | checkpoint not produced by this project; checkpoints load with `weights_only=True` on purpose |
| `processed ... not found` in `infer.py --processed-seq-id` | `make prepare-data`; the cache digest must match the checkpoint's preprocessing |
| `cv2.error: The function is not implemented ... GTK+` / "headless build" warning | several OpenCV wheels share the `cv2` folder and the headless one won. Fix: `pip uninstall -y opencv-python-headless && pip install --force-reinstall --no-deps opencv-contrib-python==<same version>` (check with `python -c "import cv2; print(cv2.getBuildInformation())" \| grep GUI` → must not be `NONE`). The recogniser now falls back to headless instead of crashing |
| no OpenCV window in WSL | WSLg needs Windows 11 (`echo $DISPLAY` should print `:0`); otherwise add `RT_ARGS=--headless` |
| `[bridge] cannot load Kinect10.dll` | install Kinect for Windows SDK or Runtime 1.8; use 64-bit Python with the 64-bit DLL (`C:\Windows\System32`) |
| `NuiInitialize failed: hr=0x...` | sensor used by another app (close Kinect Studio / samples), power adapter unplugged, or USB 3 controller issue; check Device Manager |
| bridge says "retrying" forever | the WSL listener is not running or localhost forwarding is disabled (`localhostForwarding=false` in `.wslconfig`); run `make realtime-kinect` first or use `--host <WSL IP>` |
| live window is empty / "NO SKELETON TRACKED", bridge prints `no tracked person` | the window draws only the tracked skeleton, not the camera image. Run `make kinect-diag` (Windows viewer: depth image, player mask, tilt angle). Black depth = closer than 0.8 m or no IR return; seeing the ceiling = tilted up (`make kinect-diag TILT=0`). Place the sensor 0.6-1.8 m high with 2-3.5 m of free floor in front; stand back so the whole body is visible. Room light does not matter (IR), direct sunlight does |
| status stays `warming_up` | not enough tracked frames for the window, a tracking gap > 0.35 s, or core joints missing; stand 1.5–3.5 m away, full body visible |
| status `uncertain` most of the time | thresholds too strict for the domain shift; tune `confidence_threshold` / `margin_threshold`, try `window_seconds`, `source_fps: 25`, `torso: spine_shoulder_mid` |
| predictions mirrored (left/right swapped) | verify `flip_x: true`; for left-handed users set `swap_left_right: true` |
| `make live` leaves a bridge running | it exits when the recogniser closes the socket, or after 20 s without a receiver (`--give-up-after`); otherwise end `python.exe` in Task Manager |
| Docker: no Kinect | by design; Docker is for training/evaluation only |
