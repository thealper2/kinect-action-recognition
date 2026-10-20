# Skeleton action recognition — KARD + Xbox 360 Kinect v1

Deep-learning pipeline that trains its own models (MLP, BiLSTM, ST-GCN) on the skeleton
data of **KARD** and uses them for real-time recognition with an **Xbox 360 Kinect (v1)**.

```
KARD *_realworld.txt → parse/validate → preprocess → train (MLP | BiLSTM | ST-GCN)
   → subject-independent evaluation → self-contained checkpoint
   → Kinect v1 (SDK 1.8 on Windows) → TCP bridge → WSL recogniser → live label
```

Training/evaluation need no Kinect; the sensor is only used for live inference.

## Hızlı başlangıç (TR)

WSL Ubuntu içinde, proje klasöründe (`/mnt/d/work2/kinect-action-recognition`):

```bash
make install          # bağımlılıklar (~/venv içine) + paket
make data             # tar'lardan sadece iskelet .txt çıkar + inceleme raporu + önbellek
make check            # ruff + testler
make train-stgcn      # ST-GCN tam eğitim (100 epoch, early stopping)  → outputs/
make train-baseline   # MLP;   make train-lstm → BiLSTM;   make train-all → üçü birden
make experiments      # kısa deney seti: 14 koşu (~15–20 dk), kaldığı yerden devam eder
make experiments-full # tam set: 8 yapılandırma × 3 seed = 24 koşu
make loso             # (opsiyonel) leave-one-subject-out çapraz doğrulama
make evaluate         # tüm best.pt'ler için test metrikleri + karışıklık matrisleri
make notebooks        # analiz notebook'larını çalıştır
make infer-demo       # Kinect olmadan çevrimdışı çıkarım
make live             # Kinect: Windows köprüsü + WSL tanıyıcı (OpenCV penceresi, q ile çık)
make clean            # geçici dosyalar (__pycache__, .pytest_cache, .ruff_cache, *.tmp ...)
```

Değişkenler: `make train-stgcn SEED=43 EPOCHS=200 DEVICE=cuda`, `make live CKPT=...`,
`make extract-data KARD_TAR_DIR=/mnt/d/...`. Uzun eğitimleri `tmux`/`nohup` içinde başlatmanız
önerilir; loglar `outputs/logs/` altına da yazılır. Kesilen bir koşu:
`python scripts/train.py --config configs/experiments/e3_stgcn_xyz.yaml --resume outputs/checkpoints/e3_stgcn_xyz/seed42/last.pt`.

## Dataset

Morana, M., Lo Re, G., & Gaglio, S. (2022). *KARD – Kinect Activity Recognition Dataset*,
Version 2. Mendeley Data. <https://doi.org/10.17632/k28dtm7tr6.2> — **CC BY 4.0**.
18 actions × 10 subjects × 3 repetitions = 540 sequences, 15 joints. Download it manually
(never automatically, never committed). Setup: [docs/dataset_setup.md](docs/dataset_setup.md);
format, joint order, coordinate conventions and verified statistics:
[docs/dataset_format.md](docs/dataset_format.md).

Expected layout after `make extract-data` (only the skeleton text files are extracted from
the 9 official tar parts; videos/depth maps are skipped):

```
data/raw/kard/KARD_a01_a02/a01_s01_e01_realworld.txt ... (540 realworld + 540 screen)
data/interim/kard_realworld.npz          parsed cache        (make prepare-data)
data/processed/kard_realworld_<hash>.npz preprocessed cache  (make prepare-data)
```

Joint order (per 15-line frame block): Head, Neck, R Shoulder, R Elbow, R Hand, L Shoulder,
L Elbow, L Hand, Torso, R Hip, R Knee, R Foot, L Hip, L Knee, L Foot. Real-world coordinates
are millimetres, X right (sensor image), Y up, Z depth. Inspection result on this machine:
540/540 valid sequences, no missing/malformed/duplicate files, 30 per class, 54 per subject,
42–310 frames (median 112).

**Dataset facts vs. implementation choices** are labelled throughout the docs
(`[fact]`, `[verified]`, `[choice]`).

## Repository

```
configs/        dataset/ (data + preprocessing + augmentation, split), model/, training/,
                experiments/ (e1–e7 + suite.yaml), inference/realtime.yaml
src/skeleton_action/
  data/         KARD parsing, discovery, inspection, splits, torch Dataset
  preprocessing normalisation, resampling, velocity, augmentation
  models/       mlp.py, lstm.py, stgcn.py, graph.py (adjacency)
  training/     trainer (resume, early stopping, AMP), checkpoint I/O
  evaluation/   metrics, evaluator, plots, latency benchmark
  inference/    Predictor (checkpoint-only), RealtimeRecognizer, OpenCV HUD
  sensors/      SkeletonSource API, Kinect v1 mapping, SDK ctypes reader, TCP bridge,
                KARD replay + synthetic mock sources
scripts/        extract_kard, inspect_kard, prepare_dataset, train, evaluate, benchmark,
                run_experiments, infer, run_realtime, kinect_bridge_windows, build_notebooks
notebooks/      01 exploration, 02 skeleton visualisation, 03 baseline, 04 experiments
tests/          67 tests (synthetic fixtures + optional real-data check)
docs/           dataset_setup, dataset_format, preprocessing, architecture, evaluation,
                kinect_setup, troubleshooting
```

## Training

```bash
python scripts/train.py --config configs/experiments/e3_stgcn_xyz.yaml \
    --set training.seed=42 training.epochs=100          # any dotted override
python scripts/run_experiments.py --suite configs/experiments/suite.yaml
```

Models: [docs/architecture.md](docs/architecture.md). Defaults: AdamW 1e-3, cosine schedule
with warm-up, 100 epochs, batch 32, early stopping on validation macro-F1 (patience 30),
deterministic seeds, augmentation on the training subjects only.

## Evaluation protocol

Subject-independent: train s01–s06, validation s07–s08 (model selection), test s09–s10
(reported once). Optional LOSO. Metrics: accuracy, macro P/R/F1, weighted F1, per-class F1,
confusion matrix, parameters, checkpoint size, latency. Report results as *KARD
subject-independent evaluation* — see [docs/evaluation.md](docs/evaluation.md).

**Results:** none are reported here yet. Only 5-epoch smoke runs were executed to verify the
pipeline; run `make experiments` and read `outputs/metrics/suite_summary.md` / notebook 04.

## Offline inference

```bash
python scripts/infer.py --checkpoint CKPT --raw-file data/raw/kard/KARD_a17_a18/a17_s09_e01_realworld.txt
python scripts/infer.py --checkpoint CKPT --seq-id a14_s10_e02            # cached raw sequence
python scripts/infer.py --checkpoint CKPT --processed-seq-id a14_s10_e02  # preprocessed tensor
python scripts/infer.py --checkpoint CKPT --synthetic                     # smoke test only
```

Output: label, top-k scores, validity (+ reason), preprocessing and forward latency. Invalid
inputs (too short, missing core joints) return `valid: false` and no label.

## Kinect v1 (live)

Skeleton tracking for the Xbox 360 Kinect is only available through the Kinect for Windows
SDK 1.8 (Windows). A stdlib-only bridge (`scripts/kinect_bridge_windows.py`, ctypes →
`Kinect10.dll`) streams skeletons to the recogniser in WSL over localhost TCP:

```bash
make kinect-test   # hardware check from WSL via Windows Python
make live          # bridge + recogniser + OpenCV window
make realtime-replay RT_ARGS=--headless   # mock live stream from KARD test subjects
```

Mapping, coordinate conversion, tested/untested items and limitations:
[docs/kinect_setup.md](docs/kinect_setup.md). Tested on this PC: SDK initialisation and the
skeleton stream (~29 fps) and the Windows→WSL bridge; **not yet tested with a person in front
of the sensor**.

## Known limitations

* Small dataset (10 subjects); test estimates on 2 subjects are noisy — use seeds/LOSO.
* KARD clips are pre-segmented single actions; live streams are continuous, there is no
  "no action" class; rejection relies on confidence/margin thresholds.
* Different skeleton trackers (OpenNI-style KARD vs. Microsoft SDK); Neck/Torso/Foot are
  approximate mappings; KARD frame rate and camera pose are undocumented.
* Right-limb actions only in KARD; left-handed users → `swap_left_right: true`.
* Offline accuracy does not guarantee live accuracy.

## Installation

Python ≥ 3.10. `make install` installs `requirements-dev.txt` into `$(VENV)` (default
`~/venv`) and the package in editable mode. Core deps: PyTorch, NumPy, scikit-learn, pandas,
matplotlib, PyYAML; dev: pytest, ruff, nbformat/nbclient. OpenCV is optional (window only).
The Windows bridge needs only a stock 64-bit Python and the Kinect SDK/Runtime 1.8.
Docker (training/evaluation only): `make docker-build`.

Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md).

## License and attribution

Code: MIT. Data: KARD v2 by Morana, Lo Re & Gaglio (CC BY 4.0) — not redistributed; cite
the dataset as above when publishing results. ST-GCN after Yan, Xiong & Lin, "Spatial
Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018.
