# Dataset setup (KARD v2)

**Source:** Morana, M., Lo Re, G., & Gaglio, S. (2022). *KARD – Kinect Activity Recognition
Dataset*, Version 2. Mendeley Data. <https://doi.org/10.17632/k28dtm7tr6.2>
(page: <https://data.mendeley.com/datasets/k28dtm7tr6/2>). License: **CC BY 4.0** — keep the
citation above in any derived work. The data are never committed to this repository and are
never downloaded automatically.

## 1. Download (manual)

Download the archive parts from the Mendeley page into any folder, e.g.
`D:\Datasets\KARD - Kinect Activity Recognition Dataset\` (= `/mnt/d/Datasets/KARD - Kinect
Activity Recognition Dataset` in WSL). What was actually found on this machine:

| file | size | members | skeleton text files |
|---|---:|---:|---:|
| `KARD_a01_a02.tar` … `KARD_a17_a18.tar` (9 parts, 2 actions each) | 290–498 MB | 240 each | 60 `_realworld.txt` + 60 `_screen.txt` |
| `loadJoints.m` | 2.7 kB | – | MATLAB reference loader |

Each tar is flat (no sub-folders) and holds, per recording, `aAA_sSS_eEE.mp4` (RGB),
`aAA_sSS_eEE_depthmaps.txt.bz2` (depth), `_realworld.txt` and `_screen.txt`.

## 2. Extract only the skeleton files

```bash
make extract-data                                   # default KARD_TAR_DIR, see Makefile
make extract-data KARD_TAR_DIR="/mnt/e/somewhere/KARD"
```

`scripts/extract_kard.py` opens the tars read-only and copies only files matching
`aAA_sSS_eEE_(realworld|screen).txt` (plus `loadJoints.m`) to

```
data/raw/kard/
  KARD_a01_a02/a01_s01_e01_realworld.txt ...
  ...
  KARD_a17_a18/...
  loadJoints.m
```

Videos and depth maps are skipped (≈ 3.4 GB → ≈ 60 MB). A manifest is written to
`data/interim/extraction_manifest.json`. After extraction `data/raw/` is treated as
read-only by every other script.

## 3. Inspect and prepare

```bash
make inspect-data   # outputs/reports/kard_inspection_realworld.{json,md}
make prepare-data   # data/interim/kard_realworld.npz + data/processed/*.npz feature caches
```

Result on this machine (2026-10-20): **540 / 540 valid real-world sequences**, 0 malformed,
0 missing, 0 duplicates, 30 sequences per action, 54 per subject. See
[dataset_format.md](dataset_format.md) for the statistics.

## Using a different location

Every script takes `--raw-dir`/`--tar-dir`; the dataset YAML (`configs/dataset/*.yaml`)
holds `raw_dir`, `interim_path` and `processed_dir`.
