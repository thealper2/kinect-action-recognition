# Preprocessing

Implementation: `src/skeleton_action/preprocessing/transforms.py` (deterministic) and
`augment.py` (training only). The configuration is a `PreprocessConfig` stored in every
checkpoint; the processed-feature cache file name contains its SHA-1 digest.

## Deterministic pipeline (train, val, test and live inference)

| step | default | notes |
|---|---|---|
| validity | finite and ≠ (0,0,0) | per (frame, joint) mask; KARD itself has no invalid entries |
| unit | mm → m (`input_unit_scale: 0.001`) | |
| missing joints | linear interpolation over time; hold at the ends; never-valid joints = torso | mask kept for the optional mask channel |
| centring | `first_frame`: subtract the torso of the **first** frame | `per_frame` and `none` available |
| scale | `torso_bones`: divide by the median over frames of the mean length of neck–torso, neck–shoulders, torso–hips | |
| orientation | `none` | `yaw_hips` / `yaw_shoulders` rotate about +Y so the first-frame L−R vector lies on +X |
| resampling | linear interpolation of the whole sequence to `num_frames` (64) | endpoints preserved |
| features | XYZ (C = 3) | `velocity: true` → C = 6; `append_mask: true` → +1 |

### Why first-frame centring

Per-frame torso centring makes every frame torso-relative and removes the body's global
trajectory: the torso drop in *Sit down*, the rise in *Stand up*, displacement in *Walk*,
the forward lean in *Bend*. First-frame centring removes only the subject's absolute position
in the room. Torso-centring does **not** remove viewpoint differences (camera height, tilt,
yaw); those are handled only partially by the yaw/pitch augmentation and the optional yaw
alignment. Experiment `e7_stgcn_perframe_center` measures the effect.

### Scale

Dividing by a torso-bone length makes inputs comparable across body sizes and distances
without using limb lengths that change apparent length during motion. "Without
normalisation" (experiment `e5`) means raw camera coordinates in metres.

### Velocity definition

`velocity_mode: frame_diff` (default): `v_t = x_t − x_{t−1}` on the **resampled, normalised**
sequence, `v_0 = v_1`. Because every sequence is resampled to T frames, this is a derivative
with respect to *normalised action time* (units: body-lengths per 1/(T−1) of the
sequence), not physical speed. `per_second` rescales by `(T−1)/duration` using `source_fps`
and the original length, i.e. body-lengths per second. Velocities are computed after
training augmentation so that they always match the coordinates they come with.

## Training-only augmentation (`AugmentConfig`)

Applied on the fly, in this order, only to the training subjects (val/test/live use the
deterministic pipeline):

1. temporal crop of a random contiguous 85–100 % of the raw frames (then normal
   resampling → mild speed variation);
2. optional joint dropout (off by default; for missing-joint studies);
3. normalisation (as above);
4. after resampling: rotation about Y ±15° and about X ±5°, isotropic scale 0.9–1.1,
   global translation N(0, 0.05) body units, per-coordinate jitter N(0, 0.01);
5. velocity / mask channels.

Mirroring is not used: KARD subjects performed one-sided actions with the right limb, so a
mirrored sample is not label-preserving with respect to the training distribution.

Reproducibility: augmentation draws from a dataset-owned `numpy.Generator` seeded from the
run seed (`num_workers: 0`); its state is saved in checkpoints so resumed runs are
bit-identical (tested).

## Caches

* `data/interim/kard_realworld.npz`: parsed sequences, concatenated `[ΣT, 15, 3]` + offsets
  + ids (no pickles).
* `data/processed/kard_realworld_<digest>.npz`: deterministic features `[N, T, 15, C]` for a
  given config (used for val/test and as a lookup for `scripts/infer.py
  --processed-seq-id`).
