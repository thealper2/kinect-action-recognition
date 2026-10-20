# KARD skeleton file format

Legend: **[fact]** = stated by the dataset authors (Mendeley page / `loadJoints.m`);
**[verified]** = checked on the files on this machine; **[choice]** = implementation decision.

## File naming  [fact, verified]

`aAA_sSS_eEE_<kind>.txt` — `AA` action 01–18, `SS` subject 01–10, `EE` repetition 01–03,
`kind` ∈ {`realworld`, `screen`}. Parsed by `skeleton_action.data.kard.parse_filename`, which
rejects out-of-range ids. 18 × 10 × 3 = 540 sequences per kind; all 540 present [verified].

## Classes  [fact]

| id | action | id | action | id | action |
|---:|---|---:|---|---:|---|
| 1 | Horizontal arm wave | 7 | Draw tick | 13 | Hand clap |
| 2 | High arm wave | 8 | Toss paper | 14 | Walk |
| 3 | Two hand wave | 9 | Forward kick | 15 | Phone call |
| 4 | Catch cap | 10 | Side kick | 16 | Drink |
| 5 | High throw | 11 | Take umbrella | 17 | Sit down |
| 6 | Draw X | 12 | Bend | 18 | Stand up |

Model class index = action id − 1 [choice]; checkpoints store both `class_names` and
`action_ids`.

## Content and parsing convention

* ASCII text, one joint per line: three whitespace-separated floats `x y z` [verified].
* Lines come in consecutive blocks of 15; block *f* is frame *f* and line *j* in the block
  is joint *j* in the order below [fact: `loadJoints.m` iterates `index1 += num_joints`
  per frame; verified: 1290 lines = 86 frames for `a01_s01_e01`; with this ordering the
  head is above the torso in 99.97 % and the torso above the right foot in > 99 % of all
  frames, and bone lengths are stable (e.g. neck–torso p05–p95 187–238 mm)].
* Hence row *r* ↦ (frame `r // 15`, joint `r % 15`) and a C-order reshape
  `[15·F, 3] → [F, 15, 3]` is exact. The parser **rejects** (never reshapes) files whose row
  count is not a multiple of 15, rows without exactly 3 values, non-numeric tokens, empty
  files; non-finite values are counted and the sequence is rejected; `(0,0,0)` joints
  (OpenNI's "not tracked") are counted and masked. Unit-tested with synthetic files whose
  values encode (frame, joint, coordinate).

Joint order [fact]:

| # | joint | # | joint | # | joint |
|---:|---|---:|---|---:|---|
| 0 | Head | 5 | Left Shoulder | 10 | Right Knee |
| 1 | Neck | 6 | Left Elbow | 11 | Right Foot |
| 2 | Right Shoulder | 7 | Left Hand | 12 | Left Hip |
| 3 | Right Elbow | 8 | Torso | 13 | Left Knee |
| 4 | Right Hand | 9 | Right Hip | 14 | Left Foot |

## Coordinate systems

**`_realworld.txt`** (primary input): millimetres [verified: torso depth median 2794 mm,
neck–torso 214 mm]; Y up [verified: head above torso in 99.97 % of frames]; Z = distance
from the sensor [verified: 1739–3489 mm]; X increases towards the right of the sensor image
[verified: "Right Shoulder"/"Right Hip" have smaller X than "Left" in 98.3 % of frames, and
`_screen.txt` u-coordinates order the same way]. This is the OpenNI/NiTE real-world
convention (a left-handed frame).

**`_screen.txt`** (optional, debugging only): u, v in pixels of a 640×480 image and depth in
mm (`loadJoints.m` divides z by 10 and flips v with 640 for plotting) [fact]. In the file
inspected the depth column equals the real-world Z [verified on `a01_s01_e01`]. Never mixed
with real-world coordinates in one model [choice].

**Body side** [verified]: in every one-sided action the joint labelled *Right* is the
moving one — right hand moves more than the left in 30/30 sequences of Horizontal arm wave,
High arm wave, High throw, Phone call, Drink; right foot in 30/30 Forward/Side kicks.
Toss paper is the exception (left hand moves more in 26/30). Subjects therefore used the
right limb, and KARD "Right" denotes the anatomical right.

## Statistics (real-world, 540 sequences)  [verified, `make inspect-data`]

* Frames: 63 630 total; per sequence min 42, p05 63, median 112, mean 117.8, p95 201, max 310.
* Median length per action: shortest Hand clap (84) / Side kick (85) / High throw (86); longest
  Phone call (176) and Walk (208).
* Non-finite values: 0. All-zero (untracked) joint entries: 0.
* X −1167…1504 mm, Y −1340…1166 mm, Z 1739…3489 mm.
* Bone lengths (median): shoulder width 347, hip width 202, upper arm 294, elbow→hand 294,
  hip→knee 440, knee→foot 412 mm. Knee→foot ≈ thigh length ⇒ the KARD "Foot" joint is at
  ankle height (used for the Kinect mapping).
* Frame count of `_realworld` == `_screen` for all 540 sequences.

## Frame rate

Not documented by the authors. The `.mp4` containers report 25 fps and have exactly one
video frame per skeleton frame [verified on `a14_s01_e01` (206) and `a13_s02_e03` (79)];
the Kinect v1 skeleton stream nominally runs at 30 fps. The pipeline works in frames and
resamples every sequence to a fixed length, so the rate only matters for (a) the
`per_second` velocity mode and (b) the real-time window length; both use the configurable
`source_fps` (default 30) [choice]. If live predictions look systematically time-stretched,
try `source_fps: 25` in `configs/inference/realtime.yaml`.
