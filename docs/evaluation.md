# Evaluation protocol and experiments

## Protocol: KARD subject-independent evaluation

* Primary split (`configs/dataset/split_subject_fixed.yaml`): train s01–s06 (324 seq.),
  validation s07–s08 (108), test s09–s10 (108). Every class has 18/6/6 sequences. No subject
  appears in two partitions (enforced by `SubjectSplit.validate`, tested). Whole sequences
  are the unit; nothing is split at frame or window level.
* The subject assignment is arbitrary but fixed [choice]; nothing is known about the
  subjects that would justify another grouping. With only two test subjects the test
  estimate has high variance → run several seeds and the optional LOSO protocol.
* Model selection (best epoch, early stopping, any hyper-parameter choice) uses validation
  macro-F1 only. Test metrics are computed once per run, after training.
* LOSO (`make loso`): 10 folds; fold *k* tests subject *k*, validates on subject *k+1*
  (cyclic), trains on the other 8.
* Results must be reported as "KARD subject-independent (s09–s10)" — they do not measure
  general action-recognition performance, and published KARD numbers often use different
  protocols (e.g. random or subject-dependent splits) and are not directly comparable.

## Metrics (`evaluation/metrics.py`, aggregated over all samples of the partition)

Accuracy, macro precision/recall/F1, weighted F1, per-class P/R/F1/support, confusion matrix,
largest off-diagonal confusions, parameter count, checkpoint size (and weights-only size),
forward latency (batch 1; CPU 1 thread, and CUDA if available).

Files per run (`<run> = <experiment>/seed<k>`):

```
outputs/checkpoints/<run>/{best.pt,last.pt,config.yaml,train_log.jsonl}
outputs/metrics/<run>/{summary.json,history.csv,{val,test}_metrics.json,
                       {val,test}_predictions.csv,{val,test}_per_class.csv}
outputs/figures/<run>/{curves.png,confusion_val.png,confusion_test.png}
outputs/metrics/suite_{runs.csv,summary.csv,summary.md}   (make experiments / aggregate)
```

## Controlled experiments (`configs/experiments/`, `make experiments`)

Same split, same training config. `make experiments` (`suite.yaml`, 14 runs): e1–e3 with
seeds 42/43/44, the ablations e4–e7 with seed 42 only, compared against e3 seed 42.
`make experiments-full` (`suite_full.yaml`, 24 runs) adds seeds 43/44 for the ablations
(finished runs are reused). Single-seed ablation differences of a few points are within
seed noise on 2 test subjects; confirm important ones with the full suite or LOSO.

| id | model | input | purpose |
|---|---|---|---|
| e1_mlp_xyz | MLP | normalised XYZ, T=64 | baseline |
| e2_lstm_xyz | BiLSTM | normalised XYZ, T=64 | recurrent model |
| e3_stgcn_xyz | ST-GCN | normalised XYZ, T=64 | main model (also T=64 point) |
| e4_stgcn_xyz_vel | ST-GCN | XYZ + velocity (C=6) | velocity ablation (vs e3) |
| e5_stgcn_no_norm | ST-GCN | raw camera XYZ (m) | normalisation ablation (vs e3) |
| e6_stgcn_T32 / T96 | ST-GCN | T=32 / 96 | sequence-length comparison (with e3) |
| e7_stgcn_perframe_center | ST-GCN | per-frame torso centring | checks the centring design choice |

The suite is resumable: finished runs are skipped, interrupted runs resume from `last.pt`.
Expected time on an RTX 5060 Ti: ~0.6–0.75 s wall time per ST-GCN epoch (measured on 15-epoch
timing runs, checkpoint writes included) → ~1.2 min per 100-epoch run plus ~10 s start-up
and evaluation; estimate ~15–20 min for the 14-run suite, ~30 min for the full 24 runs.
Early stopping can shorten runs.

## Status of results

Only 5-epoch **smoke runs** (`smoke_*`) were executed while building the project, to verify
the pipeline (loss decreases, gradients finite, checkpoints/metrics written). Their numbers
are not results and must not be reported. The full suite has deliberately **not** been run;
start it with `make experiments`. Notebook 04 then produces the comparison tables, the
confusion analysis and the pose-vs-dynamics similarity analysis for confused class pairs.
