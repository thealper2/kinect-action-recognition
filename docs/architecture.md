# Model architectures

All models take `x ∈ ℝ^{B×T×J×C}` (J = 15, C = 3 or 6) and return logits `ℝ^{B×18}`. They
are built from checkpoint metadata by `skeleton_action.models.build_model`; parameter counts
are for C = 3, T = 64 with the default configs.

## Model A — MLP baseline (`configs/model/mlp.yaml`, 114 k parameters)

```
[B,T,15,C] → reshape [B,T,15C]
  → frame MLP: (Linear → LayerNorm → GELU → Dropout) × [256, 128]     → [B,T,128]
  → temporal aggregation                                              → [B,A]
  → Linear(A,128) → LayerNorm → GELU → Dropout → Linear(128,18)
```

Aggregation: `segments` (default) = adaptive average pooling into K = 4 equal temporal
segments, concatenated (A = 512). It keeps coarse order at little cost. `mean`/`meanmax`
are order-invariant and cannot separate time-reversed actions (Sit down vs Stand up) from
positions alone — a unit test checks this property. `flatten` keeps exact order but ties
the model to T and multiplies the head size by T.

## Model B — BiLSTM (`configs/model/lstm.yaml`, 189 k parameters)

```
[B,T,15C] → Linear(15C,96) → LayerNorm → ReLU → Dropout       → [B,T,96]
  → 2-layer BiLSTM, hidden 64 (dropout between layers)          → [B,T,128]
  → mean over valid steps (or last states)                       → [B,128]
  → LayerNorm → Dropout → Linear(128,18)
```

Optional `lengths` packs padded batches (`pack_padded_sequence`) and masks the mean;
padding never enters the recurrence (tested). Initialisation: Xavier input weights,
orthogonal recurrent weights per gate, forget-gate bias 1. Kept small on purpose: 324
training sequences.

## Model C — ST-GCN (`configs/model/stgcn.yaml`, 503 k parameters)

Spatial-temporal GCN after Yan, Xiong & Lin (AAAI 2018), compact variant.

### Graph

Nodes: the 15 KARD joints. Edges (undirected kinematic tree, 14 bones):
head–neck, neck–R/L shoulder, shoulder–elbow, elbow–hand (both sides), neck–torso,
torso–R/L hip, hip–knee, knee–foot (both sides). No shortcut edges [choice].

Spatial-configuration partitioning with the torso as root (`models/graph.py`): with
`d(·)` the hop distance to the torso and `N(i)` the 1-hop neighbourhood of joint *i*
including *i*,

* `A₀[j,i] = 1` if `j ∈ N(i)`, `d(j) = d(i)` (only *i* itself in a tree),
* `A₁[j,i] = 1` if `j ∈ N(i)`, `d(j) < d(i)` (parent, centripetal),
* `A₂[j,i] = 1` if `j ∈ N(i)`, `d(j) > d(i)` (children, centrifugal),

then every subset is normalised with the in-degree of the full neighbourhood,
`Âₖ = Aₖ D⁻¹`, `D_ii = Σⱼ (A+I)ⱼᵢ`. Tests check: `Σₖ Âₖ` has the support of `A + I`, which is
symmetric; `Â₁ᵀ` and `Â₂` have the same support; columns of `Σₖ Âₖ` sum to 1; all finite.
Strategies `distance` (K = 2) and `uniform` (K = 1) are also available.

### Layers

```
x [N,T,V,C] → [N,V·C,T] → BatchNorm1d(V·C) (data BN) → [N,C,T,V]
6 × ST-GCN block (channels 32,32,64,64,128,128; temporal strides 1,1,2,1,2,1):
    GCN:   Y = Σₖ (Wₖ X) (Âₖ ⊙ Mₖ)          Wₖ = 1×1 conv, C_in → C_out per subset
                                          Mₖ = learnable edge importance (init 1)
           implemented as Conv2d(C_in, K·C_out, 1) + einsum('nkctv,kvw->nctw')
    TCN:   BN → ReLU → Conv2d(C_out, C_out, (9,1), stride (s,1), pad (4,0)) → BN → Dropout(0.2)
    out:   ReLU(TCN(GCN(x)) + residual)     residual: identity, or 1×1 conv + BN when the
                                            shape changes; none in the first block
global average pooling over (T, V) → [N,128] → Dropout(0.3) → Linear(128, 18)
```

Shapes for T = 64: `[N,3,64,15] → [N,32,64,15] → [N,64,32,15] → [N,128,16,15] → [N,128]`.

Not included (possible future work): adaptive/learned adjacency (2s-AGCN), multi-stream
bone/joint fusion, attention.

## Training

`skeleton_action/training/trainer.py`: AdamW (lr 1e-3, wd 1e-2), 5-epoch linear warm-up +
cosine decay to 1e-5, 100 epochs, batch 32, gradient clipping 1.0, cross-entropy (no class
weights: KARD is perfectly balanced), early stopping on validation macro-F1 (patience 30),
optional bf16/fp16 AMP. Best checkpoint = highest validation macro-F1; the test subjects are
evaluated once after training with that checkpoint. Checkpoints (`best.pt`, `last.pt`) are
self-contained: weights, architecture name + hyper-parameters, optimiser/scheduler/scaler
state, epoch, best metric, class mapping, dataset id, preprocessing/augmentation/split
config, seed, full resolved config, software versions, RNG states, history. They load with
`torch.load(weights_only=True)`.
