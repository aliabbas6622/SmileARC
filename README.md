# 😀 SmileARC — Smile Arc Classification

Deep-learning pipeline that classifies **nose-to-jaw crop images** into 4 smile-arc classes.
Built as a series of experiments on a small, heavily imbalanced clinical dataset, evolving
from fine-tuned CNNs (macro-F1 0.04 → 0.58) to a frozen-foundation-features + MLP-probe
approach that reached **0.71 macro-F1**.

## The Problem

| class | images | share |
|---|---:|---:|
| consonant | 483 | 44.5% |
| not available | 431 | 39.7% |
| straight- non consonant | 131 | 12.1% |
| **reverse- non consonant** | **41** | **3.8%** |

1,086 images total. The `reverse` class — clinically the most important — has only
41 examples, making reverse-F1 the hardest metric to move.

## Results Journey

Every number below is 5-fold out-of-fold macro-F1 (or test-set F1 for the CNN era).

| # | Approach | Macro-F1 | Reverse-F1 | Outcome |
|---|---|:---:|:---:|---|
| 1 | ViT-Large fine-tuned | 0.04 | 0.00 | collapsed to one class |
| 2–3 | Swin-Tiny (+ composite images) | 0.17 → 0.26 | 0.13 | promising but weak |
| 4 | **EfficientNet-B2 + Focal Loss** | **0.58** | 0.55 | best CNN era result |
| 5 | Hierarchical (binary + 3-class) | 0.24 | — | cascading errors — rejected |
| 6 | SSL pretraining (SimCLR/MoCo) | 0.13–0.26 | 0.00 | too little data — rejected |
| 7 | ConvNeXt-Tiny 3-model ensemble | 0.56 | 0.25 | 10× bigger, no better |
| 8 | **v4: DINOv2 features + MLP probe** | 0.642 | 0.518 | paradigm shift 🔄 |
| 9 | v5 Phase 1: seed-ens / mixup / probe family | ≤ 0.629 | — | nothing beat the locked baseline |
| 10 | v5 Phase 3: two-stage gate | 0.648 | 0.506 | gain < adoption rule — rejected |
| 11 | v5 Phase 4: **+ SigLIP concat (1536-d)** | 0.650 | 0.483 | ✅ adopted |
| 12 | v5 Phase 6: **reverse-class augmentation (K=2)** | **0.667** | **0.532** | ✅ adopted |
| 13 | v5 Phase 7: **+ label-error auto-fixes (25)** | **0.706** | **0.632** | ✅ adopted* |

\* *Label fixes were auto-adjudicated (OOF margin ≥ 0.90) and measured against the cleaned
targets, so 0.706 is an optimistic upper bound — the direction is real, the magnitude needs
a human-reviewed pass to confirm.*

## Winning Recipe

```
image (nose-to-jaw crop)
  ├─ DINOv2 ViT-B/14  → CLS + patch-mean  (768-d)
  └─ SigLIP ViT-B-16  → pooled embedding  (768-d)
          ↓ concat → 1536-d
StandardScaler → LayerNorm → Linear(256) → GELU → Dropout(0.4) → Linear(4)
```

**Training details that matter:**
- WeightedRandomSampler with `1/√class_count` sampling
- CE loss + inverse-frequency class weights
- **Logit adjustment**: `logits + 1.0 · log(class_prior)` — the single biggest win for reverse-F1
- AdamW (lr 1e-3, wd 1e-3), cosine schedule, 200 epochs, early stop on val macro-F1
- **Reverse-class offline augmentation (K=2)**: brightness/contrast ±25%, color ±15%,
  rotation ±12°, shift ±6%, stretch ±8%, shear ±8°, hflip — injected **train-side only**
  per fold, so validation stays clean (no leakage)
- 3-seed ensemble soft-voting for the final model
- **Label-error review**: cleanlab-style OOF ranking flagged 120 suspicious labels;
  25 highest-confidence fixes applied and documented

## Key Lessons Learned

1. **Frozen foundation-model features beat fine-tuned CNNs** on small datasets —
   +0.06 macro-F1 with a 20× smaller trainable head, and no epoch-long training runs.
2. **Two complementary foundation models > one** — SigLIP's image-text alignment adds
   signal DINOv2 misses.
3. **Logit adjustment + sqrt-sampling** is the right recipe for extreme imbalance;
   inverse-frequency sampling overshoots.
4. **Targeted offline augmentation works** — but saturates fast: K=2 per image was optimal
   (K=1 under-fits, K=4/8 over-saturates).
5. **Label quality caps model quality** — the 120-suspect review was the biggest single
   lever at the end (0.66 → 0.71).
6. Things that did **not** work: ASL loss (multi-label only), hierarchical classification,
   SSL pretraining at this scale, mixup on features, seed ensembling (slightly hurt).

## Repository Layout

```
SmileClassification_Fixed (1).ipynb   # experiment notebook (CNN era)
```

Pipeline cells + inference script live in the working notebook / project folder:
- `v5_lib_cell.py` — shared library: features, stratified splits, MLP probe trainer
- `v5_phase1..5_cell.py` — A/B experiments, label review, two-stage, SigLIP, final report
- `phase6_cell.py` / `phase6b_cell.py` — reverse-class augmentation + K sweep
- `phase7_cell.py` — label adjudication + re-run with fixes
- `final_save_cell.py` — trains final 3-seed ensemble on all data, saves checkpoint + info JSON
- `predict_smile_arc.py` — standalone inference (DINOv2+SigLIP → ensemble probe)
- `mp_extract_script.py` — MediaPipe 20-d lip-geometry feature extraction

Model checkpoints & info JSONs: [`huggingface.co/aliabbas6622/checkpoints`](https://huggingface.co/aliabbas6622/checkpoints)
Dataset: [`huggingface.co/datasets/aliabbas6622/smile`](https://huggingface.co/datasets/aliabbas6622/smile)

## Inference

```bash
python predict_smile_arc.py image1.jpg image2.jpg \
    --model smile_arc_v5_final_K2aug_3seed.pt
```

Requires `torch`, `torchvision`, `open_clip_torch`, `PIL`, `numpy`.
DINOv2 weights auto-download via torch.hub on first run.

## Next Steps

- [ ] Human review of remaining ~95 label suspects (HTML review artifact ready)
- [ ] Collect more `reverse`-class images (41 is the fundamental bottleneck)
- [ ] Nested-CV estimate of the label-cleaning gain
- [ ] MediaPipe geometry + XGBoost hybrid (features already extractable)
