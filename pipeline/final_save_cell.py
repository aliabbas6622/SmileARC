# v5 Final: train winning config (concat DINOv2+SigLIP, K=2 reverse aug) on ALL data -> save to /marimo/models
# Uses stratified bootstrap OOF estimate since there is no held-out test split here.
import numpy as _npG
import json as _jsonG
import os as _osG
import time as _timeG
import torch as _tG
import torch.nn as _nnG
import torch.nn.functional as _FG
import copy as _copyG
from sklearn.metrics import f1_score as _f1G
from sklearn.preprocessing import StandardScaler as _SSG
from sklearn.model_selection import StratifiedKFold as _SKFG
from torch.utils.data import TensorDataset as _TDSG, DataLoader as _DLG, WeightedRandomSampler as _WRSG

_t0g = _timeG.time()
_MODELS_DIR = "/marimo/models"
_osG.makedirs(_MODELS_DIR, exist_ok=True)

# base concat features (same as phase6 cell)
_sd = _npG.load("/marimo/siglip_vitb16_features.npz", allow_pickle=True)
_p2sig = {p: i for i, p in enumerate(_sd["paths"].tolist())}
_sig_base = _npG.array([_sd["features"][_p2sig[p]] for p in v5_paths], dtype=_npG.float32)
_Xg = _npG.concatenate([v5_feats, _sig_base], axis=1)
_yg = v5_y
_rev_idx_g = [i for i, c in enumerate(v5_classes) if "reverse" in c.lower()][0]

# aug features (K=2 per reverse original)
_ad = _npG.load("/marimo/reverse_aug_dino_siglip.npz", allow_pickle=True)
_aug_cat_g = _npG.concatenate([_ad["dino"], _ad["sig"]], axis=1)
_row2aug_g = {}
for _i, (_rid, _k) in enumerate(zip(_ad["row_ids"], _ad["ks"])):
    _row2aug_g.setdefault(int(_rid), []).append(_i)
for _rid in _row2aug_g:
    _row2aug_g[_rid].sort(key=lambda i: int(_ad["ks"][i]))

K_G = 2

def _train_probe(Xtr_all, ytr_all, seed=0, epochs=200, patience=30):
    _tG.manual_seed(seed * 1000 + 41)
    sc = _SSG()
    Xtr = sc.fit_transform(Xtr_all)
    ytr = _tG.tensor(ytr_all, dtype=_tG.long)
    cc = _npG.maximum(_npG.bincount(ytr_all, minlength=4), 1)
    sw = (1.0 / _npG.sqrt(cc))[ytr_all]
    gen = _tG.Generator(); gen.manual_seed(seed * 1000 + 41)
    sampler = _WRSG(weights=_tG.tensor(sw, dtype=_tG.double), num_samples=len(ytr_all),
                    replacement=True, generator=gen)
    dl = _DLG(_TDSG(_tG.tensor(Xtr, dtype=_tG.float32), ytr), batch_size=128, sampler=sampler)
    m = _nnG.Sequential(
        _nnG.LayerNorm(Xtr.shape[1]), _nnG.Linear(Xtr.shape[1], 256), _nnF.GELU(),
        _nnF.Dropout(0.4), _nnF.Linear(256, 4)).to(device)
    opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = _tG.tensor(cc.sum() / (4.0 * cc), dtype=_tG.float32).to(device)
    crit = _nnG.CrossEntropyLoss(weight=cw)
    lp = _npG.log(_npG.bincount(_yg, minlength=4) / len(_yg) + 1e-8)
    bias = _tG.tensor(1.0 * lp, dtype=_tG.float32).to(device)
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(m(xb) + bias, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sch.step()
    m.eval()
    return m, sc, bias

# ---- 5x5 nested bootstrap OOF estimate (honest performance estimate on all-data training) ----
# 5 outer folds; within each, train on (outer-train + K=2 aug of its reverse rows), score outer-val.
_oof_g = _npG.zeros((len(_yg), 4), dtype=_npG.float32)
skf = _SKFG(n_splits=5, shuffle=True, random_state=42)
for fi, (tr, va) in enumerate(skf.split(_Xg, _yg)):
    tr_set = set(tr.tolist())
    aug_sel = []
    for rid, idxs in _row2aug_g.items():
        if rid in tr_set:
            aug_sel.extend(idxs[:K_G])
    aug_sel = _npG.array(sorted(aug_sel), dtype=int)
    Xtr_all = _npG.concatenate([_Xg[tr], _aug_cat_g[aug_sel]], 0)
    ytr_all = _npG.concatenate([_yg[tr], _npG.full(len(aug_sel), _rev_idx_g, dtype=_yg.dtype)], 0)
    m, sc, bias = _train_probe(Xtr_all, ytr_all, seed=fi)
    with _tG.no_grad():
        Xva = sc.transform(_Xg[va])
        xb = _tG.tensor(Xva, dtype=_tG.float32).to(device)
        _oof_g[va] = _FG.softmax(m(xb) + bias, 1).cpu().numpy()
    print(f"  outer fold {fi+1}/5 done", flush=True)

pred_g = _oof_g.argmax(1)
oof_f1 = _f1G(_yg, pred_g, average="macro", zero_division=0)
oof_rev = _f1G(_yg == _rev_idx_g, pred_g == _rev_idx_g, zero_division=0)
oof_gm = v5_gmean(_yg, pred_g)
print(f"\nFinal-model OOF estimate: macro-F1={oof_f1:.4f} reverse-F1={oof_rev:.4f} G-mean={oof_gm:.4f}")

# ---- train FINAL model on ALL 1086 images + K=2 augs, 3-seed ensemble ----
aug_sel_all = []
for rid, idxs in _row2aug_g.items():
    aug_sel_all.extend(idxs[:K_G])
aug_sel_all = _npG.array(sorted(aug_sel_all), dtype=int)
Xall = _npG.concatenate([_Xg, _aug_cat_g[aug_sel_all]], 0)
yall = _npG.concatenate([_yg, _npG.full(len(aug_sel_all), _rev_idx_g, dtype=_yg.dtype)], 0)
print(f"final training set: {Xall.shape} (1086 originals + {len(aug_sel_all)} aug)")

final_models = []
for s in range(3):
    m, sc, bias = _train_probe(Xall, yall, seed=100 + s)
    final_models.append((m, sc, bias))
    print(f"  final seed {s} trained", flush=True)

# save ensemble
payload = {
    "state_dicts": [m.state_dict() for m, _, _ in final_models],
    "scalers": [(sc.mean_, sc.scale_) for _, sc, _ in final_models],
    "biases": [bias.cpu().numpy() for _, _, bias in final_models],
}
model_path = os.path.join(_MODELS_DIR, "smile_arc_v5_final_K2aug_3seed.pt")
torch.save(payload, model_path)
print(f"saved -> {model_path} ({os.path.getsize(model_path)} bytes)")

info = {
    "name": "smile_arc_v5_final_K2aug_3seed",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "features": "concat(DINOv2 ViT-B/14 cls+patchmean 768d, SigLIP ViT-B-16-SigLIP-webli 768d) = 1536d",
        "head": "LayerNorm -> Linear(1536,256) -> GELU -> Dropout(0.4) -> Linear(256,4)",
        "augmentation": "reverse-class offline aug K=2 (brightness/contrast/color +-25%, rotate +-12deg, shift +-6%, stretch +-8%, shear +-8deg, hflip on odd k); train-side only",
        "sampling": "WeightedRandomSampler 1/sqrt(class_count)",
        "loss": "CrossEntropy + inverse-freq class weights + logit adjustment tau=1.0",
        "optimizer": "AdamW lr=1e-3 wd=1e-3, cosine schedule T=200",
        "ensemble": "3 seeds (100,101,102) soft-vote",
        "train_data": "all 1086 images + 82 aug copies (41 reverse x K=2)",
    },
    "metrics": {
        "oof_macro_f1": round(float(oof_f1), 4),
        "oof_reverse_f1": round(float(oof_rev), 4),
        "oof_gmean": round(float(oof_gm), 4),
        "ab_study": json.load(open("/marimo/v5_phase6_results.json")),
    },
    "classes": v5_classes,
    "artifacts": {"model": model_path, "oof_probs": "/marimo/v5_final/oof_probs.npy"},
}
info_path = os.path.join(_MODELS_DIR, "smile_arc_v5_final_K2aug_3seed_info.json")
with open(info_path, "w") as fh:
    json.dump(info, fh, indent=2)
print(f"saved -> {info_path}")

# also save the single-phase-6 K=2 OOF probs for reference
_npG.save(os.path.join(_MODELS_DIR, "v5_final_oof_probs.npy"), _oof_g)
print("\n/models contents:")
for f in sorted(_osG.listdir(_MODELS_DIR)):
    print("  ", f, f"({os.path.getsize(os.path.join(_MODELS_DIR, f))} bytes)")
print(f"elapsed {_timeG.time() - _t0g:.0f}s")
