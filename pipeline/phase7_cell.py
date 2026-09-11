# v5 Phase 7: label adjudication (conservative auto-fix) -> re-run champion config on cleaned labels -> save v2 model
# Policy: apply suggested_fix only when OOF margin >= 0.90; cap 40 fixes; record verdicts in CSV.
import numpy as _npJ
import json as _jsonJ
import csv as _csvJ
import os as _osJ
import time as _timeJ
import torch as _tJ
import torch.nn as _nnJ
import torch.nn.functional as _FJ
import copy as _copyJ
from sklearn.metrics import f1_score as _f1J, classification_report as _crJ
from sklearn.preprocessing import StandardScaler as _SSJ
from sklearn.model_selection import StratifiedKFold as _SKFJ
from torch.utils.data import TensorDataset as _TDSJ, DataLoader as _DLJ, WeightedRandomSampler as _WRSJ

_t0j = _timeJ.time()
_cls = v5_classes
_rev_idx = [i for i, c in enumerate(_cls) if "reverse" in c.lower()][0]

# ---------- 1) conservative auto-adjudication ----------
with open("/marimo/v5_label_review.csv") as f:
    _rows = list(_csvJ.DictReader(f))

MARGIN_MIN = 0.90
MAX_FIXES = 40
_path2row = {p: i for i, p in enumerate(v5_paths)}
y_clean = v5_y.copy()
applied = []
for r in _rows:
    if len(applied) >= MAX_FIXES:
        break
    if float(r["margin"]) < MARGIN_MIN:
        continue
    fix_name = r["suggested_fix"].strip()
    if fix_name not in _cls:
        continue
    idx = _path2row.get(r["path"])
    if idx is None:
        continue
    old_lbl = int(y_clean[idx])
    new_lbl = _cls.index(fix_name)
    if old_lbl != new_lbl:
        y_clean[idx] = new_lbl
        applied.append({"path": r["path"], "from": _cls[old_lbl], "to": fix_name,
                        "margin": float(r["margin"])})
        r["reviewer_verdict"] = f"auto-fixed: model v5 (margin={r['margin']})"

# write filled CSV (full 120 rows with verdicts)
with open("/marimo/v5_label_review_filled.csv", "w", newline="") as f:
    w = _csvJ.DictWriter(f, fieldnames=list(_rows[0].keys()))
    w.writeheader()
    w.writerows(_rows)

# also apply the recorded verdicts to the original review csv reviewer_verdict column? No - keep original untouched.
print(f"auto-fixed {len(applied)} labels (margin >= {MARGIN_MIN}, cap {MAX_FIXES})")
_from_cnt = {}
_to_cnt = {}
for a in applied:
    _from_cnt[a["from"]] = _from_cnt.get(a["from"], 0) + 1
    _to_cnt[a["to"]] = _to_cnt.get(a["to"], 0) + 1
print("  from:", _from_cnt)
print("  to:  ", _to_cnt)
print("  class counts after cleaning:", _npJ.bincount(y_clean))

# ---------- 2) champion config (concat feats + K=2 aug) on cleaned labels ----------
_sd = _npJ.load("/marimo/siglip_vitb16_features.npz", allow_pickle=True)
_p2sig = {p: i for i, p in enumerate(_sd["paths"].tolist())}
_sig_base = _npJ.array([_sd["features"][_p2sig[p]] for p in v5_paths], dtype=_npJ.float32)
_Xj = _npJ.concatenate([v5_feats, _sig_base], axis=1)

_ad = _npJ.load("/marimo/reverse_aug_dino_siglip.npz", allow_pickle=True)
_aug_cat_j = _npJ.concatenate([_ad["dino"], _ad["sig"]], axis=1)
_row2aug_j = {}
for _i, (_rid, _k) in enumerate(zip(_ad["row_ids"], _ad["ks"])):
    _row2aug_j.setdefault(int(_rid), []).append(_i)
for _rid in _row2aug_j:
    _row2aug_j[_rid].sort(key=lambda i: int(_ad["ks"][i]))

Kj = 2
# folds stratified on ORIGINAL labels -> identical to the baseline comparison
splits_j = list(_SKFJ(n_splits=5, shuffle=True, random_state=42).split(_Xj, v5_y))

def _train_fold_j(Xtr_all, ytr_all, seed, Xva_base, va_idx, y_va):
    _tJ.manual_seed(seed)
    _j_sc = _SSJ()
    Xtr = _j_sc.fit_transform(Xtr_all)
    Xva = _j_sc.transform(Xva_base)
    ytr = _tJ.tensor(ytr_all, dtype=_tJ.long)
    cc = _npJ.maximum(_npJ.bincount(ytr_all, minlength=4), 1)
    sw = (1.0 / _npJ.sqrt(cc))[ytr_all]
    gen = _tJ.Generator(); gen.manual_seed(seed)
    sampler = _WRSJ(weights=_tJ.tensor(sw, dtype=_tJ.double), num_samples=len(ytr_all),
                    replacement=True, generator=gen)
    dl = _DLJ(_TDSJ(_tJ.tensor(Xtr, dtype=_tJ.float32), ytr), batch_size=128, sampler=sampler)
    vdl = _DLJ(_TDSJ(_tJ.tensor(Xva, dtype=_tJ.float32),
                     _tJ.tensor(y_va, dtype=_tJ.long)), batch_size=256, shuffle=False)
    _j_m = _nnJ.Sequential(
        _nnJ.LayerNorm(Xtr.shape[1]), _nnJ.Linear(Xtr.shape[1], 256), _nnJ.GELU(),
        _nnJ.Dropout(0.4), _nnJ.Linear(256, 4)).to(device)
    opt = optim.AdamW(_j_m.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    cw = _tJ.tensor(cc.sum() / (4.0 * cc), dtype=_tJ.float32).to(device)
    crit = _nnJ.CrossEntropyLoss(weight=cw)
    lp = _npJ.log(_npJ.bincount(y_clean, minlength=4) / len(y_clean) + 1e-8)
    _j_bias = _tJ.tensor(1.0 * lp, dtype=_tJ.float32).to(device)
    bf1, pat, best = -1.0, 0, None
    for ep in range(200):
        _j_m.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(_j_m(xb) + _j_bias, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_j_m.parameters(), 5.0)
            opt.step()
        sch.step()
        _j_m.eval()
        pr, lb = [], []
        with torch.no_grad():
            for xb, yb in vdl:
                pr.extend((_j_m(xb.to(device)) + _j_bias).argmax(1).cpu().numpy())
                lb.extend(yb.numpy())
        vf1 = _f1J(lb, pr, average="macro", zero_division=0)
        if vf1 > bf1:
            bf1, pat, best = vf1, 0, _copyJ.deepcopy(_j_m.state_dict())
        else:
            pat += 1
            if pat >= 30:
                break
    _j_m.load_state_dict(best)
    _j_m.eval()
    with torch.no_grad():
        xb = _tJ.tensor(Xva, dtype=_tJ.float32).to(device)
        probs = _FJ.softmax(_j_m(xb) + _j_bias, 1).cpu().numpy()
    return _j_m, _j_sc, _j_bias, probs

def _oof_clean(y_target, seed=0):
    oof = _npJ.zeros((len(y_target), 4), dtype=_npJ.float32)
    for fi, (tr, va) in enumerate(splits_j):
        tr_set = set(tr.tolist())
        aug_sel = []
        for _j_rid, _j_idxs in _row2aug_j.items():
            if _j_rid in tr_set:
                aug_sel.extend(_j_idxs[:Kj])
        aug_sel = _npJ.array(sorted(aug_sel), dtype=int)
        # augmented copies inherit cleaned label of their original
        aug_labels = _npJ.array([y_clean[int(_ad["row_ids"][i])] for i in aug_sel], dtype=y_clean.dtype)
        Xtr_all = _npJ.concatenate([_Xj[tr], _aug_cat_j[aug_sel]], 0)
        ytr_all = _npJ.concatenate([y_target[tr], aug_labels], 0)
        _j_m, _j_sc, _j_bias, probs = _train_fold_j(Xtr_all, ytr_all, seed * 1000 + 61 + fi, _Xj[va], va, y_target[va])
        oof[va] = probs
    return oof

def _report_j(tag, oof, y_target):
    pred = oof.argmax(1)
    f1 = _f1J(y_target, pred, average="macro", zero_division=0)
    rev = _f1J(y_target == _rev_idx, pred == _rev_idx, zero_division=0)
    gm = v5_gmean(y_target, pred)
    print(f"{tag:34s} macro-F1={f1:.4f} reverse-F1={rev:.4f} G-mean={gm:.4f}")
    return {"macro_f1": round(float(f1), 4), "reverse_f1": round(float(rev), 4),
            "gmean": round(float(gm), 4)}

print("\n=== Phase 7: cleaned labels vs original (identical folds, K=2 aug) ===")
_p7 = {}
_oof_orig = _oof_clean(v5_y)
_p7["original_labels"] = _report_j("original labels (ref)", _oof_orig, v5_y)
_oof_clean_probs = _oof_clean(y_clean)
_p7["cleaned_labels"] = _report_j("cleaned labels", _oof_clean_probs, y_clean)

print("\nPer-class report (cleaned):")
print(_crJ(y_clean, _oof_clean_probs.argmax(1), target_names=_cls, zero_division=0))

# ---------- 3) train final v2 ensemble on cleaned labels, save ----------
_j_aug_sel_all = []
for _j_rid, _j_idxs in _row2aug_j.items():
    _j_aug_sel_all.extend(_j_idxs[:Kj])
_j_aug_sel_all = _npJ.array(sorted(_j_aug_sel_all), dtype=int)
aug_labels_all = _npJ.array([y_clean[int(_ad["row_ids"][i])] for i in _j_aug_sel_all], dtype=y_clean.dtype)
Xall_j = _npJ.concatenate([_Xj, _aug_cat_j[_j_aug_sel_all]], 0)
yall_j = _npJ.concatenate([y_clean, aug_labels_all], 0)
print(f"v2 final training set: {Xall_j.shape}")

final_models_j = []
for _j_s in range(3):
    _j_m, _j_sc, _j_bias, _ = _train_fold_j(Xall_j, yall_j, 200 + _j_s, _Xj[:2], [0, 1], y_clean[:2])
    final_models_j.append((_j_m, _j_sc, _j_bias))
    print(f"  v2 seed {_j_s} trained", flush=True)

payload_j = {
    "state_dicts": [_j_m.state_dict() for _j_m, _, _ in final_models_j],
    "scalers": [(_j_sc.mean_, _j_sc.scale_) for _, _j_sc, _ in final_models_j],
    "biases": [_j_bias.cpu().numpy() for _, _, _j_bias in final_models_j],
}
model_path_j = "/marimo/models/smile_arc_v5_clean_K2aug_3seed.pt"
torch.save(payload_j, model_path_j)
print(f"saved -> {model_path_j} ({_osJ.path.getsize(model_path_j)} bytes)")

_p7["n_fixes"] = len(applied)
_p7["fixes"] = applied
info_j = {
    "name": "smile_arc_v5_clean_K2aug_3seed",
    "date": _timeJ.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "features": "concat(DINOv2 ViT-B/14 768d, SigLIP ViT-B-16 768d) = 1536d",
        "head": "LayerNorm -> Linear(1536,256) -> GELU -> Dropout(0.4) -> Linear(256,4)",
        "augmentation": "reverse-class offline aug K=2 (train-side only)",
        "labels": f"cleaned: {len(applied)} auto-fixed (OOF margin >= {MARGIN_MIN}, cap {MAX_FIXES})",
        "sampling": "WeightedRandomSampler 1/sqrt(class_count)",
        "loss": "CE + inverse-freq weights + logit adjustment tau=1.0",
        "optimizer": "AdamW lr=1e-3 wd=1e-3, cosine T=200",
        "ensemble": "3 seeds (200,201,202)",
    },
    "metrics": {
        "oof_original_labels": _p7["original_labels"],
        "oof_cleaned_labels": _p7["cleaned_labels"],
        "n_fixes": len(applied),
    },
    "classes": _cls,
    "artifacts": {
        "model": model_path_j,
        "review_csv": "/marimo/v5_label_review_filled.csv",
    },
}
with open("/marimo/models/smile_arc_v5_clean_K2aug_3seed_info.json", "w") as fhJ:
    _jsonJ.dump(info_j, fhJ, indent=2)
print("saved -> /marimo/models/smile_arc_v5_clean_K2aug_3seed_info.json")
print(f"elapsed {_timeJ.time() - _t0j:.0f}s")
