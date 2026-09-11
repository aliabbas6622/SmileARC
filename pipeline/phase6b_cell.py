# v5 Phase 6b: fine-grained K sweep (K=1, K=3) around the K=2 sweet spot + multi-seed ensembling
import numpy as _npH
import json as _jsonH
import os as _osH
import time as _timeH
import torch as _tH
import torch.nn as _nnH
import torch.nn.functional as _FH
import copy as _copyH
from sklearn.metrics import f1_score as _f1H
from sklearn.preprocessing import StandardScaler as _SSH
from torch.utils.data import TensorDataset as _TDSH, DataLoader as _DLH, WeightedRandomSampler as _WRSH

_t0h = _timeH.time()
# base concat features (same as phase6 cell)
_sd = _npH.load("/marimo/siglip_vitb16_features.npz", allow_pickle=True)
_p2sig = {p: i for i, p in enumerate(_sd["paths"].tolist())}
_sig_base = _npH.array([_sd["features"][_p2sig[p]] for p in v5_paths], dtype=_npH.float32)
_Xh = _npH.concatenate([v5_feats, _sig_base], axis=1)
_yh = v5_y
_rev_idx_h = [i for i, c in enumerate(v5_classes) if "reverse" in c.lower()][0]

# aug features
_ad = _npH.load("/marimo/reverse_aug_dino_siglip.npz", allow_pickle=True)
_aug_cat_h = _npH.concatenate([_ad["dino"], _ad["sig"]], axis=1)
_row2aug_h = {}
for _i, (_rid, _k) in enumerate(zip(_ad["row_ids"], _ad["ks"])):
    _row2aug_h.setdefault(int(_rid), []).append(_i)
for _rid in _row2aug_h:
    _row2aug_h[_rid].sort(key=lambda i: int(_ad["ks"][i]))

def _train_fold(Xtr_all, ytr_all, seed, Xva_base, va_idx):
    """Train one fold probe with the locked recipe; return val probs."""
    _tH.manual_seed(seed)
    sc = _SSH()
    Xtr = sc.fit_transform(Xtr_all)
    Xva = sc.transform(Xva_base)
    ytr = _tH.tensor(ytr_all, dtype=_tH.long)
    cc = _npH.maximum(_npH.bincount(ytr_all, minlength=4), 1)
    sw = (1.0 / _npH.sqrt(cc))[ytr_all]
    gen = _tH.Generator(); gen.manual_seed(seed)
    sampler = _WRSH(weights=_tH.tensor(sw, dtype=_tH.double), num_samples=len(ytr_all),
                    replacement=True, generator=gen)
    dl = _DLH(_TDSH(_tH.tensor(Xtr, dtype=_tH.float32), ytr), batch_size=128, sampler=sampler)
    vdl = _DLH(_TDSH(_tH.tensor(Xva, dtype=_tH.float32),
                     _tH.tensor(_yh[va_idx], dtype=_tH.long)), batch_size=256, shuffle=False)
    m = _nnH.Sequential(
        _nnH.LayerNorm(Xtr.shape[1]), _nnH.Linear(Xtr.shape[1], 256), _nnH.GELU(),
        _nnH.Dropout(0.4), _nnH.Linear(256, 4)).to(device)
    opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    cw = _tH.tensor(cc.sum() / (4.0 * cc), dtype=_tH.float32).to(device)
    crit = _nnH.CrossEntropyLoss(weight=cw)
    lp = _npH.log(_npH.bincount(_yh, minlength=4) / len(_yh) + 1e-8)
    bias = _tH.tensor(1.0 * lp, dtype=_tH.float32).to(device)
    bf1, pat, best = -1.0, 0, None
    for ep in range(200):
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
        pr, lb = [], []
        with torch.no_grad():
            for xb, yb in vdl:
                pr.extend((m(xb.to(device)) + bias).argmax(1).cpu().numpy())
                lb.extend(yb.numpy())
        vf1 = _f1H(lb, pr, average="macro", zero_division=0)
        if vf1 > bf1:
            bf1, pat, best = vf1, 0, _copyH.deepcopy(m.state_dict())
        else:
            pat += 1
            if pat >= 30:
                break
    m.load_state_dict(best)
    m.eval()
    with torch.no_grad():
        xb = _tH.tensor(Xva, dtype=_tH.float32).to(device)
        probs = _FH.softmax(m(xb) + bias, 1).cpu().numpy()
    return probs

def _oof_for_K(K, seeds=(0,)):
    """OOF probs averaged over seeds; fold-safe K aug injection on train side."""
    splits = v5_splits()
    oof_acc = _npH.zeros((len(_yh), 4), dtype=_npH.float32)
    for seed in seeds:
        oof = _npH.zeros((len(_yh), 4), dtype=_npH.float32)
        for fi, (tr, va) in enumerate(splits):
            tr_set = set(tr.tolist())
            aug_sel = []
            for rid, idxs in _row2aug_h.items():
                if rid in tr_set:
                    aug_sel.extend(idxs[:K])
            aug_sel = _npH.array(sorted(aug_sel), dtype=int)
            Xtr_all = _npH.concatenate([_Xh[tr], _aug_cat_h[aug_sel]], 0)
            ytr_all = _npH.concatenate([_yh[tr], _npH.full(len(aug_sel), _rev_idx_h, dtype=_yh.dtype)], 0)
            oof[va] = _train_fold(Xtr_all, ytr_all, seed * 1000 + 51 + fi, _Xh[va], va)
        oof_acc += oof
    return oof_acc / len(seeds)

def _report_h(tag, oof):
    pred = oof.argmax(1)
    f1 = _f1H(_yh, pred, average="macro", zero_division=0)
    rev = _f1H(_yh == _rev_idx_h, pred == _rev_idx_h, zero_division=0)
    gm = v5_gmean(_yh, pred)
    print(f"{tag:30s} macro-F1={f1:.4f} reverse-F1={rev:.4f} G-mean={gm:.4f}")
    return {"macro_f1": round(float(f1), 4), "reverse_f1": round(float(rev), 4),
            "gmean": round(float(gm), 4)}

print("=== Phase 6b: fine K sweep + multi-seed ensembling (fold-safe) ===")
_p6b = {}

# single-seed K=1 and K=3 (K=2 single seed = 0.6669 from phase6)
_oof_k1 = _oof_for_K(1, seeds=(0,))
_p6b["K1_seed0"] = _report_h("K=1 (seed 0)", _oof_k1)

_oof_k3 = _oof_for_K(3, seeds=(0,))
_p6b["K3_seed0"] = _report_h("K=3 (seed 0)", _oof_k3)

# multi-seed ensembles (3 seeds averaged) at the best K values
for K in [2, 3]:
    _oof_ms = _oof_for_K(K, seeds=(0, 1, 2))
    _p6b[f"K{K}_3seed"] = _report_h(f"K={K} (3-seed ensemble)", _oof_ms)

# save the best OOF probs
best_tag = max(_p6b, key=lambda k: _p6b[k]["macro_f1"])
print(f"\nbest: {best_tag} -> {_p6b[best_tag]}")

with open("/marimo/v5_phase6b_results.json", "w") as fh:
    _jsonH.dump(_p6b, fh, indent=2)
print(f"saved -> /marimo/v5_phase6b_results.json | elapsed {_timeH.time() - _t0h:.0f}s")
