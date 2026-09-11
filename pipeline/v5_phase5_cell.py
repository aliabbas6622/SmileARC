# v5 Phase 5: FINAL — winning config concat(DINOv2+SigLIP) -> full OOF report + bias tuning + artifacts
# Re-runnable after label adjudication: fill reviewer_verdict in /marimo/v5_label_review.csv, then re-run this cell.
import numpy as _npE
import json as _jsonE
import csv as _csvE
import os as _osE
import hashlib as _hlE
import time as _timeE
import torch as _tE
import torch.nn as _nnE
import torch.nn.functional as _FE
import copy as _copyE
from sklearn.metrics import f1_score as _f1E, classification_report as _crE, confusion_matrix as _cmE
from sklearn.preprocessing import StandardScaler as _SSE
from torch.utils.data import TensorDataset as _TDSE, DataLoader as _DLE, WeightedRandomSampler as _WRSE

_t0e = _timeE.time()
_cls_e = v5_classes
_na_e = _cls_e.index("not available")
_rev_e = [i for i, c in enumerate(_cls_e) if "reverse" in c.lower()][0]
_XE = _npE.concatenate([v5_feats, v5_siglip_feats], axis=1).astype(_npE.float32)

# ---- optional label fixes from the Phase 2 review CSV ----
def _load_label_fixes():
    fixes = {}
    p = "/marimo/v5_label_review.csv"
    if not _osE.path.exists(p):
        return fixes
    with open(p) as fh:
        for row in _csvE.DictReader(fh):
            v = (row.get("reviewer_verdict") or "").strip().lower()
            if v.startswith("fix_to:"):
                cls = v.split("fix_to:", 1)[1].strip()
                if cls in _cls_e:
                    fixes[row["path"]] = _cls_e.index(cls)
    return fixes

_label_fixes = _load_label_fixes()
_ye = v5_y.copy()
for _p, _c in _label_fixes.items():
    if _p in v5_paths:
        _ye[v5_paths.index(_p)] = _c
print(f"label fixes applied: {len(_label_fixes)}")


def _final_oof(X, y, splits, seed=0):
    """Locked v4 recipe on given features/labels -> OOF probs + per-fold F1."""
    oof = _npE.zeros((len(y), 4), dtype=_npE.float32)
    folds = []
    for _f, (_tr, _va) in enumerate(splits):
        _tE.manual_seed(seed * 1000 + 31)
        sc = _SSE()
        Xtr = sc.fit_transform(X[_tr])
        ytr = _tE.tensor(y[_tr], dtype=_tE.long)
        cc = _npE.maximum(_npE.bincount(y[_tr], minlength=4), 1)
        sw = (1.0 / _npE.sqrt(cc))[y[_tr]]
        gen = _tE.Generator(); gen.manual_seed(seed * 1000 + 31)
        sampler = _WRSE(weights=_tE.tensor(sw, dtype=_tE.double), num_samples=len(_tr), replacement=True, generator=gen)
        dl = _DLE(_TDSE(_tE.tensor(Xtr, dtype=_tE.float32), ytr), batch_size=128, sampler=sampler)
        net = _nnE.Sequential(
            _nnE.LayerNorm(X.shape[1]), _nnE.Linear(X.shape[1], 256), _nnE.GELU(),
            _nnE.Dropout(0.4), _nnE.Linear(256, 4),
        ).to(device)
        opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        cw = _tE.tensor(cc.sum() / (4.0 * cc), dtype=_tE.float32).to(device)
        crit = _nnE.CrossEntropyLoss(weight=cw)
        lp = _npE.log(cc / cc.sum() + 1e-8)
        bias = _tE.tensor(lp, dtype=_tE.float32).to(device)
        Xva = _tE.tensor(sc.transform(X[_va]), dtype=_tE.float32).to(device)
        best, pat, best_state = -1.0, 0, None
        for _ep in range(200):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(net(xb) + bias, yb)
                loss.backward()
                _tE.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            sch.step()
            net.eval()
            with _tE.no_grad():
                pr = _FE.softmax(net(Xva) + bias, 1).cpu().numpy()
            f1 = _f1E(y[_va], pr.argmax(1), average="macro", zero_division=0)
            if f1 > best:
                best, pat, best_state = f1, 0, _copyE.deepcopy(net.state_dict())
            else:
                pat += 1
                if pat >= 30:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with _tE.no_grad():
            oof[_va] = _FE.softmax(net(Xva) + bias, 1).cpu().numpy()
        folds.append({"fold": _f + 1, "f1": float(best)})
        del net, opt, sch, dl
        torch.cuda.empty_cache()
    return oof, folds


def _report(y, probs, tag):
    pred = probs.argmax(1)
    mf1 = _f1E(y, pred, average="macro", zero_division=0)
    rf1 = _f1E((y == _rev_e).astype(int), (pred == _rev_e).astype(int), zero_division=0)
    gm = v5_gmean(y, pred)
    print(f"\n===== {tag} =====")
    print(f"OOF macro-F1={mf1:.4f}  reverse-F1={rf1:.4f}  G-mean={gm:.4f}")
    print(_crE(y, pred, target_names=_cls_e, zero_division=0))
    print("confusion (rows=true):")
    print(_cmE(y, pred))
    # post-hoc log-prior bias tuning
    prior = _npE.bincount(y, minlength=4) / len(y)
    lp = _npE.log(prior + 1e-8)
    best_tau, best_f1 = 0.0, mf1
    for tau in _npE.arange(0.1, 3.01, 0.1):
        p2 = probs.copy()
        p2 = p2 - p2.max(1, keepdims=True)
        adj = _npE.exp(p2 + tau * lp)
        adj = adj / adj.sum(1, keepdims=True)
        f1v = _f1E(y, adj.argmax(1), average="macro", zero_division=0)
        if f1v > best_f1:
            best_f1, best_tau = f1v, float(tau)
    print(f"tuned tau={best_tau:.1f} -> macro-F1={best_f1:.4f}")
    return {"oof_f1": float(mf1), "reverse_f1": float(rf1), "gmean": float(gm),
            "tuned_tau": best_tau, "tuned_f1": float(best_f1), "pred": pred}


_splits_e = v5_splits()
_oof_e, _folds_e = _final_oof(_XE, _ye, _splits_e)
_f1l = [f["f1"] for f in _folds_e]
print(f"per-fold macro-F1: {[round(x, 4) for x in _f1l]}  mean={_npE.mean(_f1l):.4f} +/- {_npE.std(_f1l):.4f}")
_rep = _report(_ye, _oof_e, f"FINAL concat(DINOv2+SigLIP) n_fixes={len(_label_fixes)}")

# ---- artifact packaging ----
_out = "/marimo/v5_final"
_osE.makedirs(_out, exist_ok=True)
_npE.save(f"{_out}/oof_probs.npy", _oof_e)

def _sha(p):
    h = _hlE.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

_manifest = {}
for _c in ["/marimo/dinov2_vitb14_features.npz", "/marimo/siglip_vitb16_features.npz", "/marimo/mediapipe_features.npz"]:
    if _osE.path.exists(_c):
        _manifest[_osE.path.basename(_c)] = {"sha256_16": _sha(_c), "bytes": _osE.path.getsize(_c)}

_config = {
    "version": "v5-final",
    "features": {"backbones": ["dinov2_vitb14(cls+patchmean,768d)", "siglip_vitb16_webli(pooled,768d)"],
                 "mode": "concat 1536-d", "caches": _manifest},
    "probe": {"arch": "LayerNorm->Linear(256)->GELU->Dropout(0.4)->Linear(4)", "params_approx": int(1536 * 256 + 256 * 4 + 256)},
    "training": {"epochs": 200, "lr": 1e-3, "wd": 1e-3, "batch": 128, "sampler": "sqrt-inverse weighted",
                 "loss": "CE(class-balanced) + logit-adjust(tau=1, train prior)", "sched": "cosine", "patience": 30, "seed": 0},
    "cv": {"scheme": "StratifiedKFold k=5 shuffle seed=42", "fold_f1": _f1l,
           "mean_f1": float(_npE.mean(_f1l)), "std_f1": float(_npE.std(_f1l))},
    "results": {k: v for k, v in _rep.items() if k != "pred"},
    "ab_decisions": {
        "1a seed ensemble": "reject (-0.013)", "1b probe family": "reject (all < baseline)",
        "1c feature mixup": "reject (-0.046)", "phase2 label review": "artifact ready (pending human adjudication)",
        "phase3 two-stage gate": "reject (probe gate +0.006 < +0.01)",
        "phase4 siglip": "ADOPT concat (+0.012)", "phase4 prob-avg": "reject",
    },
    "label_fixes_applied": len(_label_fixes),
}
with open(f"{_out}/config.json", "w") as _fh:
    _jsonE.dump(_config, _fh, indent=2)

print(f"\n{'='*74}\nv5 FINAL SUMMARY\n{'='*74}")
print(f"config: concat(DINOv2+SigLIP) probe | OOF macro-F1={_rep['oof_f1']:.4f} (tuned {_rep['tuned_f1']:.4f})")
print(f"vs v4 DINOv2-only OOF 0.6501 | vs old hybrid test 0.592 | reverse-F1 {_rep['reverse_f1']:.4f}")
print(f"artifacts: {_out}/{{config.json,oof_probs.npy}}")
import marimo as _moE
_moE.status.toast(f"v5 FINAL: macro-F1 {_rep['oof_f1']:.4f} (tuned {_rep['tuned_f1']:.4f})", kind="success")
v5_final_results = {k: v for k, v in _config.items() if k != "features"}
v5_final_results
