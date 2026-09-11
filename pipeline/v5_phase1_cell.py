# v5 Phase 1: A/B on identical folds — baseline / 1a seed-ensemble / 1b probe family / 1c mixup
# Adopt rule: dOOF macro-F1 >= +0.01 AND reverse F1 drop <= 0.05 vs locked v4 baseline.
import numpy as _npA
import json as _jsonA
import time as _timeA
from sklearn.linear_model import LogisticRegression as _LRA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDAA
from sklearn.neighbors import KNeighborsClassifier as _KNNA
from sklearn.neighbors import NearestCentroid as _NCA

_t0a = _timeA.time()
_splits_a = v5_splits()
_Xa, _ya = v5_feats, v5_y
_rev_idx_a = [i for i, c in enumerate(v5_classes) if "reverse" in c.lower()][0]
_NCA4 = len(v5_classes)


def _oof_metrics(probs):
    pred = _npA.argmax(probs, axis=1)
    from sklearn.metrics import f1_score as _f1
    mf1 = _f1(_ya, pred, average="macro", zero_division=0)
    rf1 = _f1((_ya == _rev_idx_a).astype(int), (pred == _rev_idx_a).astype(int), zero_division=0)
    gm = v5_gmean(_ya, pred)
    return mf1, rf1, gm


def _sklearn_oof(make_model):
    """Run a sklearn probe family member over the locked folds; return OOF probs."""
    from sklearn.preprocessing import StandardScaler as _SS
    oof = _npA.zeros((len(_ya), _NCA4), dtype=_npA.float32)
    for tr, va in _splits_a:
        sc = _SS().fit(_Xa[tr])
        m = make_model()
        m.fit(sc.transform(_Xa[tr]), _ya[tr])
        oof[va] = m.predict_proba(sc.transform(_Xa[va]))
    return oof


print("=== baseline: locked v4 MLP (seed 0) ===")
_oof_base = _npA.zeros((len(_ya), _NCA4), dtype=_npA.float32)
for _f, (_tr, _va) in enumerate(_splits_a):
    _, _p = v5_train_mlp(_Xa, _ya, _tr, _va, seed=0)
    _oof_base[_va] = _p
_m, _r, _g = _oof_metrics(_oof_base)
print(f"baseline OOF macro-F1={_m:.4f} reverse-F1={_r:.4f} G-mean={_g:.4f}")

print("\n=== 1a: 5-fold prob averaging over seeds 0-3 ===")
_seed_oofs = [_oof_base]
for _s in (1, 2, 3):
    _o = _npA.zeros((len(_ya), _NCA4), dtype=_npA.float32)
    for _tr, _va in _splits_a:
        _, _p = v5_train_mlp(_Xa, _ya, _tr, _va, seed=_s)
        _o[_va] = _p
    _seed_oofs.append(_o)
    _sm, _sr, _sg = _oof_metrics(_o)
    print(f"  seed {_s}: macro-F1={_sm:.4f} reverse-F1={_sr:.4f}")
_ens = _npA.mean(_seed_oofs, axis=0)
_em, _er, _eg = _oof_metrics(_ens)
print(f"1a ensemble OOF macro-F1={_em:.4f} reverse-F1={_er:.4f} G-mean={_eg:.4f}")

print("\n=== 1b: probe family on same features ===")
_fam = {
    "logreg_bal": lambda: _LRA(max_iter=2000, C=1.0, class_weight="balanced"),
    "logreg_C0.1": lambda: _LRA(max_iter=2000, C=0.1, class_weight="balanced"),
    "lda_shrink": lambda: _LDAA(solver="lsqr", shrinkage="auto"),
    "knn_k20": lambda: _KNNA(n_neighbors=20, weights="distance"),
    "knn_k50": lambda: _KNNA(n_neighbors=50, weights="distance"),
    "prototype": None,  # handled below (no predict_proba)
}
_fam_oof = {}
for _name, _mk in _fam.items():
    if _name == "prototype":
        oof = _npA.zeros((len(_ya), _NCA4), dtype=_npA.float32)
        from sklearn.preprocessing import StandardScaler as _SS2
        for tr, va in _splits_a:
            sc = _SS2().fit(_Xa[tr])
            Xtr, Xva = sc.transform(_Xa[tr]), sc.transform(_Xa[va])
            cents = _npA.stack([Xtr[_ya[tr] == c].mean(0) for c in range(_NCA4)])
            d = ((Xva[:, None, :] - cents[None]) ** 2).sum(-1)
            _z = -d
            _z = _z - _z.max(axis=1, keepdims=True)  # stable softmax
            _e = _npA.exp(_z)
            oof[va] = _e / _e.sum(axis=1, keepdims=True)
    else:
        oof = _sklearn_oof(_mk)
    fm, fr, fg = _oof_metrics(oof)
    _fam_oof[_name] = oof
    print(f"  {_name:12s}: macro-F1={fm:.4f} reverse-F1={fr:.4f} G-mean={fg:.4f}")

print("\n=== 1c: MLP + feature-space mixup (minority-focused) ===")
_oof_mix = _npA.zeros((len(_ya), _NCA4), dtype=_npA.float32)
for _tr, _va in _splits_a:
    _, _p = v5_train_mlp(_Xa, _ya, _tr, _va, seed=0, mixup=True)
    _oof_mix[_va] = _p
_mm, _mr, _mg = _oof_metrics(_oof_mix)
print(f"1c mixup OOF macro-F1={_mm:.4f} reverse-F1={_mr:.4f} G-mean={_mg:.4f}")

# ---- Adopt/reject table vs baseline ----
print(f"\n{'='*74}\nPHASE 1 ADOPT/REJECT (rule: dOOF-F1 >= +0.01 AND reverse drop <= 0.05)\n{'='*74}")
print(f"baseline          macro-F1={_m:.4f} reverse={_r:.4f}")
_rows = [("1a_seed_ensemble", _em, _er), ("1c_mixup", _mm, _mr)] + [
    (f"1b_{k}", *_oof_metrics(v)[:2]) for k, v in _fam_oof.items()
]
_adopted = []
for _n, _fm1, _fr1 in _rows:
    _d = _fm1 - _m
    _dr = _fr1 - _r
    _ok = (_d >= 0.01) and (_dr >= -0.05)
    if _ok:
        _adopted.append(_n)
    print(f"  {_n:20s} F1={_fm1:.4f} ({_d:+.4f}) reverse={_fr1:.4f} ({_dr:+.4f}) -> {'ADOPT' if _ok else 'reject'}")

_res = {
    "baseline": {"oof_f1": float(_m), "reverse_f1": float(_r), "gmean": float(_g)},
    "variants": {n: {"oof_f1": float(f), "reverse_f1": float(r)} for n, f, r in _rows},
    "adopted": _adopted,
    "elapsed_s": round(_timeA.time() - _t0a, 1),
}
with open("/marimo/v5_phase1_results.json", "w") as _fh:
    _jsonA.dump(_res, _fh, indent=2)

v5_phase1_results = _res
print(f"\nelapsed {_res['elapsed_s']}s | adopted: {_adopted or 'none'}")
v5_phase1_results
