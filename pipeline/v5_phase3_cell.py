# v5 Phase 3: two-stage classification — usability gate -> 3-class arc head -> map back to 4 labels
# A/B: rule gate (MP-fail => NA) vs binary DINOv2 probe gate. Stage 2: arc head on usable-only.
import numpy as _npC
import json as _jsonC
import time as _timeC
from sklearn.metrics import f1_score as _f1C, classification_report as _crC
from sklearn.preprocessing import StandardScaler as _SSC
import torch as _tC
import torch.nn as _nnC
import torch.nn.functional as _FC
import copy as _copyC
from torch.utils.data import TensorDataset as _TDSC, DataLoader as _DLC, WeightedRandomSampler as _WRSC

_t0c = _timeC.time()
_Xc, _yc, _paths_c = v5_feats, v5_y, v5_paths
_splits_c = v5_splits()
_cls_c = v5_classes
_na = _cls_c.index("not available")
_rev_c = [i for i, c in enumerate(_cls_c) if "reverse" in c.lower()][0]

# arc class order: consonant(0), straight(3), reverse(2) -> map back to 4-cls
_ARC2Q = [0, 3, 2]
_z = (_yc != _na).astype(int)          # 1 = usable (has arc), 0 = NA
_y3 = _npC.zeros(len(_yc), dtype=_npC.int64)
_y3[_yc == 0] = 0                       # consonant -> arc 0
_y3[_yc == 3] = 1                       # straight  -> arc 1
_y3[_yc == 2] = 2                       # reverse  -> arc 2

# ---- Rule-gate data: MP success (cache stores ALL paths; failures are zero-norm rows) ----
_mp = _npC.load("/marimo/mediapipe_features.npz")
_mp_ok = _npC.linalg.norm(_mp["features"], axis=1) > 0
_mpfail = ~_mp_ok
print("=== Gate stats ===")
print(f"MP-fail rate: {_mpfail.mean():.3f} ({_mpfail.sum()}/{len(_yc)})")
print(f"P(NA | MP-fail) = {(_yc[_mpfail] == _na).mean():.3f}   P(MP-fail | NA) = {_mpfail[_yc == _na].mean():.3f}")
print(f"P(usable | MP-ok) = {(_yc[_mp_ok] != _na).mean():.3f}")


# ---- generic probe trainer (binary or 3-class), locked v4 recipe ----
def _train_nc(fit_idx, stop_idx, pred_idx, y_label, n_cls, seed=0):
    _tC.manual_seed(seed * 1000 + 11)
    dev = device
    sc = _SSC()
    Xtr = sc.fit_transform(_Xc[fit_idx])
    ytr = _tC.tensor(y_label[fit_idx], dtype=_tC.long)
    cc = _npC.maximum(_npC.bincount(y_label[fit_idx], minlength=n_cls), 1)
    sw = (1.0 / _npC.sqrt(cc))[y_label[fit_idx]]
    gen = _tC.Generator(); gen.manual_seed(seed * 1000 + 11)
    sampler = _WRSC(weights=_tC.tensor(sw, dtype=_tC.double), num_samples=len(fit_idx), replacement=True, generator=gen)
    dl = _DLC(_TDSC(_tC.tensor(Xtr, dtype=_tC.float32), ytr), batch_size=128, sampler=sampler)
    net = _nnC.Sequential(
        _nnC.LayerNorm(_Xc.shape[1]), _nnC.Linear(_Xc.shape[1], 256), _nnC.GELU(),
        _nnC.Dropout(0.4), _nnC.Linear(256, n_cls),
    ).to(dev)
    opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    cw = _tC.tensor(cc.sum() / (n_cls * cc), dtype=_tC.float32).to(dev)
    crit = _nnC.CrossEntropyLoss(weight=cw)
    lp = _npC.log(cc / cc.sum() + 1e-8)
    bias = _tC.tensor(lp, dtype=_tC.float32).to(dev)

    best, pat, best_state = -1.0, 0, None
    Xstop = _tC.tensor(sc.transform(_Xc[stop_idx]), dtype=_tC.float32).to(dev)
    for _ep in range(200):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(net(xb) + bias, yb)
            loss.backward()
            _tC.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        sch.step()
        net.eval()
        with _tC.no_grad():
            pr = _FC.softmax(net(Xstop) + bias, 1).cpu().numpy()
        f1 = _f1C(y_label[stop_idx], pr.argmax(1), average="macro", zero_division=0)
        if f1 > best:
            best, pat, best_state = f1, 0, _copyC.deepcopy(net.state_dict())
        else:
            pat += 1
            if pat >= 30:
                break
    net.load_state_dict(best_state)
    net.eval()
    with _tC.no_grad():
        Xp = _tC.tensor(sc.transform(_Xc[pred_idx]), dtype=_tC.float32).to(dev)
        pr = _FC.softmax(net(Xp) + bias, 1).cpu().numpy()
    return pr


def _metrics4(pred):
    mf1 = _f1C(_yc, pred, average="macro", zero_division=0)
    rf1 = _f1C((_yc == _rev_c).astype(int), (pred == _rev_c).astype(int), zero_division=0)
    gm = v5_gmean(_yc, pred)
    return mf1, rf1, gm


# ---- OOF: binary gate probe + 3-class arc head ----
print("\n=== training OOF: gate probe + arc head (5 folds) ===")
_gate_prob = _npC.zeros(len(_yc), dtype=_npC.float32)
_arc_prob = _npC.zeros((len(_yc), 3), dtype=_npC.float32)
for _f, (_tr, _va) in enumerate(_splits_c):
    _gp = _train_nc(_tr, _va, _va, _z, 2, seed=0)
    _gate_prob[_va] = _gp[:, 1]
    _tr_u = _tr[_z[_tr] == 1]
    _va_u = _va[_z[_va] == 1]
    _ap = _train_nc(_tr_u, _va_u, _va, _y3, 3, seed=0)
    _arc_prob[_va] = _ap
    print(f"  fold {_f+1}: gate done, arc trained on {len(_tr_u)} usable rows")

_arc_arg4 = _npC.array([_ARC2Q[i] for i in _arc_prob.argmax(1)])


def _final_pred(usable_bool):
    out = _npC.full(len(_yc), _na, dtype=_npC.int64)
    out[usable_bool] = _arc_arg4[usable_bool]
    return out


# ---- (a) rule gate ----
_pred_rule = _final_pred(_mp_ok)
_m, _r, _g = _metrics4(_pred_rule)
print(f"\n(a) RULE gate (MP-ok):  macro-F1={_m:.4f} reverse={_r:.4f} G-mean={_g:.4f}")

# ---- (b) probe gate, t=0.5 and tuned ----
_pred_p05 = _final_pred(_gate_prob >= 0.5)
_m5, _r5, _g5 = _metrics4(_pred_p05)
print(f"(b) PROBE gate t=0.5:   macro-F1={_m5:.4f} reverse={_r5:.4f} G-mean={_g5:.4f}")

_best = (0.5, _m5, _r5, _g5, _pred_p05)
for _t in _npC.arange(0.30, 0.71, 0.05):
    _p = _final_pred(_gate_prob >= _t)
    _fm, _fr, _fg = _metrics4(_p)
    if _fm > _best[1]:
        _best = (float(_t), _fm, _fr, _fg, _p)
_tb, _mb, _rb, _gb, _pb = _best
print(f"(b) PROBE gate t={_tb:.2f}:  macro-F1={_mb:.4f} reverse={_rb:.4f} G-mean={_gb:.4f}   <- best probe gate")

_agree = (_mp_ok == (_gate_prob >= _tb)).mean()
print(f"\ngate agreement (rule vs probe@t={_tb:.2f}): {_agree:.3f}")

# ---- per-class report for the best two-stage variant ----
_best_pred = _pb if _mb >= _m else _pred_rule
print("\nBest two-stage variant per-class report:")
print(_crC(_yc, _best_pred, target_names=_cls_c, zero_division=0))

# ---- adopt/reject vs baseline (locked seed-0 OOF from Phase 1/2) ----
_oof_base_pred = v5_oof_probs.argmax(1)
_bm = _f1C(_yc, _oof_base_pred, average="macro", zero_division=0)
_br = _f1C((_yc == _rev_c).astype(int), (_oof_base_pred == _rev_c).astype(int), zero_division=0)
print(f"{'='*74}\nPHASE 3 ADOPT/REJECT vs baseline (macro-F1={_bm:.4f}, reverse={_br:.4f})\n{'='*74}")
for _n, _fm, _fr in [("rule_gate", _m, _r), ("probe_gate_t0.5", _m5, _r5), (f"probe_gate_t{_tb:.2f}", _mb, _rb)]:
    _d, _dr = _fm - _bm, _fr - _br
    _ok = (_d >= 0.01) and (_dr >= -0.05)
    print(f"  {_n:20s} F1={_fm:.4f} ({_d:+.4f}) reverse={_fr:.4f} ({_dr:+.4f}) -> {'ADOPT' if _ok else 'reject'}")

v5_phase3_results = {
    "baseline": {"oof_f1": float(_bm), "reverse_f1": float(_br)},
    "rule_gate": {"oof_f1": float(_m), "reverse_f1": float(_r), "gmean": float(_g)},
    "probe_gate_t05": {"oof_f1": float(_m5), "reverse_f1": float(_r5), "gmean": float(_g5)},
    "probe_gate_tuned": {"t": float(_tb), "oof_f1": float(_mb), "reverse_f1": float(_rb), "gmean": float(_gb)},
    "gate_agreement": float(_agree),
    "elapsed_s": round(_timeC.time() - _t0c, 1),
}
with open("/marimo/v5_phase3_results.json", "w") as _fh:
    _jsonC.dump(v5_phase3_results, _fh, indent=2)
print(f"\nelapsed {v5_phase3_results['elapsed_s']}s")
v5_phase3_results
