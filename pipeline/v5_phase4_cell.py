# v5 Phase 4: SigLIP ViT-B/16 features -> A/B concat-probe vs prob-average vs DINOv2-only baseline
import numpy as _npD
import json as _jsonD
import os as _osD
import time as _timeD
import torch as _tD
import torch.nn as _nnD
import torch.nn.functional as _FD
import copy as _copyD
from PIL import Image as _ImgD
import torchvision.transforms.v2 as _T2D
from sklearn.metrics import f1_score as _f1D
from sklearn.preprocessing import StandardScaler as _SSD
from torch.utils.data import TensorDataset as _TDSD, DataLoader as _DLD, WeightedRandomSampler as _WRSD

_t0d = _timeD.time()
_Xd, _yd = v5_feats, v5_y          # DINOv2 (768-d, locked baseline)
_splits_d = v5_splits()
_rev_d = [i for i, c in enumerate(v5_classes) if "reverse" in c.lower()][0]
_cls_d = v5_classes

# ---------- 1) extract + cache SigLIP ViT-B/16 ----------
_sig_cache = "/marimo/siglip_vitb16_features.npz"
_SIGLIP_TAG = "ViT-B-16-SigLIP_webli_v1"
if _osD.path.exists(_sig_cache):
    _sd = _npD.load(_sig_cache, allow_pickle=True)
    _sig_feats = _sd["features"]
    print(f"SigLIP cache loaded: {_sig_feats.shape} (tag={str(_sd['tag'])})")
else:
    import open_clip as _ocD
    print("Downloading/loading SigLIP ViT-B-16-SigLIP (webli)...")
    _sm, _, _ = _ocD.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    _sm.eval().to(device)
    _sig_tf = _T2D.Compose([
        _T2D.Resize((224, 224), antialias=True),
        _T2D.ToImage(), _T2D.ToDtype(torch.float32, scale=True),
        _T2D.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    _feats = []
    with torch.no_grad():
        for _s in range(0, len(v5_paths), 64):
            _batch = torch.stack([
                _sig_tf(_ImgD.open(_p).convert("RGB")) for _p in v5_paths[_s:_s + 64]
            ]).to(device)
            _emb = _sm.encode_image(_batch).float().cpu().numpy()  # 768-d pooled
            _feats.append(_emb)
            if (_s // 64) % 5 == 0:
                print(f"  {_s + 64}/{len(v5_paths)} [{_timeD.time() - _t0d:.0f}s]")
    _sig_feats = _npD.concatenate(_feats, 0).astype(_npD.float32)
    _npD.savez_compressed(_sig_cache, features=_sig_feats, paths=_npD.array(v5_paths), tag=_npD.array(_SIGLIP_TAG))
    del _sm
    torch.cuda.empty_cache()
    print(f"SigLIP extracted: {_sig_feats.shape} in {_timeD.time() - _t0d:.0f}s -> cached")

v5_siglip_feats = _sig_feats  # public, aligned with v5_feats/v5_paths


# ---------- 2) generic MLP probe on any feature matrix (locked recipe) ----------
def _mlp_oof(X, splits, seed=0):
    oof = _npD.zeros((len(_yd), 4), dtype=_npD.float32)
    for _tr, _va in splits:
        _tD.manual_seed(seed * 1000 + 23)
        sc = _SSD()
        Xtr = sc.fit_transform(X[_tr])
        ytr = _tD.tensor(_yd[_tr], dtype=_tD.long)
        cc = _npD.maximum(_npD.bincount(_yd[_tr], minlength=4), 1)
        sw = (1.0 / _npD.sqrt(cc))[_yd[_tr]]
        gen = _tD.Generator(); gen.manual_seed(seed * 1000 + 23)
        sampler = _WRSD(weights=_tD.tensor(sw, dtype=_tD.double), num_samples=len(_tr), replacement=True, generator=gen)
        dl = _DLD(_TDSD(_tD.tensor(Xtr, dtype=_tD.float32), ytr), batch_size=128, sampler=sampler)
        net = _nnD.Sequential(
            _nnD.LayerNorm(X.shape[1]), _nnD.Linear(X.shape[1], 256), _nnD.GELU(),
            _nnD.Dropout(0.4), _nnD.Linear(256, 4),
        ).to(device)
        opt = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        cw = _tD.tensor(cc.sum() / (4.0 * cc), dtype=_tD.float32).to(device)
        crit = _nnD.CrossEntropyLoss(weight=cw)
        lp = _npD.log(cc / cc.sum() + 1e-8)
        bias = _tD.tensor(lp, dtype=_tD.float32).to(device)
        Xva = _tD.tensor(sc.transform(X[_va]), dtype=_tD.float32).to(device)
        best, pat, best_state = -1.0, 0, None
        for _ep in range(200):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(net(xb) + bias, yb)
                loss.backward()
                _tD.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            sch.step()
            net.eval()
            with _tD.no_grad():
                pr = _FD.softmax(net(Xva) + bias, 1).cpu().numpy()
            f1 = _f1D(_yd[_va], pr.argmax(1), average="macro", zero_division=0)
            if f1 > best:
                best, pat, best_state = f1, 0, _copyD.deepcopy(net.state_dict())
            else:
                pat += 1
                if pat >= 30:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with _tD.no_grad():
            oof[_va] = _FD.softmax(net(Xva) + bias, 1).cpu().numpy()
        del net, opt, sch, dl
        torch.cuda.empty_cache()
    return oof


def _mets(probs):
    pred = probs.argmax(1)
    return (
        _f1D(_yd, pred, average="macro", zero_division=0),
        _f1D((_yd == _rev_d).astype(int), (pred == _rev_d).astype(int), zero_division=0),
    )


# ---------- 3) A/B ----------
print("\n=== (a) DINOv2-only (locked baseline) ===")
_oof_dino = _mlp_oof(_Xd, _splits_d)
_m, _r = _mets(_oof_dino)
print(f"(a) DINOv2-only:        macro-F1={_m:.4f} reverse={_r:.4f}")

print("\n=== (b) SigLIP-only probe ===")
_oof_sig = _mlp_oof(v5_siglip_feats, _splits_d)
_ms, _rs = _mets(_oof_sig)
print(f"(b) SigLIP-only:        macro-F1={_ms:.4f} reverse={_rs:.4f}")

print("\n=== (c) concat(DINOv2, SigLIP) -> one probe ===")
_Xc4 = _npD.concatenate([_Xd, v5_siglip_feats], axis=1)
_oof_cat = _mlp_oof(_Xc4, _splits_d)
_mc, _rc = _mets(_oof_cat)
print(f"(c) concat 1536-d:      macro-F1={_mc:.4f} reverse={_rc:.4f}")

print("\n=== (d) prob-average of (a)+(b) ===")
_oof_avg = 0.5 * _oof_dino + 0.5 * _oof_sig
_ma, _ra = _mets(_oof_avg)
print(f"(d) prob-avg 50/50:     macro-F1={_ma:.4f} reverse={_ra:.4f}")
_bestw, _bw, _brw = 0.5, _ma, _ra
for _w in _npD.arange(0.2, 0.81, 0.1):
    _o = _w * _oof_dino + (1 - _w) * _oof_sig
    _fm, _fr = _mets(_o)
    if _fm > _bw:
        _bestw, _bw, _brw = float(_w), _fm, _fr
print(f"(d) prob-avg w={_bestw:.1f} dino: macro-F1={_bw:.4f} reverse={_brw:.4f}")

# ---------- adopt/reject ----------
print(f"\n{'='*74}\nPHASE 4 ADOPT/REJECT vs baseline (macro-F1={_m:.4f}, reverse={_r:.4f})\n{'='*74}")
_rows4 = [("concat_1536d", _mc, _rc), ("prob_avg_50/50", _ma, _ra), (f"prob_avg_w{_bestw:.1f}", _bw, _brw)]
_adopt4 = []
for _n, _fm, _fr in _rows4:
    _d, _dr = _fm - _m, _fr - _r
    _ok = (_d >= 0.01) and (_dr >= -0.05)
    if _ok:
        _adopt4.append(_n)
    print(f"  {_n:20s} F1={_fm:.4f} ({_d:+.4f}) reverse={_fr:.4f} ({_dr:+.4f}) -> {'ADOPT' if _ok else 'reject'}")

v5_phase4_results = {
    "baseline_dino": {"oof_f1": float(_m), "reverse_f1": float(_r)},
    "siglip_only": {"oof_f1": float(_ms), "reverse_f1": float(_rs)},
    "concat": {"oof_f1": float(_mc), "reverse_f1": float(_rc)},
    "prob_avg": {"oof_f1": float(_ma), "reverse_f1": float(_ra)},
    "prob_avg_tuned": {"w_dino": _bestw, "oof_f1": float(_bw), "reverse_f1": float(_brw)},
    "adopted": _adopt4,
    "elapsed_s": round(_timeD.time() - _t0d, 1),
}
with open("/marimo/v5_phase4_results.json", "w") as _fh:
    _jsonD.dump(v5_phase4_results, _fh, indent=2)
print(f"\nelapsed {v5_phase4_results['elapsed_s']}s | adopted: {_adopt4 or 'none'}")
v5_phase4_results
