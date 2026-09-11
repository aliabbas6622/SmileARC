# v5 Phase 6: reverse-class offline augmentation (brightness/contrast/shift/rotate/stretch/flip)
# Fold-safe: augmented copies are added ONLY to the train side of each fold; val folds untouched.
import numpy as _npF
import json as _jsonF
import os as _osF
import time as _timeF
import torch as _tF
import torch.nn as _nnF
import torch.nn.functional as _FF
import copy as _copyF
from PIL import Image as _ImgF, ImageEnhance as _EnhF
import torchvision.transforms.v2 as _T2F
from sklearn.metrics import f1_score as _f1F
from sklearn.preprocessing import StandardScaler as _SSF
from torch.utils.data import TensorDataset as _TDSF, DataLoader as _DLF, WeightedRandomSampler as _WRSF

_t0f = _timeF.time()
_Xf, _yf = v5_feats, v5_y
_splits_f = v5_splits()
_rev_idx_f = [i for i, c in enumerate(v5_classes) if "reverse" in c.lower()][0]
_rev_rows_f = _npF.where(_yf == _rev_idx_f)[0]
print(f"reverse originals: {len(_rev_rows_f)} rows")

# ---------- 1) deterministic offline augmentation of reverse images ----------
_AUG_CACHE = "/marimo/reverse_aug_dino_siglip.npz"
_KMAX = 8  # max augmented variants per original (K is swept below)

def _augment_pil(img, k):
    """Deterministic per-k variant: brightness/contrast/color jitter, rotate, shift, stretch/shear, flip."""
    rng = _npF.random.RandomState(1000 + k)
    out = img
    out = _EnhF.Brightness(out).enhance(1.0 + rng.uniform(-0.25, 0.25))
    out = _EnhF.Contrast(out).enhance(1.0 + rng.uniform(-0.25, 0.25))
    out = _EnhF.Color(out).enhance(1.0 + rng.uniform(-0.15, 0.15))
    angle = rng.uniform(-12, 12)
    tx = rng.uniform(-0.06, 0.06) * out.size[0]
    ty = rng.uniform(-0.06, 0.06) * out.size[1]
    sx = 1.0 + rng.uniform(-0.08, 0.08)   # stretch x
    sy = 1.0 + rng.uniform(-0.08, 0.08)   # stretch y
    shear = rng.uniform(-8, 8)
    sh = _npF.tan(_npF.radians(shear))
    ra = _npF.tan(_npF.radians(angle))
    # AFFINE matrix maps output(x,y) -> input(a*x + b*y + c, d*x + e*y + f); combining
    # stretch (sx, sy), shear, rotation and shift in one resample.
    out = out.transform(
        out.size, _ImgF.AFFINE,
        (sx, sh, -tx,
         ra, sy, ty),
        resample=_ImgF.BILINEAR, fillcolor=(0, 0, 0),
    )
    if k % 2 == 1:
        out = out.transpose(_ImgF.FLIP_LEFT_RIGHT)
    return out

_dino_tf_f = _T2F.Compose([
    _T2F.Resize((224, 224)),
    _T2F.ToImage(), _T2F.ToDtype(torch.float32, scale=True),
    _T2F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_sig_tf_f = _T2F.Compose([
    _T2F.Resize((224, 224)),
    _T2F.ToImage(), _T2F.ToDtype(torch.float32, scale=True),
    _T2F.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

if _osF.path.exists(_AUG_CACHE):
    _ad = _npF.load(_AUG_CACHE, allow_pickle=True)
    _aug_feats_dino = _ad["dino"]
    _aug_feats_sig = _ad["sig"]
    _aug_row_ids = _ad["row_ids"]
    _aug_ks = _ad["ks"]
    print(f"aug cache loaded: dino={_aug_feats_dino.shape}")
else:
    import open_clip as _ocF
    print("Loading DINOv2 + SigLIP for augmented feature extraction...")
    _dinoF = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    _dinoF.eval().to(device)
    _smF, _, _ = _ocF.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    _smF.eval().to(device)

    _aug_imgs = []  # (original_row_id, k, PIL image)
    for rid in _rev_rows_f:
        img = _ImgF.open(v5_paths[int(rid)]).convert("RGB")
        for k in range(_KMAX):
            _aug_imgs.append((int(rid), k, _augment_pil(img, k)))
    print(f"generated {len(_aug_imgs)} augmented images ({_KMAX} per original)")

    _fd, _fs = [], []
    with torch.no_grad():
        for s in range(0, len(_aug_imgs), 64):
            batch_imgs = [a[2] for a in _aug_imgs[s:s + 64]]
            bd = torch.stack([_dino_tf_f(im) for im in batch_imgs]).to(device)
            bs = torch.stack([_sig_tf_f(im) for im in batch_imgs]).to(device)
            od = _dinoF(bd)
            if isinstance(od, dict):
                fd_ = _npF.concatenate([od["x_norm_clstoken"].cpu().numpy(),
                                        od["x_norm_patchtokens"].mean(dim=1).cpu().numpy()], axis=1)
            else:
                fd_ = od.cpu().numpy()
            fs_ = _smF.encode_image(bs).float().cpu().numpy()
            _fd.append(fd_)
            _fs.append(fs_)
            if (s // 64) % 3 == 0:
                print(f"  {s + 64}/{len(_aug_imgs)} [{_timeF.time() - _t0f:.0f}s]", flush=True)

    _aug_feats_dino = _npF.concatenate(_fd, 0).astype(_npF.float32)
    _aug_feats_sig = _npF.concatenate(_fs, 0).astype(_npF.float32)
    _aug_row_ids = _npF.array([a[0] for a in _aug_imgs], dtype=_npF.int64)
    _aug_ks = _npF.array([a[1] for a in _aug_imgs], dtype=_npF.int64)
    _npF.savez_compressed(_AUG_CACHE, dino=_aug_feats_dino, sig=_aug_feats_sig,
                          row_ids=_aug_row_ids, ks=_aug_ks)
    del _dinoF, _smF
    torch.cuda.empty_cache()
    print(f"aug features: dino={_aug_feats_dino.shape} in {_timeF.time() - _t0f:.0f}s -> cached")

# map original row_id -> list of aug feature indices (ordered by k)
_row2aug_f = {}
for _i, (_rid, _k) in enumerate(zip(_aug_row_ids, _aug_ks)):
    _row2aug_f.setdefault(int(_rid), []).append(_i)
for _rid in _row2aug_f:
    _row2aug_f[_rid].sort(key=lambda i: int(_aug_ks[i]))

# concat(dino, siglip) features for aug rows, aligned with the adopted Phase-4 config
_aug_feats_cat = _npF.concatenate([_aug_feats_dino, _aug_feats_sig], axis=1)

# ---------- 2) locked-recipe MLP probe with fold-safe aug injection ----------
def _mlp_aug_oof(X_base, splits, K, seed=0):
    """OOF probs; adds the first K augmented variants of each reverse TRAIN row (val untouched)."""
    oof = _npF.zeros((len(_yf), 4), dtype=_npF.float32)
    for fi, (_tr, _va) in enumerate(splits):
        _tF.manual_seed(seed * 1000 + 31 + fi)
        if K > 0:
            tr_set = set(_tr.tolist())
            aug_sel = []
            for rid, idxs in _row2aug_f.items():
                if rid in tr_set:
                    aug_sel.extend(idxs[:K])
            aug_sel = _npF.array(sorted(aug_sel), dtype=int)
            Xtr_all = _npF.concatenate([X_base[_tr], _aug_feats_cat[aug_sel]], 0)
            ytr_all = _npF.concatenate([_yf[_tr], _npF.full(len(aug_sel), _rev_idx_f, dtype=_yf.dtype)], 0)
        else:
            Xtr_all, ytr_all = X_base[_tr], _yf[_tr]
        sc = _SSF()
        Xtr = sc.fit_transform(Xtr_all)
        Xva = sc.transform(X_base[_va])
        ytr = _tF.tensor(ytr_all, dtype=_tF.long)
        cc = _npF.maximum(_npF.bincount(ytr_all, minlength=4), 1)
        sw = (1.0 / _npF.sqrt(cc))[ytr_all]
        gen = _tF.Generator(); gen.manual_seed(seed * 1000 + 31 + fi)
        sampler = _WRSF(weights=_tF.tensor(sw, dtype=_tF.double), num_samples=len(ytr_all),
                        replacement=True, generator=gen)
        dl = _DLF(_TDSF(_tF.tensor(Xtr, dtype=_tF.float32), ytr), batch_size=128, sampler=sampler)
        vdl = _DLF(_TDSF(_tF.tensor(Xva, dtype=_tF.float32),
                         _tF.tensor(_yf[_va], dtype=_tF.long)), batch_size=256, shuffle=False)
        m = _nnF.Sequential(
            _nnF.LayerNorm(Xtr.shape[1]), _nnF.Linear(Xtr.shape[1], 256), _nnF.GELU(),
            _nnF.Dropout(0.4), _nnF.Linear(256, 4)).to(device)
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        cw = _tF.tensor(cc.sum() / (4.0 * cc), dtype=_tF.float32).to(device)
        crit = _nnF.CrossEntropyLoss(weight=cw)
        lp = _npF.log(_npF.bincount(_yf, minlength=4) / len(_yf) + 1e-8)
        bias = _tF.tensor(1.0 * lp, dtype=_tF.float32).to(device)
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
            vf1 = _f1F(lb, pr, average="macro", zero_division=0)
            if vf1 > bf1:
                bf1, pat, best = vf1, 0, _copyF.deepcopy(m.state_dict())
            else:
                pat += 1
                if pat >= 30:
                    break
        m.load_state_dict(best)
        m.eval()
        with torch.no_grad():
            xb = _tF.tensor(Xva, dtype=_tF.float32).to(device)
            oof[_va] = _FF.softmax(m(xb) + bias, 1).cpu().numpy()
    return oof

def _report_f(tag, oof):
    pred = oof.argmax(1)
    f1 = _f1F(_yf, pred, average="macro", zero_division=0)
    rev = _f1F(_yf == _rev_idx_f, pred == _rev_idx_f, zero_division=0)
    gm = v5_gmean(_yf, pred)
    print(f"{tag:22s} macro-F1={f1:.4f} reverse-F1={rev:.4f} G-mean={gm:.4f}")
    return {"macro_f1": round(float(f1), 4), "reverse_f1": round(float(rev), 4),
            "gmean": round(float(gm), 4)}

print("\n=== A/B: reverse-class offline augmentation (fold-safe) ===")
_phase6_results = {}
_oof_base = _mlp_aug_oof(_Xf, _splits_f, K=0)
_phase6_results["baseline_K0"] = _report_f("(a) baseline K=0", _oof_base)
for _k in [2, 4, 8]:
    _oof_k = _mlp_aug_oof(_Xf, _splits_f, K=_k)
    _phase6_results[f"aug_K{_k}"] = _report_f(f"(b) K={_k} aug", _oof_k)

with open("/marimo/v5_phase6_results.json", "w") as _fh:
    _jsonF.dump(_phase6_results, _fh, indent=2)
print(f"\nsaved -> /marimo/v5_phase6_results.json | elapsed {_timeF.time() - _t0f:.0f}s")
