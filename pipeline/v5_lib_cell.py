# v5 shared library: data, splits, metrics, MLP probe trainer
# Public: v5_feats, v5_y, v5_paths, v5_classes, v5_splits, v5_gmean, v5_train_mlp_probs, v5_HP
import numpy as _np5
import os as _os5
import torch as _torch5
import torch.nn as _nn5
import torch.nn.functional as _F5
import copy as _copy5
from torch.utils.data import TensorDataset as _TDS5, DataLoader as _DL5, WeightedRandomSampler as _WRS5
from sklearn.model_selection import StratifiedKFold as _SKF5
from sklearn.preprocessing import StandardScaler as _SS5
from sklearn.metrics import recall_score as _rec5

# ---- Data: DINOv2-only (locked v4 baseline features) ----
_v5_dc = _np5.load("/marimo/dinov2_vitb14_features.npz")
_v5_cache_feats = _v5_dc["features"]
_v5_cache_paths = _v5_dc["paths"].tolist()

# CLASSES_4 / DATA_ROOT / SOURCE_FOLDERS / device / optim come from the notebook's
# config + imports cells (reused, not redefined, to keep the graph clean).

def _v5_gather(data_root, folders, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for folder in folders:
        fp = data_root + "/" + folder
        for cls in classes:
            cp = fp + "/" + cls
            if not _os5.path.isdir(cp):
                continue
            for f in sorted(_os5.listdir(cp)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    samples.append((cp + "/" + f, class_to_idx[cls]))
    return samples

_v5_all = _v5_gather(DATA_ROOT, SOURCE_FOLDERS, CLASSES_4)
v5_paths = [p for p, _ in _v5_all]
v5_y = _np5.array([l for _, l in _v5_all])
_v5_p2i = {p: i for i, p in enumerate(_v5_cache_paths)}
v5_feats = _np5.array([_v5_cache_feats[_v5_p2i[p]] for p in v5_paths], dtype=_np5.float32)
v5_classes = list(CLASSES_4)

print(f"v5_lib ready: feats={v5_feats.shape} y={v5_y.shape}")
for _i, _c in enumerate(v5_classes):
    print(f"  {_c}: {(v5_y == _i).sum()}")

# ---- Hyperparams (locked v4 config) ----
v5_HP = {
    "epochs": 200,
    "lr": 1e-3,
    "wd": 1e-3,
    "patience": 30,
    "logit_tau": 1.0,
    "dropout": 0.4,
    "batch": 128,
    "n_folds": 5,
    "split_seed": 42,
}

# ---- Splits (identical across ALL A/B configs) ----
def v5_splits():
    skf = _SKF5(n_splits=v5_HP["n_folds"], shuffle=True, random_state=v5_HP["split_seed"])
    return list(skf.split(v5_feats, v5_y))

# ---- Metrics ----
def v5_gmean(y_true, y_pred):
    rec = _rec5(y_true, y_pred, average=None, zero_division=0)
    return float(_np5.exp(_np5.mean(_np5.log(rec + 1e-8))))

# ---- Probe model ----
class _V5Probe(_nn5.Module):
    def __init__(self, in_dim, n_classes, dropout=0.4):
        super().__init__()
        self.net = _nn5.Sequential(
            _nn5.LayerNorm(in_dim),
            _nn5.Linear(in_dim, 256),
            _nn5.GELU(),
            _nn5.Dropout(dropout),
            _nn5.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.net(x)

_v5_log_prior = _np5.log(_np5.bincount(v5_y, minlength=4) / len(v5_y) + 1e-8)

def v5_train_mlp(X, y, train_idx, val_idx, seed=0, mixup=False, hp=None):
    """Train the locked v4 MLP probe on one fold; return (model, val_probs)."""
    hp = hp or v5_HP
    _torch5.manual_seed(seed * 1000 + 7)
    dev = device  # from imports_cell

    scaler = _SS5()
    Xtr = scaler.fit_transform(X[train_idx])
    Xva = scaler.transform(X[val_idx])
    ytr = _torch5.tensor(y[train_idx], dtype=_torch5.long)

    cc = _np5.maximum(_np5.bincount(y[train_idx], minlength=4), 1)
    sw = (1.0 / _np5.sqrt(cc))[y[train_idx]]
    gen = _torch5.Generator(); gen.manual_seed(seed * 1000 + 7)
    sampler = _WRS5(weights=_torch5.tensor(sw, dtype=_torch5.double), num_samples=len(train_idx), replacement=True, generator=gen)

    train_dl = _DL5(_TDS5(_torch5.tensor(Xtr, dtype=_torch5.float32), ytr), batch_size=hp["batch"], sampler=sampler)
    val_dl = _DL5(_TDS5(_torch5.tensor(Xva, dtype=_torch5.float32),
                        _torch5.tensor(y[val_idx], dtype=_torch5.long)), batch_size=256, shuffle=False)

    model = _V5Probe(Xtr.shape[1], 4, hp["dropout"]).to(dev)
    opt = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    cw = _torch5.tensor(cc.sum() / (4.0 * cc), dtype=_torch5.float32).to(dev)
    crit = _nn5.CrossEntropyLoss(weight=cw)
    logit_bias = _torch5.tensor(hp["logit_tau"] * _v5_log_prior, dtype=_torch5.float32).to(dev)

    _MINORITY = _torch5.tensor([2, 3], dtype=_torch5.long)  # reverse, straight

    bf1, patience, best_state = -1.0, 0, None
    for ep in range(hp["epochs"]):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            if mixup and _np5.random.rand() < 0.5:
                min_mask = _torch5.isin(yb, _MINORITY.to(dev))
                if min_mask.sum() > 0:
                    lam = float(_np5.clip(_np5.random.beta(0.4, 0.4), 0.2, 0.8))
                    partner_idx = _torch5.randint(0, xb.size(0), (xb.size(0),), device=dev)
                    x_mix = xb.clone()
                    x_mix[min_mask] = lam * xb[min_mask] + (1 - lam) * xb[partner_idx][min_mask]
                    y_partner = yb[partner_idx]
                    opt.zero_grad(set_to_none=True)
                    logits = model(x_mix) + logit_bias
                    ce_main = crit(logits, yb)
                    # partner loss only on the actually-mixed (minority) rows
                    ce_part = _F5.cross_entropy(logits[min_mask], y_partner[min_mask], weight=cw, label_smoothing=0.0)
                    loss = ce_main + (1 - lam) * ce_part
                    loss.backward()
                    _torch5.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    continue
            opt.zero_grad(set_to_none=True)
            logits = model(xb) + logit_bias
            loss = crit(logits, yb)
            loss.backward()
            _torch5.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        preds, labs = [], []
        with _torch5.no_grad():
            for xb, yb in val_dl:
                logits = model(xb.to(dev)) + logit_bias
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(yb.numpy())
        from sklearn.metrics import f1_score as _f1s
        vf1 = _f1s(labs, preds, average="macro", zero_division=0)
        if vf1 > bf1:
            bf1 = vf1
            best_state = _copy5.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= hp["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    # stash inference artifacts on the model for v5_predict_probs
    model.v5_logit_bias = logit_bias
    model.v5_scaler = scaler
    with _torch5.no_grad():
        xb = _torch5.tensor(Xva, dtype=_torch5.float32).to(dev)
        probs = _F5.softmax(model(xb) + logit_bias, dim=1).cpu().numpy()
    return model, probs


def v5_predict_probs(model, X):
    """Score arbitrary rows with a probe returned by v5_train_mlp."""
    Xs = model.v5_scaler.transform(X)
    with _torch5.no_grad():
        xb = _torch5.tensor(Xs, dtype=_torch5.float32).to(device)
        return _F5.softmax(model(xb) + model.v5_logit_bias, dim=1).cpu().numpy()

print("v5 lib ready: v5_train_mlp + v5_predict_probs + v5_splits + v5_gmean")