"""
Standalone inference for the Smile Arc Classifier v5 (clean, K=2 aug, 3-seed ensemble).

Usage:
    python predict_smile_arc.py <image1> [image2 ...]
    python predict_smile_arc.py --model /marimo/models/smile_arc_v5_clean_K2aug_3seed.pt <image>

Requires: torch, torchvision, numpy, PIL, open_clip (for SigLIP features).
The DINOv2 model is pulled from torch.hub on first run and cached.

Pipeline (must match training exactly):
  1. DINOv2 ViT-B/14  -> CLS token (768-d)
  2. SigLIP ViT-B-16  -> pooled image embedding (768-d)
  3. concat -> 1536-d, per-seed StandardScaler -> MLP head + logit bias
  4. 3-seed soft-vote -> class probabilities
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.v2 as T2

DEFAULT_MODEL = "/marimo/models/smile_arc_v5_clean_K2aug_3seed.pt"
CLASSES = ["consonant", "not available", "reverse- non consonant", "straight- non consonant"]

_dino_tf = T2.Compose([
    T2.Resize((224, 224)),
    T2.ToImage(), T2.ToDtype(torch.float32, scale=True),
    T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_sig_tf = T2.Compose([
    T2.Resize((224, 224)),
    T2.ToImage(), T2.ToDtype(torch.float32, scale=True),
    T2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_backbones(device):
    print("Loading DINOv2 ViT-B/14 ...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    dino.eval().to(device)
    print("Loading SigLIP ViT-B-16 (webli) ...")
    import open_clip
    sig, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli")
    sig.eval().to(device)
    return dino, sig


@torch.no_grad()
def extract_features(paths, dino, sig, device, batch=32):
    fd, fs = [], []
    for s in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        bd = torch.stack([_dino_tf(im) for im in imgs]).to(device)
        bs = torch.stack([_sig_tf(im) for im in imgs]).to(device)
        od = dino(bd)
        if isinstance(od, dict):
            f = np.concatenate([od["x_norm_clstoken"].cpu().numpy(),
                                od["x_norm_patchtokens"].mean(dim=1).cpu().numpy()], axis=1)
        else:
            f = od.cpu().numpy()
        fd.append(f)
        fs.append(sig.encode_image(bs).float().cpu().numpy())
    dino_f = np.concatenate(fd, 0).astype(np.float32)
    sig_f = np.concatenate(fs, 0).astype(np.float32)
    return np.concatenate([dino_f, sig_f], axis=1)  # (N, 1536)


class Probe(nn.Module):
    def __init__(self, in_dim=1536, n_classes=4, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_ensemble(model_path, device):
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    models = []
    for sd, (mean, scale), bias in zip(payload["state_dicts"], payload["scalers"], payload["biases"]):
        m = Probe(in_dim=sd["net.1.weight"].shape[1], n_classes=sd["net.4.weight"].shape[0])
        m.load_state_dict(sd)
        m.eval().to(device)
        models.append({
            "model": m,
            "mean": torch.tensor(mean, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "bias": torch.tensor(bias, dtype=torch.float32).to(device),
        })
    return models


@torch.no_grad()
def predict(paths, models, dino, sig, device):
    X = extract_features(paths, dino, sig, device)
    xt = torch.tensor(X, dtype=torch.float32)
    prob_sum = None
    for en in models:
        xs = (xt - en["mean"]) / en["scale"]
        logits = en["model"](xs.to(device)) + en["bias"]
        probs = F.softmax(logits, dim=1).cpu().numpy()
        prob_sum = probs if prob_sum is None else prob_sum + probs
    return prob_sum / len(models)


def main():
    ap = argparse.ArgumentParser(description="Smile Arc v5 inference")
    ap.add_argument("images", nargs="+", help="image path(s)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_ensemble(args.model, device)
    dino, sig = load_backbones(device)

    probs = predict(args.images, models, dino, sig, device)
    for p, pr in zip(args.images, probs):
        idx = int(np.argmax(pr))
        print(f"\n{p}")
        print(f"  -> {CLASSES[idx]}  (p={pr[idx]:.3f})")
        for c, pv in sorted(zip(CLASSES, pr), key=lambda t: -t[1]):
            print(f"     {c:28s} {pv:.3f}")


if __name__ == "__main__":
    main()
