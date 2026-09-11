# v5 Phase 2: label-error suspect ranking (cleanlab-style) -> CSV + HTML review artifact
# Confident-learning heuristic: rank by p(gold) low vs p(pred) high on OOF probs of the locked baseline probe.
import numpy as _npB
import json as _jsonB
import base64 as _b64B
import io as _ioB
from PIL import Image as _ImgB
from sklearn.metrics import f1_score as _f1B, confusion_matrix as _cmB

_Xb, _yb = v5_feats, v5_y
_splits_b = v5_splits()
_rev_b = [i for i, c in enumerate(v5_classes) if "reverse" in c.lower()][0]

# --- OOF probs from the locked baseline (seed 0) ---
_oof_b = _npB.zeros((len(_yb), 4), dtype=_npB.float32)
for _tr, _va in _splits_b:
    _, _p = v5_train_mlp(_Xb, _yb, _tr, _va, seed=0)
    _oof_b[_va] = _p
v5_oof_probs = _oof_b  # public: reused by later phases

_pred_b = _oof_b.argmax(1)
_pgold = _oof_b[_npB.arange(len(_yb)), _yb]
_ppred = _oof_b.max(1)
_margin = _ppred - _pgold

print("OOF macro-F1:", round(_f1B(_yb, _pred_b, average="macro", zero_division=0), 4))
print("OOF confusion (rows=true):")
print(_cmB(_yb, _pred_b))

# --- suspicion score: likely label error if gold prob low & pred prob high ---
_susp = _margin + 0.5 * (1.0 - _pgold)   # margin dominates, low gold-confidence adds
_priority = (_yb == _rev_b) | (_yb == 3) | (_pred_b == _rev_b)  # minority rows + anything predicted reverse

_order = _npB.lexsort((-_susp, ~_priority))  # priority class first, then by suspicion
_top_n = 120  # ~11% of data, within plan's 5-10% + minority emphasis
_sel = _order[:_top_n]

# --- CSV ---
_rows_csv = ["path,true_class,pred_class,p_gold,p_pred,margin,priority,suggested_fix,reviewer_verdict"]
_rows_data = []
for _i in _sel:
    fix = v5_classes[_pred_b[_i]]
    _rows_data.append({
        "path": v5_paths[_i], "true_class": v5_classes[_yb[_i]],
        "pred_class": v5_classes[_pred_b[_i]],
        "p_gold": round(float(_pgold[_i]), 3), "p_pred": round(float(_ppred[_i]), 3),
        "margin": round(float(_margin[_i]), 3), "priority": bool(_priority[_i]),
        "suggested_fix": fix,
    })
    _rows_csv.append(f"{v5_paths[_i]},{v5_classes[_yb[_i]]},{v5_classes[_pred_b[_i]]},"
                     f"{_pgold[_i]:.3f},{_ppred[_i]:.3f},{_margin[_i]:.3f},{int(_priority[_i])},{fix},")
with open("/marimo/v5_label_review.csv", "w") as _f:
    _f.write("\n".join(_rows_csv))

# --- HTML with embedded thumbnails ---
_thumb = 150
_cards = []
for _i in _sel:
    im = _ImgB.open(v5_paths[_i]).convert("RGB")
    im.thumbnail((_thumb, _thumb))
    buf = _ioB.BytesIO()
    im.save(buf, "JPEG", quality=70)
    uri = "data:image/jpeg;base64," + _b64B.b64encode(buf.getvalue()).decode()
    tag = '<span style="background:#fee2e2;color:#b91c1c;padding:1px 6px;border-radius:4px;font-size:11px">PRIORITY</span>' if _priority[_i] else ""
    _cards.append(f'''<div style="border:1px solid #ddd;border-radius:8px;padding:8px;width:170px;font-size:12px;font-family:sans-serif">
<img src="{uri}" style="width:150px;height:112px;object-fit:cover;border-radius:4px"><br>
<b style="color:#0645ad">TRUE:</b> {v5_classes[_yb[_i]]}<br>
<b style="color:#8b0000">PRED:</b> {v5_classes[_pred_b[_i]]} ({_ppred[_i]:.2f})<br>
p_gold={_pgold[_i]:.2f} {tag}<br>
<i style="color:#555">{v5_paths[_i].split("/")[-1]}</i><br>
Suggested: <b>{v5_classes[_pred_b[_i]]}</b>
</div>''')

_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>v5 Label Review</title></head>
<body style="font-family:sans-serif;margin:20px">
<h2>Smile-Arc Label Review — top {_top_n} suspects (cleanlab-style ranking)</h2>
<p>Fill <code>reviewer_verdict</code> in <code>/marimo/v5_label_review.csv</code>: keep | fix_to:&lt;class&gt; | unsure.
Suggested fix = OOF argmax of the locked baseline probe. PRIORITY = reverse/straight or predicted-reverse rows.</p>
<div style="display:flex;flex-wrap:wrap;gap:10px">{''.join(_cards)}</div></body></html>"""
with open("/marimo/v5_label_review.html", "w") as _f:
    _f.write(_html)

print(f"\nreview artifact: {_top_n} suspects")
print("  priority rows:", int(_priority[_sel].sum()))
print("  -> /marimo/v5_label_review.csv")
print("  -> /marimo/v5_label_review.html")
print("\nTop 15 suspects:")
for _i in _sel[:15]:
    print(f"  [{_susp[_i]:.2f}] true={v5_classes[_yb[_i]]:26s} pred={v5_classes[_pred_b[_i]]:26s} p_gold={_pgold[_i]:.2f}  {v5_paths[_i].split('/')[-1]}")

v5_label_review_info = {"n_suspects": _top_n, "csv": "/marimo/v5_label_review.csv", "html": "/marimo/v5_label_review.html"}
v5_label_review_info
