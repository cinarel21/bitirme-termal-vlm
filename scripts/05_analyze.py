"""Adım 5: Sonuçları özetle — B* dağılımı, kararsızlık, logit lens 'karar katmanı',
dikkat eğrileri, probe doğruluğu ve probe tabanlı erken durdurmanın token tasarrufu.

Probe'lar TRAIN split çıktılarıyla eğitilir, VAL üzerinde eşik seçilir; TEST en sona saklanır.

python scripts/05_analyze.py --train /kaggle/working/runs/ir_free_train \
    --eval /kaggle/working/runs/ir_free_val --out /kaggle/working/report_ir_free
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tvlm.budget import DEFAULT_BUDGETS, bstar_distribution, instability_rate  # noqa: E402
from tvlm.probe import early_exit, train_probes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--threshold", type=float, default=0.9)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def load(d):
    recs = [json.load(open(p, encoding="utf-8")) for p in sorted(Path(d).glob("*.json"))]
    feats = np.stack([np.load(Path(d) / f"{r['crop_id']}_feats.npy").astype(np.float32) for r in recs])
    return recs, feats


tr, Xtr = load(args.train)
ev, Xev = load(args.eval)
ytr = np.array([r["label"] for r in tr])
yev = np.array([r["label"] for r in ev])

report = {}

# 1) Davranışsal: B* dağılımı ve kararsızlık (etiket ve boyut sınıfına göre)
groups = defaultdict(list)
for r in ev:
    correct = {int(b): (v > 0) == bool(r["label"]) for b, v in r["sweep_logit_diff"].items()}
    groups["hepsi"].append(correct)
    groups[f"etiket={r['label']}"].append(correct)
    if r["label"] == 1:
        groups[f"boyut={r['size_bin']}"].append(correct)
report["bstar"] = {g: bstar_distribution(v) for g, v in groups.items()}
report["instability"] = {g: instability_rate(v) for g, v in groups.items()}
report["accuracy_by_budget"] = {
    str(b): float(np.mean([(r["sweep_logit_diff"][str(b)] > 0) == bool(r["label"]) for r in ev]))
    for b in DEFAULT_BUDGETS
}

# 2) Logit lens: kararın işaretinin son katmanla aynı olduğu ilk katman (B=0 ve B=512)
def decision_layer(diffs):
    s = np.sign(diffs[-1])
    for i in range(len(diffs)):
        if np.all(np.sign(diffs[i:]) == s):
            return i
    return len(diffs) - 1


report["decision_layer"] = {
    str(b): float(np.mean([decision_layer(np.array(r["lens_by_budget"][str(b)])) for r in ev]))
    for b in (0, 512)
}

# 3) Dikkat: muhakeme ilerledikçe görüntüye giden pay (ilk 32 vs son 32 token)
img_first, img_last = [], []
for r in ev:
    a = r["attention"].get("image")
    if a and len(a) >= 64:
        img_first.append(np.mean(a[:32]))
        img_last.append(np.mean(a[-32:]))
if img_first:
    report["image_attention_first32_vs_last32"] = [float(np.mean(img_first)), float(np.mean(img_last))]

# 4) Probe: hangi checkpoint/katmanda etiket doğrusal olarak okunabiliyor
res, models = train_probes(Xtr, ytr, Xev, yev)
report["probe_auroc"] = res.auroc.tolist()
report["probe_accuracy"] = res.accuracy.tolist()

# 5) Erken durdurma: en iyi katmanın probe'u ile
best_layer = int(np.nanargmax(np.nanmean(res.auroc, axis=0)))
ck = tr[0]["checkpoints"]
probs = np.stack([models[(c, best_layer)].predict_proba(Xev[:, c, best_layer])[:, 1]
                  for c in range(len(ck))], axis=1)
full_dec = np.array([int(r["sweep_logit_diff"]["512"] > 0) for r in ev])
rlens = np.array([r["reasoning_len"] for r in ev])
ee = early_exit(probs, ck, args.threshold, full_dec, rlens)
report["early_exit"] = {
    "threshold": args.threshold, "probe_layer_index": best_layer,
    "mean_tokens_full": float(rlens.mean()), "mean_tokens_exit": float(ee["tokens"].mean()),
    "acc_full": float((full_dec == yev).mean()), "acc_exit": float((ee["decisions"] == yev).mean()),
}

with open(out / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(json.dumps(report["early_exit"], indent=2, ensure_ascii=False))
print("Rapor:", out / "report.json")
