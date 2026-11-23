"""Adım 1: Veri setinin gerçek yapısını göster. Hiçbir şey varsayma, önce bak.

Kaggle: python scripts/01_inspect_dataset.py --root /kaggle/input/<veri-klasörü>
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--max-depth", type=int, default=3)
args = ap.parse_args()
root = Path(args.root)

ext = Counter()
jsons, imgs, vids = [], [], []
for dirpath, dirnames, files in os.walk(root):
    dirnames.sort()
    depth = len(Path(dirpath).relative_to(root).parts)
    if depth <= args.max_depth:
        print("  " * depth + f"[DIR] {Path(dirpath).relative_to(root)} ({len(files)} dosya)")
    for f in files:
        e = Path(f).suffix.lower()
        ext[e] += 1
        p = Path(dirpath) / f
        if e == ".json" and len(jsons) < 20:
            jsons.append(p)
        elif e in {".jpg", ".jpeg", ".png", ".bmp"} and len(imgs) < 20:
            imgs.append(p)
        elif e in {".mp4", ".avi"} and len(vids) < 20:
            vids.append(p)

print("\nUzantılar:", ext.most_common())
for title, lst in (("JSON", jsons), ("Görüntü", imgs), ("Video", vids)):
    print(f"\nİlk {title} dosyaları:")
    for p in lst:
        print("  ", p.relative_to(root))

if jsons:
    with open(jsons[0], encoding="utf-8") as f:
        d = json.load(f)
    print(f"\nÖrnek JSON: {jsons[0].relative_to(root)}")
    if isinstance(d, dict):
        for k, v in d.items():
            n = len(v) if hasattr(v, "__len__") else ""
            print(f"  {k}: {type(v).__name__} {n}  örnek={str(v)[:120]}")
        if "exist" in d:
            c = Counter(d["exist"])
            print("  exist dağılımı:", dict(c))
