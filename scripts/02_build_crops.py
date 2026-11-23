"""Adım 2: Anti-UAV300'den pozitif (drone), eşleşmiş negatif (arka plan) ve drone'u
silinmiş (inpaint) kırpmalar üret. Video düzeyindeki train/val/test bölmesi korunur.

python scripts/02_build_crops.py --root /kaggle/input/<veri> --out /kaggle/working/crops --modality ir
python scripts/02_build_crops.py --root /kaggle/input/<veri> --out /kaggle/working/crops --modality rgb
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tvlm.data_antiuav import build_crops, write_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--modality", choices=["rgb", "ir"], required=True)
ap.add_argument("--frames-per-video", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rows = build_crops(Path(args.root), Path(args.out), args.modality,
                   frames_per_video=args.frames_per_video, seed=args.seed)
if not rows:
    sys.exit("Hiç kırpma üretilmedi: 01_inspect_dataset çıktısına göre data_antiuav.py'yi uyarlayın.")
man = Path(args.out) / f"manifest_{args.modality}.csv"
write_manifest(rows, man)
print(f"{len(rows)} kırpma -> {man}")
print("split x etiket:", Counter((r.split, r.label) for r in rows))
print("boyut sınıfı (pozitif):", Counter(r.size_bin for r in rows if r.label == 1))
