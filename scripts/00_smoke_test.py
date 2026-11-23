"""Deneme: tüm boru hattının çalıştığını birkaç görüntüyle gösterir (Kaggle'da ~10 dakika).

Veri setinden ilk N videonun birer karesini alır, drone etrafından kırpar ve her kırpma için:
  * modelin düşüncesini üretir (kısa: 128 token),
  * 0/32/64/128 bütçelerinde kararı okur,
  * logit lens ile kararın hangi katmanda oluştuğunu bulur,
  * düşünme boyunca görüntüye giden dikkat payını ölçer.

Önce yalnızca veri kısmını dene (model yüklemez):
  python scripts/00_smoke_test.py --root /kaggle/input/<veri> --dry
Sonra modelle:
  python scripts/00_smoke_test.py --root /kaggle/input/<veri> --modality ir --n 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tvlm.data_antiuav import FrameReader, find_videos  # noqa: E402
from tvlm.geometry import box_in_crop, context_crop, size_bin  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--modality", choices=["rgb", "ir"], default="ir")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Thinking")
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--out", default="smoke_out")
ap.add_argument("--dry", action="store_true", help="modeli yükleme, sadece kırpmaları kaydet")
args = ap.parse_args()

from PIL import Image  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# 1) Veri: her videodan drone'un göründüğü ilk kareyi al
crops = []
for va in find_videos(Path(args.root), args.modality):
    idx = next((i for i, (e, r) in enumerate(zip(va.exist, va.rects)) if e and r), None)
    if idx is None:
        continue
    reader = FrameReader(va.json_path.parent, args.modality)
    img = reader.read(idx)
    reader.close()
    H, W = img.shape[:2]
    box = va.rects[idx]
    c = context_crop(box, W, H)
    sub = img[int(c.y):int(c.y + c.h), int(c.x):int(c.x + c.w)]
    p = out / f"{va.split}_{va.video}_{args.modality}_{idx}.png"
    Image.fromarray(sub).save(p)
    crops.append({"path": str(p), "split": va.split, "video": va.video, "frame": idx,
                  "img_size": [W, H], "box": [box.x, box.y, box.w, box.h],
                  "box_in_crop": list(vars(box_in_crop(box, c)).values()), "size_bin": size_bin(box)})
    if len(crops) >= args.n:
        break

if not crops:
    sys.exit("Hiç kırpma üretilemedi. Önce 01_inspect_dataset.py ile veri yapısına bakın.")
for c in crops:
    print(f"kırpma: {c['path']}  kare={c['img_size']}  kutu={c['box']}  boyut={c['size_bin']}")
if args.dry:
    sys.exit(0)

# 2) Model
import torch  # noqa: E402
from tvlm import vlm as V  # noqa: E402

vlm = V.load_vlm(args.model)
print(f"\nModel: {args.model} | katman={len(V.decoder_layers(vlm))} | "
      f"düşünme şablonu={vlm.thinking_template} | piksel/token={vlm.pixels_per_token}")

budgets = (0, 32, 64, 128)
results = []
for c in crops:
    inputs = V.build_inputs(vlm, V.prepare_image(c["path"]), V.PROMPTS["free"])
    gen = V.generate_reasoning(vlm, inputs, args.max_new)
    rlen = V.reasoning_length(vlm, gen)
    sweep, lens_layer = {}, {}
    for b in budgets:
        ids = V.forced_sequence(vlm, inputs, gen, min(b, rlen))
        o = V.forward(vlm, inputs, ids, output_hidden_states=True)
        sweep[b] = V.logit_diff(vlm, o.logits[0, -1].float())
        d = V.logit_lens(vlm, o.hidden_states)
        s = np.sign(d[-1])
        lens_layer[b] = next(i for i in range(len(d)) if np.all(np.sign(d[i:]) == s))
        del o
    att = V.visual_attention_shares(vlm, inputs, gen[:rlen])["image"].mean(0) if rlen else np.array([])
    r = {**c, "reasoning_len": rlen,
         "reasoning": vlm.tok.decode(gen, skip_special_tokens=True),
         "logit_diff": sweep, "decision_layer": lens_layer,
         "image_attention_first_last": [float(att[:16].mean()), float(att[-16:].mean())] if len(att) >= 32 else None}
    results.append(r)
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{Path(c['path']).name}  (gerçek: DRONE, boyut={c['size_bin']})")
    print(f"düşünme ({rlen} token): {r['reasoning'][:300]!r}")
    for b in budgets:
        print(f"  B={b:>3}  YES-NO={sweep[b]:+6.2f}  karar={'DRONE' if sweep[b] > 0 else 'DEĞİL'}"
              f"  karar katmanı={lens_layer[b]}")
    if r["image_attention_first_last"]:
        f, l = r["image_attention_first_last"]
        print(f"  görüntüye dikkat: ilk 16 token={f:.3f}  son 16 token={l:.3f}")

with open(out / "smoke_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSonuçlar: {out / 'smoke_results.json'}")
