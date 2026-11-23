"""Adım 4 (nedensellik): Karar görüntüden mi geliyor, dil önyargısından mı?

Her pozitif kırpma için temiz görüntü ile drone'u silinmiş (inpaint) görüntü karşılaştırılır.
Bozuk çalıştırmada belirli bir katmanda görüntü-token durumları temiz olanlarla değiştirilir:
  * tüm görüntü tokenları
  * yalnızca drone bölgesindeki tokenlar
Kararın ne kadar geri geldiği (recovery) katman katman kaydedilir. Termalde drone bölgesi
yaması kararı geri getirmiyorsa, model termal kanıtı kullanmıyor demektir.

python scripts/04_patching.py --manifest .../manifest_ir.csv --model Qwen/Qwen3-VL-2B-Thinking \
    --out /kaggle/working/runs/ir_patch --split val --limit 100
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tvlm.data_antiuav import read_manifest  # noqa: E402
from tvlm.geometry import Box, token_grid_mask  # noqa: E402
from tvlm import vlm as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--layer-step", type=int, default=2)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rows = [r for r in read_manifest(Path(args.manifest))
        if r["split"] == args.split and int(r["label"]) == 1 and r["corrupt_path"]]
if args.limit:
    rows = rows[: args.limit]

vlm = V.load_vlm(args.model)
L = len(V.decoder_layers(vlm))
prompt = V.PROMPTS["free"]

for i, r in enumerate(rows):
    clean = V.build_inputs(vlm, V.prepare_image(r["path"]), prompt)
    corrupt = V.build_inputs(vlm, V.prepare_image(r["corrupt_path"]), prompt)
    assert torch.equal(clean["input_ids"], corrupt["input_ids"]), "token sayıları farklı"
    ids = V.forced_sequence(vlm, clean, torch.empty(0, dtype=torch.long), 0)  # B=0 kararı
    img_pos = V.image_token_positions(vlm, clean)
    crop_side = V.Image.open(r["path"]).size[0]
    mask = token_grid_mask(Box(float(r["bx"]), float(r["by"]), float(r["bw"]), float(r["bh"])),
                           crop_side, crop_side, V.IMAGE_SIDE, V.IMAGE_SIDE, vlm.pixels_per_token)
    flat = torch.from_numpy(mask.reshape(-1)).to(img_pos.device)
    region_pos = img_pos[flat[: len(img_pos)]]
    res = {"crop_id": r["crop_id"], "size_bin": r["size_bin"], "modality": r["modality"],
           "n_region_tokens": int(len(region_pos)), "layers": {}}
    for l in range(0, L, args.layer_step):
        res["layers"][l] = {
            "all_image": V.patch_image_tokens(vlm, clean, corrupt, ids, l, img_pos),
            "drone_region": V.patch_image_tokens(vlm, clean, corrupt, ids, l, region_pos)
            if len(region_pos) else None,
        }
    with open(out / f"{r['crop_id']}.json", "w", encoding="utf-8") as f:
        json.dump(res, f)
    torch.cuda.empty_cache()
    print(f"[{i + 1}/{len(rows)}] {r['crop_id']} bölge_token={len(region_pos)}", flush=True)
