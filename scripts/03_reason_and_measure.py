"""Adım 3 (ana deney): her kırpma için
  (a) muhakemeyi bir kez 512 tokena kadar üret (greedy),
  (b) 0/32/64/128/256/512 bütçelerinde modeli karara zorla (davranışsal B*),
  (c) her bütçe noktasında logit lens ile kararın hangi katmanda oluştuğunu kaydet,
  (d) muhakeme boyunca görüntüye ve drone bölgesine giden dikkat payını kaydet,
  (e) probe için gizli durum özelliklerini kaydet.

python scripts/03_reason_and_measure.py --manifest /kaggle/working/crops/manifest_ir.csv \
    --model Qwen/Qwen3-VL-2B-Thinking --prompt free --out /kaggle/working/runs/ir_free \
    --split val --limit 200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tvlm.budget import DEFAULT_BUDGETS  # noqa: E402
from tvlm.data_antiuav import read_manifest  # noqa: E402
from tvlm.geometry import Box, token_grid_mask  # noqa: E402
from tvlm.probe import CHECKPOINTS, collect_features  # noqa: E402
from tvlm import vlm as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--prompt", choices=list(V.PROMPTS), default="free")
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-new", type=int, default=512)
ap.add_argument("--probe-layers", default="-1", help="virgüllü katman listesi; -1 = her 4 katmanda bir")
ap.add_argument("--no-attention", action="store_true", help="dikkat ölçümünü atla (bellek)")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rows = [r for r in read_manifest(Path(args.manifest)) if r["split"] == args.split]
if args.limit:
    rows = rows[: args.limit]

vlm = V.load_vlm(args.model)
n_layers = len(V.decoder_layers(vlm))
probe_layers = (list(range(0, n_layers, 4)) + [n_layers - 1]) if args.probe_layers == "-1" \
    else [int(x) for x in args.probe_layers.split(",")]
prompt = V.PROMPTS[args.prompt]

done = {p.stem for p in out.glob("*.json")}
for i, r in enumerate(rows):
    cid = r["crop_id"]
    if cid in done:
        continue
    img = V.prepare_image(r["path"])
    inputs = V.build_inputs(vlm, img, prompt)
    plen = inputs["input_ids"].shape[1]
    gen = V.generate_reasoning(vlm, inputs, args.max_new)
    rlen = V.reasoning_length(vlm, gen)

    # (b) + (c): bütçe taraması ve her bütçede logit lens
    sweep, lens = {}, {}
    for b in DEFAULT_BUDGETS:
        ids = V.forced_sequence(vlm, inputs, gen, min(b, rlen))
        o = V.forward(vlm, inputs, ids, output_hidden_states=True)
        sweep[b] = V.logit_diff(vlm, o.logits[0, -1].float())
        lens[b] = V.logit_lens(vlm, o.hidden_states).tolist()
        del o

    # (e) probe özellikleri: muhakeme dizisi üzerinde tek ileri geçiş
    full_ids = torch.cat([inputs["input_ids"], gen[:rlen].unsqueeze(0)], dim=1)
    o = V.forward(vlm, inputs, full_ids, output_hidden_states=True)
    feats = collect_features(o.hidden_states, plen, rlen, probe_layers, CHECKPOINTS)
    np.save(out / f"{cid}_feats.npy", feats.astype(np.float16))
    del o

    # (d) dikkat payları
    att = {}
    if not args.no_attention and rlen > 0:
        mask = None
        if int(r["label"]) == 1:
            gh, gw = V.image_token_grid(vlm, inputs)
            side = V.IMAGE_SIDE
            crop_side = V.Image.open(r["path"]).size[0]
            mask = token_grid_mask(Box(float(r["bx"]), float(r["by"]), float(r["bw"]), float(r["bh"])),
                                   crop_side, crop_side, side, side, vlm.pixels_per_token)
            assert mask.shape == (gh, gw), (mask.shape, gh, gw)
        sh = V.visual_attention_shares(vlm, inputs, gen[:rlen], mask)
        att = {k: v.mean(0).tolist() for k, v in sh.items()}  # katman ortalaması, adım boyunca

    rec = {
        "crop_id": cid, "label": int(r["label"]), "size_bin": r["size_bin"],
        "modality": r["modality"], "prompt": args.prompt, "model": args.model,
        "reasoning_len": rlen, "gen_text": vlm.tok.decode(gen, skip_special_tokens=False),
        "sweep_logit_diff": {str(k): v for k, v in sweep.items()},
        "lens_by_budget": {str(k): v for k, v in lens.items()},
        "attention": att, "probe_layers": probe_layers, "checkpoints": list(CHECKPOINTS),
    }
    with open(out / f"{cid}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    torch.cuda.empty_cache()
    print(f"[{i + 1}/{len(rows)}] {cid} label={rec['label']} rlen={rlen} "
          f"LD@0={sweep[0]:+.2f} LD@512={sweep[512]:+.2f}", flush=True)
