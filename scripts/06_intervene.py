"""Adım 6: Düşünme katmanlarına müdahale ve etkisinin ölçülmesi.

Aşama A (kalibrasyon, TRAIN): müdahale yönlerini hesapla
  * durma yönü: muhakeme sonundaki gizli durum - muhakeme ortasındaki gizli durum
  * termal->RGB yönü: aynı karelerin RGB ve IR görüntü-token ortalamaları farkı (--rgb-manifest)
Aşama B (değerlendirme, VAL): her örnekte koşulları karşılaştır
  * base, halt(alpha), boost(alpha), t2r(alpha)
  Ölçütler: doğruluk, muhakeme uzunluğu (token), süre, karar verememe oranı.

python scripts/06_intervene.py --manifest .../manifest_ir.csv --rgb-manifest .../manifest_rgb.csv \
    --model Qwen/Qwen3-VL-2B-Thinking --layer 14 --calib 60 --limit 60 \
    --halt 0.05,0.1 --boost 1.5,2.0 --t2r 0.05 --out /kaggle/working/runs/ir_intervene
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tvlm import steer as S  # noqa: E402
from tvlm import vlm as V  # noqa: E402
from tvlm.data_antiuav import read_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, help="IR manifest")
ap.add_argument("--rgb-manifest", default="", help="termal->RGB yönü için RGB manifest")
ap.add_argument("--model", required=True)
ap.add_argument("--layer", type=int, default=-1, help="müdahale katmanı; -1 = orta katman")
ap.add_argument("--boost-layers", default="", help="virgüllü; boş = orta üçte bir")
ap.add_argument("--calib", type=int, default=60)
ap.add_argument("--limit", type=int, default=60)
ap.add_argument("--max-new", type=int, default=512)
ap.add_argument("--halt", default="0.05,0.1")
ap.add_argument("--boost", default="1.5,2.0")
ap.add_argument("--t2r", default="")
ap.add_argument("--out", required=True)
args = ap.parse_args()


def floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
halts, boosts, t2rs = floats(args.halt), floats(args.boost), floats(args.t2r)

attn = "eager"
if boosts:
    attn = {"text_config": S.register_visual_boost(), "vision_config": "eager"}
vlm = V.load_vlm(args.model, attn_implementation=attn)
layers = V.decoder_layers(vlm)
L = len(layers)
layer = L // 2 if args.layer < 0 else args.layer
boost_layers = [int(x) for x in args.boost_layers.split(",")] if args.boost_layers \
    else list(range(L // 3, 2 * L // 3))
prompt = V.PROMPTS["free"]
ir_rows = read_manifest(Path(args.manifest))

# ------------------------------------------------------------------ A) kalibrasyon
dir_path = out / f"directions_L{layer}.npz"
if dir_path.exists():
    D = dict(np.load(dir_path))
else:
    ends, mids = [], []
    for r in [r for r in ir_rows if r["split"] == "train"][: args.calib]:
        inp = V.build_inputs(vlm, V.prepare_image(r["path"]), prompt)
        gen = V.generate_reasoning(vlm, inp, args.max_new)
        rlen = V.reasoning_length(vlm, gen)
        if rlen < 16:
            continue
        plen = inp["input_ids"].shape[1]
        ids = torch.cat([inp["input_ids"], gen[:rlen].unsqueeze(0)], dim=1)
        o = V.forward(vlm, inp, ids, output_hidden_states=True)
        hs = o.hidden_states[layer + 1][0]
        ends.append(hs[plen + rlen - 1].float().cpu().numpy())
        mids.append(hs[plen + rlen // 2].float().cpu().numpy())
        del o
    D = {"halt": S.compute_direction(np.stack(ends), np.stack(mids))}
    print(f"durma yönü: {len(ends)} örnekten, katman {layer}")

    if args.rgb_manifest:
        rgb = {(r["split"], r["video"], r["frame"]): r for r in read_manifest(Path(args.rgb_manifest))
               if int(r["label"]) == 1}
        rgb_m, ir_m = [], []
        for r in [r for r in ir_rows if r["split"] == "train" and int(r["label"]) == 1]:
            k = (r["split"], r["video"], r["frame"])
            if k not in rgb:
                continue
            for path, acc in ((rgb[k]["path"], rgb_m), (r["path"], ir_m)):
                inp = V.build_inputs(vlm, V.prepare_image(path), prompt)
                pos = V.image_token_positions(vlm, inp)
                o = V.forward(vlm, inp, inp["input_ids"], output_hidden_states=True)
                acc.append(o.hidden_states[layer + 1][0, pos].float().mean(0).cpu().numpy())
                del o
            if len(ir_m) >= args.calib:
                break
        if ir_m:
            D["t2r"] = S.compute_direction(np.stack(rgb_m), np.stack(ir_m))
            print(f"termal->RGB yönü: {len(ir_m)} eşleşmiş kareden")
    np.savez(dir_path, **D)

# ------------------------------------------------------------------ B) değerlendirme
conds = [("base", None, 0.0)] + [("halt", "halt", a) for a in halts] + \
        [("boost", None, a) for a in boosts] + \
        ([("t2r", "t2r", a) for a in t2rs] if "t2r" in D else [])
eval_rows = [r for r in ir_rows if r["split"] == "val"][: args.limit]
stats = defaultdict(lambda: {"correct": 0, "nodec": 0, "tokens": [], "secs": [], "n": 0})

for i, r in enumerate(eval_rows):
    inp = V.build_inputs(vlm, V.prepare_image(r["path"]), prompt)
    img_pos = V.image_token_positions(vlm, inp)
    y = int(r["label"])
    rec = {"crop_id": r["crop_id"], "label": y, "size_bin": r["size_bin"], "conds": {}}
    for name, dkey, a in conds:
        steers = []
        if dkey == "halt":
            steers = [S.ResidualSteer(layer, torch.from_numpy(D["halt"]), a, "generated")]
        elif dkey == "t2r":
            steers = [S.ResidualSteer(layer, torch.from_numpy(D["t2r"]), a, "image", img_pos)]
        torch.cuda.synchronize()
        t0 = time.time()
        with S.residual_steering(layers, steers):
            if name == "boost":
                with S.visual_boost(a, boost_layers, img_pos):
                    gen = V.generate_reasoning(vlm, inp, args.max_new)
            else:
                gen = V.generate_reasoning(vlm, inp, args.max_new)
        torch.cuda.synchronize()
        secs = time.time() - t0
        text = vlm.tok.decode(gen, skip_special_tokens=True)
        dec = S.parse_decision(text)
        key = f"{name}@{a:g}" if name != "base" else "base"
        st = stats[key]
        st["n"] += 1
        st["correct"] += int(dec == y)
        st["nodec"] += int(dec is None)
        st["tokens"].append(V.reasoning_length(vlm, gen))
        st["secs"].append(secs)
        rec["conds"][key] = {"decision": dec, "tokens": st["tokens"][-1], "secs": secs, "text": text}
    with open(out / f"{r['crop_id']}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    print(f"[{i + 1}/{len(eval_rows)}] {r['crop_id']}", flush=True)

summary = {k: {"n": v["n"], "accuracy": v["correct"] / v["n"], "no_decision": v["nodec"] / v["n"],
               "mean_tokens": float(np.mean(v["tokens"])), "mean_secs": float(np.mean(v["secs"]))}
           for k, v in stats.items()}
with open(out / "summary.json", "w", encoding="utf-8") as f:
    json.dump({"layer": layer, "boost_layers": boost_layers, "summary": summary}, f, indent=2)
print(f"\n{'koşul':<14}{'doğruluk':>10}{'karar yok':>11}{'token':>8}{'saniye':>9}")
for k, s in summary.items():
    print(f"{k:<14}{s['accuracy']:>10.3f}{s['no_decision']:>11.3f}{s['mean_tokens']:>8.1f}{s['mean_secs']:>9.2f}")
