"""Görsel dil modeli (Qwen-VL ailesi) etrafında mekanistik ölçüm araçları.

Desteklenen ölçümler:
  * Bütçeli muhakeme: düşünceyi k token'da kesip modeli karara zorlama (budget forcing)
  * Logit lens: her katmanda kararın (YES - NO logit farkı) ne zaman oluştuğu
  * Görsel dikkat payı: muhakeme boyunca dikkatin görüntüye / drone bölgesine giden kısmı
  * Aktivasyon yamalama: drone'u silinmiş görüntüde, temiz görüntünün görüntü-token
    durumlarını belirli bir katmanda geri koyunca kararın ne kadar geri geldiği

Gereksinim: transformers >= 4.57, GPU (Kaggle T4/P100 16 GB yeterli, 2B-4B model).
Dikkat ağırlıkları için attn_implementation="eager" zorunludur.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image

IMAGE_SIDE = 448  # 448 = 16*28 = 14*32 -> hem Qwen2.5-VL hem Qwen3-VL ızgarasına tam oturur

PROMPTS = {
    # Serbest muhakeme
    "free": (
        "Look at the image. Is there a drone (a small multirotor UAV) in it? "
        "Think step by step about the visual evidence, then end with exactly one line: "
        "DECISION: YES or DECISION: NO"
    ),
    # Termal kanıta dayalı kısa muhakeme şablonu
    "thermal": (
        "This is a thermal infrared image: brightness means heat, not color. "
        "Is there a drone (a small multirotor UAV) in it? Reason briefly using only thermal cues: "
        "(1) is there a compact hot spot against a cooler sky or background, "
        "(2) does its heat pattern fit motors/battery of a small multirotor, "
        "(3) could it be noise or another warm object. "
        "Then end with exactly one line: DECISION: YES or DECISION: NO"
    ),
}


@dataclass
class VLM:
    model: torch.nn.Module
    processor: object
    tok: object
    yes_id: int
    no_id: int
    thinking_template: bool
    pixels_per_token: int


def load_vlm(model_id: str, dtype: torch.dtype = torch.float16, device: str = "cuda",
             attn_implementation="eager") -> VLM:
    """attn_implementation: "eager" ya da steer.register_visual_boost() sonrası
    {"text_config": steer.ATTN_NAME, "vision_config": "eager"}."""
    import transformers
    from transformers import AutoProcessor

    kw = dict(torch_dtype=dtype, attn_implementation=attn_implementation, device_map=device)
    model = None
    for cls_name in ("AutoModelForImageTextToText", "Qwen3VLForConditionalGeneration",
                     "Qwen2_5_VLForConditionalGeneration"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kw)
            break
        except (ValueError, KeyError):
            continue
    if model is None:
        raise RuntimeError(f"{model_id} yüklenemedi; transformers sürümünü güncelleyin.")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    tok = processor.tokenizer
    yes_id = tok.encode(" YES", add_special_tokens=False)[0]
    no_id = tok.encode(" NO", add_special_tokens=False)[0]
    if yes_id == no_id:
        raise RuntimeError("YES ve NO aynı ilk tokena düşüyor; karar kelimelerini değiştir.")
    ip = processor.image_processor
    ppt = int(getattr(ip, "patch_size", 14)) * int(getattr(ip, "merge_size", 2))
    probe = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        add_generation_prompt=True, tokenize=False,
    )
    thinking = probe.rstrip().endswith("<think>")
    return VLM(model, processor, tok, yes_id, no_id, thinking, ppt)


def prepare_image(path: str, side: int = IMAGE_SIDE) -> Image.Image:
    return Image.open(path).convert("RGB").resize((side, side), Image.BICUBIC)


def build_inputs(vlm: VLM, image: Image.Image, prompt: str) -> dict:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = vlm.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = vlm.processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to(vlm.model.device) for k, v in inputs.items()}


def image_token_positions(vlm: VLM, inputs: dict) -> torch.Tensor:
    img_id = vlm.model.config.image_token_id
    return (inputs["input_ids"][0] == img_id).nonzero(as_tuple=True)[0]


def image_token_grid(vlm: VLM, inputs: dict) -> tuple[int, int]:
    """(satır, sütun) cinsinden birleştirilmiş görüntü-token ızgarası."""
    t, h, w = (int(v) for v in inputs["image_grid_thw"][0])
    merge = int(getattr(vlm.processor.image_processor, "merge_size", 2))
    return h // merge, w // merge


# --------------------------------------------------------------------------- muhakeme

@torch.no_grad()
def generate_reasoning(vlm: VLM, inputs: dict, max_new_tokens: int = 512) -> torch.Tensor:
    out = vlm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    return out[0, inputs["input_ids"].shape[1]:]


def reasoning_length(vlm: VLM, gen_ids: torch.Tensor) -> int:
    """Muhakemenin bittiği token indeksi ('</think>' veya 'DECISION' başlamadan önce)."""
    def has_marker(n: int) -> bool:
        txt = vlm.tok.decode(gen_ids[:n], skip_special_tokens=False)
        return "</think>" in txt or "DECISION" in txt

    lo, hi = 0, len(gen_ids)
    if not has_marker(hi):
        return hi
    while lo < hi:  # işaretin ilk göründüğü en küçük n
        mid = (lo + hi) // 2
        if has_marker(mid):
            hi = mid
        else:
            lo = mid + 1
    return max(lo - 1, 0)


def forced_sequence(vlm: VLM, inputs: dict, gen_ids: torch.Tensor, k: int) -> torch.Tensor:
    """İstem + ilk k muhakeme tokenı + karar soneki."""
    prefix = gen_ids[:k]
    ptxt = vlm.tok.decode(prefix, skip_special_tokens=False)
    think_open = (vlm.thinking_template or "<think>" in ptxt) and "</think>" not in ptxt
    suffix = "\n</think>\n\nDECISION:" if think_open else "\nDECISION:"
    suf_ids = vlm.tok(suffix, add_special_tokens=False, return_tensors="pt").input_ids
    return torch.cat([inputs["input_ids"], prefix.unsqueeze(0).to(inputs["input_ids"].device),
                      suf_ids.to(inputs["input_ids"].device)], dim=1)


@torch.no_grad()
def forward(vlm: VLM, inputs: dict, input_ids: torch.Tensor, **kw):
    return vlm.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"], **kw)


def logit_diff(vlm: VLM, logits_last: torch.Tensor) -> float:
    return float(logits_last[vlm.yes_id] - logits_last[vlm.no_id])


@torch.no_grad()
def budget_sweep(vlm: VLM, inputs: dict, gen_ids: torch.Tensor,
                 budgets: Sequence[int]) -> dict[int, float]:
    """Her bütçe için zorlanmış kararın YES-NO logit farkı. >0 => YES (drone)."""
    rlen = reasoning_length(vlm, gen_ids)
    res = {}
    for b in budgets:
        ids = forced_sequence(vlm, inputs, gen_ids, min(b, rlen))
        out = forward(vlm, inputs, ids)
        res[b] = logit_diff(vlm, out.logits[0, -1].float())
    return res


# --------------------------------------------------------------------------- iç yapı erişimi

def _language_model(model: torch.nn.Module) -> torch.nn.Module:
    for path in ("model.language_model", "model", "language_model.model"):
        m = model
        try:
            for p in path.split("."):
                m = getattr(m, p)
        except AttributeError:
            continue
        if hasattr(m, "layers") and hasattr(m, "norm"):
            return m
    raise AttributeError("Dil modeli katmanları bulunamadı")


def decoder_layers(vlm: VLM) -> torch.nn.ModuleList:
    return _language_model(vlm.model).layers


# --------------------------------------------------------------------------- logit lens

@torch.no_grad()
def logit_lens(vlm: VLM, hidden_states: Sequence[torch.Tensor], position: int = -1) -> np.ndarray:
    """Her katmanın çıkışında YES-NO logit farkı. hidden_states: output_hidden_states çıktısı.

    HF Qwen modellerinde hidden_states[-1] zaten son normdan geçmiştir; ara katmanlara
    normu biz uygularız.
    """
    lm = _language_model(vlm.model)
    head = vlm.model.lm_head
    W = head.weight[[vlm.yes_id, vlm.no_id]]  # (2, d)
    diffs = []
    n = len(hidden_states)
    for i in range(1, n):
        h = hidden_states[i][0, position]
        if i < n - 1:
            h = lm.norm(h)
        lg = (W.float() @ h.float())
        diffs.append(float(lg[0] - lg[1]))
    return np.array(diffs)


# --------------------------------------------------------------------------- dikkat

@torch.no_grad()
def visual_attention_shares(vlm: VLM, inputs: dict, gen_ids: torch.Tensor,
                            region_mask: Optional[np.ndarray] = None,
                            layers: Optional[Iterable[int]] = None) -> dict[str, np.ndarray]:
    """Her muhakeme tokenında dikkatin görüntüye ve drone bölgesine giden payı.

    Dönüş: {"image": (katman, adım), "region": (katman, adım)} — başlar üzerinden ortalama.
    """
    ids = torch.cat([inputs["input_ids"], gen_ids.unsqueeze(0)], dim=1)
    out = forward(vlm, inputs, ids, output_attentions=True)
    img_pos = image_token_positions(vlm, inputs)
    start = inputs["input_ids"].shape[1]
    q = torch.arange(start, ids.shape[1], device=ids.device)
    region_pos = None
    if region_mask is not None:
        flat = torch.from_numpy(region_mask.reshape(-1)).to(ids.device)
        region_pos = img_pos[flat[: len(img_pos)]]
    sel = list(range(len(out.attentions))) if layers is None else list(layers)
    img, reg = [], []
    for li in sel:
        a = out.attentions[li][0].float().mean(0)  # (T, T) başlar ortalaması
        rows = a[q]
        img.append(rows[:, img_pos].sum(-1).cpu().numpy())
        if region_pos is not None and len(region_pos):
            reg.append(rows[:, region_pos].sum(-1).cpu().numpy())
    res = {"image": np.stack(img)}
    if reg:
        res["region"] = np.stack(reg)
    return res


# --------------------------------------------------------------------------- aktivasyon yamalama

@contextmanager
def _hook(module: torch.nn.Module, fn):
    h = module.register_forward_hook(fn)
    try:
        yield
    finally:
        h.remove()


def _hidden_of(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def patch_image_tokens(vlm: VLM, clean_inputs: dict, corrupt_inputs: dict,
                       input_ids: torch.Tensor, layer: int,
                       positions: torch.Tensor) -> dict[str, float]:
    """Bozuk (drone silinmiş) çalıştırmada `layer` çıkışındaki `positions` durumlarını
    temiz çalıştırmanınkilerle değiştirir. İki görüntü aynı boyutta olmalı (aynı token sayısı).

    Dönüş: temiz, bozuk, yamalı logit farkları ve normalize geri kazanım
    (1 = karar tamamen geri geldi, 0 = hiç etkisi yok).
    """
    layers = decoder_layers(vlm)
    cache = {}

    def save(_m, _i, o):
        cache["h"] = _hidden_of(o)[:, positions].clone()

    with _hook(layers[layer], save):
        clean = forward(vlm, clean_inputs, input_ids)
    corrupt = forward(vlm, corrupt_inputs, input_ids)

    def put(_m, _i, o):
        h = _hidden_of(o)
        h[:, positions] = cache["h"].to(h.dtype)
        return o

    with _hook(layers[layer], put):
        patched = forward(vlm, corrupt_inputs, input_ids)

    ld_c = logit_diff(vlm, clean.logits[0, -1].float())
    ld_x = logit_diff(vlm, corrupt.logits[0, -1].float())
    ld_p = logit_diff(vlm, patched.logits[0, -1].float())
    denom = ld_c - ld_x
    rec = (ld_p - ld_x) / denom if abs(denom) > 1e-6 else float("nan")
    return {"clean": ld_c, "corrupt": ld_x, "patched": ld_p, "recovery": rec}
