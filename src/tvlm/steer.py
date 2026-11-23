"""Düşünme katmanlarına müdahale (eğitimsiz, çıkarım anında).

Üç müdahale:
  1. Durma yönü (halt steering): muhakemenin bittiği andaki gizli durum ile ortasındaki
     gizli durum arasındaki fark vektörü. Üretim sırasında belirli bir katmanın çıkışına
     alpha * yön eklenir. alpha > 0 düşünmeyi kısaltır, alpha < 0 uzatır.
  2. Görsel dikkat güçlendirme: düşünme tokenları üretilirken, seçilen katmanlarda
     görüntü tokenlarına giden dikkat logitlerine log(alpha) eklenir. Amaç: model
     düşündükçe görüntüden kopmasın.
  3. Termal -> RGB kaydırma: aynı sahnelerin RGB ve IR görüntü-token temsilleri
     arasındaki ortalama fark. IR görüntü tokenlarına eklenir; model termali,
     daha iyi tanıdığı RGB temsillerine yakın işlesin.

Önce ölç (03-05), sonra müdahale et (06). Müdahalenin etkisi her zaman müdahalesiz
temel çalıştırmayla aynı örneklerde karşılaştırılır.
"""
from __future__ import annotations

import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import torch

# --------------------------------------------------------------------------- yön hesabı (CPU)


def compute_direction(pos: np.ndarray, neg: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Ortalama farkı yönü: mean(pos) - mean(neg). pos/neg: (örnek, d)."""
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
        raise ValueError("pos ve neg (örnek, d) boyutunda ve aynı d'ye sahip olmalı")
    d = pos.mean(0) - neg.mean(0)
    if normalize:
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Yön sıfır vektör: iki küme ayırt edilemiyor")
        d = d / n
    return d


def parse_decision(text: str) -> Optional[int]:
    """Üretilen metindeki son 'DECISION: YES/NO' satırını okur. 1=drone, 0=değil, None=yok."""
    m = re.findall(r"DECISION\s*:\s*(YES|NO)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return 1 if m[-1].upper() == "YES" else 0


# --------------------------------------------------------------------------- 1 & 3: artık akışa ekleme


@dataclass
class ResidualSteer:
    """Bir decoder katmanının çıkışına yön ekler.

    where="generated": yalnızca üretim adımlarında (tek tokenlık ileri geçişler) uygulanır.
    where="image":     yalnızca ön-doldurma (prefill) sırasında görüntü-token konumlarına uygulanır.
    Yön, katmandaki gizli durumların ortalama normuyla ölçeklenir; alpha bu ölçeğin katıdır.
    """
    layer: int
    direction: torch.Tensor
    alpha: float
    where: str = "generated"
    image_positions: Optional[torch.Tensor] = None


@contextmanager
def residual_steering(layers: torch.nn.ModuleList, steers: Iterable[ResidualSteer]):
    handles = []

    def make_hook(s: ResidualSteer):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            seq = h.shape[1]
            v = s.direction.to(h.device, h.dtype)
            scale = h.float().norm(dim=-1).mean().to(h.dtype)
            if s.where == "generated" and seq == 1:
                h[:, -1] += s.alpha * scale * v
            elif s.where == "image" and seq > 1 and s.image_positions is not None:
                h[:, s.image_positions] += s.alpha * scale * v
            return out
        return hook

    try:
        for s in steers:
            handles.append(layers[s.layer].register_forward_hook(make_hook(s)))
        yield
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- 2: görsel dikkat güçlendirme


@dataclass
class _AttnState:
    active: bool = False
    alpha: float = 1.0
    layers: set = field(default_factory=set)
    image_positions: Optional[torch.Tensor] = None
    decode_only: bool = True


_STATE = _AttnState()
ATTN_NAME = "tvlm_visual_boost"


def _repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)


def visual_boost_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Eager dikkat + görüntü tokenlarına log(alpha) önyargısı (yalnızca etkin katmanlarda)."""
    n_rep = query.shape[1] // key.shape[1]
    k = _repeat_kv(key, n_rep)
    v = _repeat_kv(value, n_rep)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : k.shape[-2]]
    st = _STATE
    li = getattr(module, "layer_idx", None)
    if (st.active and li in st.layers and st.image_positions is not None
            and (query.shape[2] == 1 or not st.decode_only)):
        pos = st.image_positions[st.image_positions < k.shape[-2]].to(w.device)
        w[..., pos] = w[..., pos] + math.log(st.alpha)
    w = torch.softmax(w.float(), dim=-1).to(query.dtype)
    if dropout and module.training:
        w = torch.nn.functional.dropout(w, p=dropout)
    out = torch.matmul(w, v).transpose(1, 2).contiguous()
    return out, w


def register_visual_boost() -> str:
    """Özel dikkat fonksiyonunu transformers'a kaydeder. Model şu şekilde yüklenmeli:
    attn_implementation={"text_config": ATTN_NAME, "vision_config": "eager"}
    """
    from transformers import AttentionInterface
    AttentionInterface.register(ATTN_NAME, visual_boost_attention)
    try:
        from transformers import AttentionMaskInterface
        from transformers.masking_utils import eager_mask
        AttentionMaskInterface.register(ATTN_NAME, eager_mask)
    except ImportError:
        pass  # eski sürümler maske arayüzü olmadan da maskeyi geçirir
    return ATTN_NAME


@contextmanager
def visual_boost(alpha: float, layers: Iterable[int], image_positions: torch.Tensor,
                 decode_only: bool = True):
    _STATE.active, _STATE.alpha = True, float(alpha)
    _STATE.layers, _STATE.image_positions = set(layers), image_positions
    _STATE.decode_only = decode_only
    try:
        yield
    finally:
        _STATE.active = False
        _STATE.image_positions = None
