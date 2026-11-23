"""Kutu, kırpma ve görüntü-token ızgarası hesapları. Modelden bağımsız, CPU'da test edilir."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Kutu boyutu sınıfları (orijinal karedeki kutu alanının karekökü, piksel).
SIZE_BINS: tuple[tuple[str, float], ...] = (
    ("tiny", 16.0),
    ("small", 32.0),
    ("medium", 96.0),
    ("large", math.inf),
)


@dataclass(frozen=True)
class Box:
    """Sol-üst köşe + genişlik/yükseklik (Anti-UAV gt_rect formatı)."""
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def size_bin(box: Box) -> str:
    s = math.sqrt(box.area)
    for name, upper in SIZE_BINS:
        if s < upper:
            return name
    return SIZE_BINS[-1][0]


def context_crop(box: Box, img_w: int, img_h: int,
                 context: float = 4.0, min_side: int = 112) -> Box:
    """Kutuyu merkeze alan, bağlam içeren kare kırpma. Görüntü sınırına kaydırılır."""
    side = max(float(min_side), context * max(box.w, box.h))
    side = min(side, float(min(img_w, img_h)))
    x0 = min(max(box.cx - side / 2, 0.0), img_w - side)
    y0 = min(max(box.cy - side / 2, 0.0), img_h - side)
    return Box(round(x0), round(y0), round(side), round(side))


def box_in_crop(box: Box, crop: Box) -> Box:
    return Box(box.x - crop.x, box.y - crop.y, box.w, box.h)


def random_negative_crop(side: int, img_w: int, img_h: int,
                         avoid: Optional[Box], rng: random.Random,
                         max_tries: int = 50) -> Optional[Box]:
    """Drone kutusuyla hiç örtüşmeyen (IoU=0) rastgele kare kırpma."""
    side = int(min(side, img_w, img_h))
    for _ in range(max_tries):
        x0 = rng.randint(0, img_w - side)
        y0 = rng.randint(0, img_h - side)
        cand = Box(x0, y0, side, side)
        if avoid is None or iou(cand, avoid) == 0.0:
            return cand
    return None


def token_grid_mask(box_in_img: Box, orig_w: int, orig_h: int,
                    resized_w: int, resized_h: int,
                    pixels_per_token: int) -> np.ndarray:
    """Kırpma içindeki drone kutusunu, VLM'in görüntü-token ızgarasına eşler.

    Qwen-VL ailesinde görüntü önce (resized_w, resized_h) boyutuna getirilir,
    sonra her token `pixels_per_token` x `pixels_per_token` piksellik bir bölgeyi
    temsil eder (Qwen2.5-VL: 14*2=28, Qwen3-VL: 16*2=32). Tokenlar satır sırasıyla dizilir.

    Dönen maske (grid_h, grid_w) boyutunda; True = drone'a değen token.
    """
    gw = resized_w // pixels_per_token
    gh = resized_h // pixels_per_token
    sx = resized_w / orig_w
    sy = resized_h / orig_h
    x0 = box_in_img.x * sx
    y0 = box_in_img.y * sy
    x1 = (box_in_img.x + box_in_img.w) * sx
    y1 = (box_in_img.y + box_in_img.h) * sy
    c0 = max(0, int(math.floor(x0 / pixels_per_token)))
    r0 = max(0, int(math.floor(y0 / pixels_per_token)))
    c1 = min(gw - 1, int(math.ceil(x1 / pixels_per_token)) - 1)
    r1 = min(gh - 1, int(math.ceil(y1 / pixels_per_token)) - 1)
    mask = np.zeros((gh, gw), dtype=bool)
    if c1 >= c0 and r1 >= r0:
        mask[r0:r1 + 1, c0:c1 + 1] = True
    return mask
