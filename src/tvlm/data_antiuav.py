"""Anti-UAV300 okuyucu ve kırpma (crop) üretici.

Resmi Anti-UAV300 yapısı (README'ye göre): her video klasöründe RGB ve IR için ayrı
etiket JSON'u bulunur (ör. visible.json / infrared.json veya IR_label.json). JSON içinde
`exist` (kare başına 0/1) ve `gt_rect` (kare başına [x, y, w, h]) alanları vardır.

Kaggle/Hugging Face kopyaları farklı paketlenmiş olabilir. Bu yüzden okuyucu:
  * video klasörünü JSON dosyasının bulunduğu klasör olarak kabul eder,
  * kareleri ya aynı adlı .mp4 dosyasından ya da bir kare klasöründen okur.
`scripts/01_inspect_dataset.py` gerçek yapıyı gösterir; uyuşmazlık varsa burası düzeltilir.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .geometry import Box, box_in_crop, context_crop, random_negative_crop, size_bin

MODALITY_JSON = {
    "rgb": ("visible.json", "RGB_label.json"),
    "ir": ("infrared.json", "IR_label.json"),
}
MODALITY_VIDEO = {
    "rgb": ("visible.mp4", "RGB.mp4"),
    "ir": ("infrared.mp4", "IR.mp4"),
}
MODALITY_FRAME_DIRS = {
    "rgb": ("visible", "RGB", "rgb"),
    "ir": ("infrared", "IR", "ir"),
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class VideoAnno:
    split: str
    video: str
    modality: str
    json_path: Path
    exist: list[int]
    rects: list[Optional[Box]]

    def __len__(self) -> int:
        return len(self.exist)


def _parse_rect(r) -> Optional[Box]:
    if r is None or len(r) < 4:
        return None
    x, y, w, h = (float(v) for v in r[:4])
    if w <= 1 or h <= 1:
        return None
    return Box(x, y, w, h)


def load_anno(json_path: Path, split: str, modality: str) -> VideoAnno:
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    rects_raw = d.get("gt_rect", [])
    exist = d.get("exist")
    if exist is None:  # bazı kopyalarda exist yok: kutu varsa 1 say
        exist = [1 if _parse_rect(r) else 0 for r in rects_raw]
    rects = [_parse_rect(r) if e else None for e, r in zip(exist, rects_raw)]
    return VideoAnno(split, json_path.parent.name, modality, json_path,
                     [int(e) for e in exist], rects)


def find_videos(root: Path, modality: str) -> Iterator[VideoAnno]:
    """root/{train,val,test}/<video>/<label>.json yapısını tarar."""
    names = MODALITY_JSON[modality]
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        split = split_dir.name.lower()
        if split not in {"train", "val", "validation", "test", "test-dev"}:
            continue
        split = {"validation": "val", "test-dev": "test"}.get(split, split)
        for vid_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for n in names:
                jp = vid_dir / n
                if jp.exists():
                    yield load_anno(jp, split, modality)
                    break


class FrameReader:
    """Kareleri .mp4'ten veya kare klasöründen okur (BGR -> RGB numpy)."""

    def __init__(self, vid_dir: Path, modality: str):
        import cv2  # yalnızca gerektiğinde
        self.cv2 = cv2
        self.cap = None
        self.frames: list[Path] = []
        for n in MODALITY_VIDEO[modality]:
            if (vid_dir / n).exists():
                self.cap = cv2.VideoCapture(str(vid_dir / n))
                break
        if self.cap is None:
            for n in MODALITY_FRAME_DIRS[modality]:
                d = vid_dir / n
                if d.is_dir():
                    self.frames = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
                    break
        if self.cap is None and not self.frames:
            raise FileNotFoundError(f"{vid_dir} içinde {modality} karesi bulunamadı")

    def read(self, idx: int) -> np.ndarray:
        if self.cap is not None:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self.cap.read()
            if not ok:
                raise IndexError(idx)
        else:
            bgr = self.cv2.imread(str(self.frames[idx]))
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


@dataclass
class CropRow:
    crop_id: str
    split: str
    video: str
    modality: str
    frame: int
    label: int          # 1 = drone, 0 = drone değil (arka plan)
    size_bin: str       # pozitifler için kutu boyutu, negatifler için eşlendiği pozitifinki
    bx: float           # kırpma içindeki drone kutusu (negatiflerde -1)
    by: float
    bw: float
    bh: float
    path: str
    corrupt_path: str   # drone'u silinmiş (inpaint) karşılığı; negatiflerde boş


def inpaint_box(img: np.ndarray, box: Box, pad: int = 3) -> np.ndarray:
    """Drone bölgesini OpenCV inpaint ile siler: aktivasyon yamalama için 'bozuk' girdi."""
    import cv2
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    x0, y0 = max(int(box.x) - pad, 0), max(int(box.y) - pad, 0)
    x1 = min(int(box.x + box.w) + pad, img.shape[1])
    y1 = min(int(box.y + box.h) + pad, img.shape[0])
    mask[y0:y1, x0:x1] = 255
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def build_crops(root: Path, out_dir: Path, modality: str,
                frames_per_video: int = 6, seed: int = 0,
                context: float = 4.0, min_side: int = 112) -> list[CropRow]:
    """Her videodan eşit aralıklı kareler seçip pozitif + eşleşmiş negatif kırpmalar üretir.

    Video düzeyinde bölme korunur: kırpmalar videonun kendi split'ini taşır.
    """
    from PIL import Image
    rng = random.Random(seed)
    rows: list[CropRow] = []
    for va in find_videos(root, modality):
        valid = [i for i, (e, r) in enumerate(zip(va.exist, va.rects)) if e and r]
        if not valid:
            continue
        step = max(1, len(valid) // frames_per_video)
        picked = valid[::step][:frames_per_video]
        reader = FrameReader(va.json_path.parent, modality)
        try:
            for fi in picked:
                img = reader.read(fi)
                H, W = img.shape[:2]
                box = va.rects[fi]
                crop = context_crop(box, W, H, context, min_side)
                inner = box_in_crop(box, crop)
                sb = size_bin(box)
                cid = f"{va.split}_{va.video}_{modality}_{fi:05d}"
                sub = img[int(crop.y):int(crop.y + crop.h), int(crop.x):int(crop.x + crop.w)]
                pos_dir = out_dir / modality / va.split
                pos_dir.mkdir(parents=True, exist_ok=True)
                p_path = pos_dir / f"{cid}_pos.png"
                c_path = pos_dir / f"{cid}_corrupt.png"
                Image.fromarray(sub).save(p_path)
                Image.fromarray(inpaint_box(sub, inner)).save(c_path)
                rows.append(CropRow(cid + "_pos", va.split, va.video, modality, fi, 1, sb,
                                    inner.x, inner.y, inner.w, inner.h,
                                    str(p_path), str(c_path)))
                neg = random_negative_crop(int(crop.w), W, H, box, rng)
                if neg is not None:
                    nsub = img[int(neg.y):int(neg.y + neg.h), int(neg.x):int(neg.x + neg.w)]
                    n_path = pos_dir / f"{cid}_neg.png"
                    Image.fromarray(nsub).save(n_path)
                    rows.append(CropRow(cid + "_neg", va.split, va.video, modality, fi, 0, sb,
                                        -1, -1, -1, -1, str(n_path), ""))
        finally:
            reader.close()
    return rows


def write_manifest(rows: list[CropRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def read_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
