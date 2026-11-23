# Kaggle'da çalıştırma rehberi

## Hazırlık

1. Kaggle'da **New Notebook** aç.
2. Sağ panel → **Add Data** → `anti-uav-dataset-300` ara ve ekle.
3. Sağ panel → **Accelerator** → `GPU T4 x2` veya `P100`.
4. Sağ panel → **Internet** → açık (model indirmek için).
5. Bu repoyu GitHub'a yükledikten sonra ilk hücrede çek:

```python
!git clone https://github.com/<kullanici-adin>/bitirme-termal-vlm.git
%cd bitirme-termal-vlm
!pip -q install -U "transformers>=4.57" accelerate
!pytest -q tests
```

## Adım 1 — Veri yapısını gör

```python
!ls /kaggle/input
!python scripts/01_inspect_dataset.py --root /kaggle/input/<klasör-adı>
```

Çıktıyı kontrol et:
- `train/`, `val/`, `test/` klasörleri var mı?
- Her video klasöründe `visible.json` / `infrared.json` (veya `IR_label.json`) var mı?
- `exist` ve `gt_rect` alanları var mı?
- Kareler `.mp4` mi, görüntü klasörü mü?

Yapı farklıysa `src/tvlm/data_antiuav.py` içindeki `MODALITY_JSON`, `MODALITY_VIDEO`, `MODALITY_FRAME_DIRS` sabitlerini güncelle. **Kopya kareleri karıştırıp video bilgisini kaybetmişse resmi sürümü kullan** (Google Drive bağlantısı resmi repoda).

## Deneme — her şey çalışıyor mu? (~10 dakika)

```python
DATA = "/kaggle/input/<klasör-adı>"
!python scripts/00_smoke_test.py --root {DATA} --modality ir --dry        # model yok, sadece veri
!python scripts/00_smoke_test.py --root {DATA} --modality ir --n 4        # modelle 4 görüntü
```

İkinci komut her görüntü için modelin düşüncesini, 0/32/64/128 token bütçelerindeki kararını, kararın oluştuğu katmanı ve görüntüye giden dikkat payını yazdırır. Bu çıktı hocaya gösterilebilecek ilk somut sonuçtur.

## Adım 2 — Kırpmalar

```python
DATA = "/kaggle/input/<klasör-adı>"
!python scripts/02_build_crops.py --root {DATA} --out /kaggle/working/crops --modality ir
!python scripts/02_build_crops.py --root {DATA} --out /kaggle/working/crops --modality rgb
```

## Adım 3 — Ana deney (önce küçük başla)

```python
M = "Qwen/Qwen3-VL-2B-Thinking"
!python scripts/03_reason_and_measure.py --manifest /kaggle/working/crops/manifest_ir.csv \
    --model {M} --prompt free --split val --limit 20 --out /kaggle/working/runs/ir_free_val
```

20 örnek sorunsuz geçerse limiti artır. Probe için train split'inden de çıktı gerekir:

```python
!python scripts/03_reason_and_measure.py --manifest /kaggle/working/crops/manifest_ir.csv \
    --model {M} --prompt free --split train --limit 300 --no-attention \
    --out /kaggle/working/runs/ir_free_train
```

Aynısını `--prompt thermal` ve RGB manifest'iyle tekrarla.

## Adım 4 — Nedensellik

```python
!python scripts/04_patching.py --manifest /kaggle/working/crops/manifest_ir.csv \
    --model {M} --split val --limit 50 --out /kaggle/working/runs/ir_patch
```

## Adım 5 — Rapor

```python
!python scripts/05_analyze.py --train /kaggle/working/runs/ir_free_train \
    --eval /kaggle/working/runs/ir_free_val --out /kaggle/working/report_ir_free
```

## Notlar

- Kaggle oturumu ~12 saatte kapanır. Betikler yarıda kalırsa aynı komutla devam eder, biten örnekleri atlar.
- `/kaggle/working` çıktısını oturum bitmeden **Save Version** ile kaydet veya bir Kaggle Dataset olarak sakla.
- Bellek yetmezse: `--no-attention` kullan veya `--max-new 256` ile başla.
- Her çalıştırmadan sonra GitHub'a küçük, açıklayıcı bir commit at (ör. "IR val için 03 deneyi: 200 örnek").
