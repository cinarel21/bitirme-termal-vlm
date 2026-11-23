# Termal Görüntülerde Görsel Dil Modellerinin Muhakeme Sürecinin Mekanistik Yorumlanması

**Drone doğrulama üzerine bir uygulama**

Görsel dil modelleri (VLM) bir görüntüye bakıp adım adım düşünerek karar verebiliyor. Bu proje, termal (kızılötesi) görüntülerde bu düşünme sürecinin **içeride nasıl işlediğini** inceliyor ve bu bilgiyi düşünme maliyetini (token, süre) azaltmak için kullanıyor.

## Araştırma soruları

| | Soru | Yöntem |
|---|---|---|
| **AS1 — Ne zaman?** | Drone kararı modelin iç temsillerinde ne zaman oluşuyor? Düşünmenin kaçıncı tokenında, kaçıncı katmanında? | Logit lens, doğrusal probe |
| **AS2 — Nereye bakıyor?** | Düşünme ilerledikçe dikkatin görüntüye ve drone bölgesine giden payı nasıl değişiyor? Termal ile RGB arasında fark var mı? | Dikkat payı analizi (drone kutusu → görüntü-token ızgarası) |
| **AS3 — Neden?** | Karar gerçekten termal kanıttan mı, yoksa dil önyargısından mı geliyor? | Aktivasyon yamalama (drone'u silinmiş görüntü ile) |
| **AS4 — Kullanım** | Kararın erken oluştuğunu gösteren iç sinyal, gereksiz düşünmeyi kesmek için kullanılabilir mi? | Probe tabanlı erken durdurma, token tasarrufu |
| **AS5 — Düzeltme** *(bitirme)* | Termale özgü kısa muhakeme şablonu veya görsel dikkatin güçlendirilmesi temellendirmeyi düzeltir mi? | Şablon karşılaştırması, eğitimsiz dikkat müdahalesi |

## Klasör yapısı

```
src/tvlm/
  budget.py         kararlı minimum yeterli bütçe (B*), kararsızlık oranı
  geometry.py       kutu/kırpma hesapları, drone kutusunu token ızgarasına eşleme
  data_antiuav.py   Anti-UAV300 okuyucu, pozitif/negatif/silinmiş kırpma üretimi
  vlm.py            model yükleme, bütçe zorlama, logit lens, dikkat payı, aktivasyon yamalama
  probe.py          doğrusal probe ve iç sinyalle erken durdurma
scripts/
  01_inspect_dataset.py    veri setinin gerçek yapısını göster
  02_build_crops.py        kırpmaları üret (video düzeyinde bölme korunur)
  03_reason_and_measure.py ana deney: muhakeme + bütçe + logit lens + dikkat + probe özellikleri
  04_patching.py           nedensellik deneyi (aktivasyon yamalama)
  05_analyze.py            özet rapor: B*, karar katmanı, dikkat, probe, erken durdurma
tests/test_core.py         modelden bağımsız birim testleri (CPU)
docs/                      literatür, deney protokolü, Kaggle rehberi
```

## Hızlı başlangıç

Yerelde (model gerekmez):

```bash
pip install -r requirements.txt
pytest -q tests
```

GPU deneyleri Kaggle Notebook'ta çalışır: [docs/kaggle_calistirma.md](docs/kaggle_calistirma.md).

## Protokol özeti

- **Veri:** Anti-UAV300 (RGB + IR). Hazır video düzeyindeki train/val/test bölmesi korunur. Probe'lar train'de eğitilir, eşikler val'de seçilir, test en sona saklanır.
- **Örnekler:** Drone'u içeren bağlamlı kırpma (pozitif), aynı kareden drone ile örtüşmeyen kırpma (negatif) ve drone'u OpenCV inpaint ile silinmiş kırpma (yamalama için bozuk girdi). Kırpmalar 448×448'e ölçeklenir.
- **Model:** Qwen3-VL-2B-Thinking (ana), Qwen3-VL-2B-Instruct (düşünmesiz karşılaştırma). Açık ağırlıklı, Apache-2.0.
- **Bütçeler:** 0, 32, 64, 128, 256, 512 token. Muhakeme greedy decoding ile bir kez üretilir, her bütçede kesilip `DECISION:` ile karara zorlanır. Karar = `YES` ile `NO` logit farkının işareti.
- **B\*:** Kararın doğru olduğu ve sonraki hiçbir bütçede bozulmadığı en küçük bütçe.
- **Kayıt kuralı:** Sonuç tablolarına yalnızca koşulan deneylerin sayıları girer.

## Etik

Sistem yalnızca savunma amaçlı tespit ve doğrulama üretir. Görüntülerde kişisel veri işlenmez. Anti-UAV veri seti MIT lisanslıdır.
