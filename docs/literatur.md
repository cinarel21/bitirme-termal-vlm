# Literatür taraması

Tarama tarihi: Kasım 2026. Özetler arama sonuçları ve makale özetlerinden derlendi. **Bir makaleyi tezde veya makalede kaynak göstermeden önce tam metnini oku.** BibTeX kaydını arXiv sayfasındaki "Export BibTeX Citation" bağlantısından al; yazar adlarını elle yazma.

## Boşluk (bu çalışmanın konumu)

| Alan | Var olan | Eksik |
|---|---|---|
| VLM'lerin mekanistik yorumlaması | Logit lens, probe, aktivasyon yamalama; RGB görüntüler, nesne tanıma, sayma, halüsinasyon | Termal görüntü yok. Muhakeme (düşünme) süreci boyunca analiz az. |
| Düşünen VLM'lerde görsel temellendirme | Uzun düşünmede görsel dikkatin azaldığı ve halüsinasyonun arttığı gösterilmiş | Genel RGB benchmark'ları. Termal ve küçük hedef yok. Mekanizmanın nedenselliği tartışmalı. |
| İç sinyalle erken durdurma | Metin modellerinde gizli durumun sonucu erkenden taşıdığı gösterilmiş | VLM'de, görsel güvenlik görevinde ve termalde yok. |
| VLM ve termal görüntü | Benchmark'lar modellerin termalde başarısız olduğunu ve dil önyargısına dayandığını gösteriyor | Neden başarısız oldukları içeriden incelenmemiş. |

Taramada **termal görüntülerde VLM'lerin mekanistik yorumlamasını yapan bir çalışma bulunamadı.** Yenilik iddiası: "Bildiğimiz kadarıyla, termal görüntülerde görsel dil modellerinin muhakeme sürecini mekanistik olarak inceleyen ve bu bulguları muhakeme maliyetini azaltmak için kullanan ilk çalışmalardan biri."

---

## A. VLM'lerin mekanistik yorumlaması

1. **Towards Interpreting Visual Information Processing in Vision-Language Models** — ICLR 2025. [arXiv 2410.07149](https://arxiv.org/pdf/2410.07149) · [kod](https://github.com/clemneo/llava-interp)
   LLaVA-1.5'te logit lens: görüntü-token temsilleri katmanlar boyunca kelime dağarcığıyla hizalanıyor; nesne bilgisi nesnenin bulunduğu görüntü tokenlarında yerel. Bulgular Qwen2-VL-2B'de de denenmiş. **Yöntemimizin temel referansı.**
2. **Where Reliability Lives in Vision-Language Models: A Mechanistic Study of Attention, Hidden States, and Causal Circuits** — ICML 2026. [arXiv 2605.08200](https://arxiv.org/html/2605.08200v1)
   Logit lens + L1 probe + nöron ablasyonu. Dikkat haritasının keskinliği doğrulukla ilişkili değil; gizli durum probe'u POPE'de AUROC > 0.95. LLaVA'da "erken kilitlenme" gözlenmiş. **AS1 ve AS4 için doğrudan referans.**
3. **Arbitration Failure, Not Perceptual Blindness: How VLMs Resolve Visual-Linguistic Conflicts** — [arXiv 2604.09364](https://arxiv.org/pdf/2604.09364)
   Yalnızca son token yamalandığında çıktıların %90-98'i değişmiyor; tüm konumlar yamalanınca %60-84'ü değişiyor. Qwen2-VL'de görsel bilginin bir kısmı metin tokenlarına taşınıyor. **AS3'te yalnızca son tokenı değil görüntü tokenlarını yamalamamızın gerekçesi.**
4. **Causal Tracing of Object Representations in Large Vision Language Models** — [arXiv 2511.05923](https://arxiv.org/pdf/2511.05923)
   Görüntüye gürültü ekleyerek nedensel izleme; token kategorilerine göre katman katman geri yükleme.
5. **Counting Circuits: Mechanistic Interpretability of Visual Reasoning in LVLMs** — [arXiv 2603.18523](https://arxiv.org/pdf/2603.18523)
6. **To Sink or Not to Sink: Visual Information Pathways in LVLMs** — ICLR 2026. [arXiv 2510.08510](https://arxiv.org/pdf/2510.08510)
   Yüksek normlu "dikkat çukuru" görüntü tokenları. Dikkat payı ölçerken bu tokenları ayrıca raporlamak gerekebilir.
7. **Visual Representations inside the Language Model** — [arXiv 2510.04819](https://arxiv.org/pdf/2510.04819)
8. **What Do VLMs NOTICE? A Mechanistic Interpretability Pipeline for Noise-free Text-Image Corruption** — [arXiv 2406.16320](https://arxiv.org/html/2406.16320v1)
   Gürültüsüz bozma ile yamalama. Bizim inpaint ile drone silme yaklaşımımıza yakın.
9. **LVLM-Interpret** — [arXiv 2404.03118](https://arxiv.org/pdf/2404.03118) — araç.
10. **VLM-Lens** — [alphaXiv 2510.02292](https://www.alphaxiv.org/abs/2510.02292) — Qwen2-VL dahil 16 VLM'i destekleyen yorumlama çerçevesi.

## B. Düşünme ve görsel temellendirme

11. **More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models** — NeurIPS 2025. [arXiv 2505.21523](https://arxiv.org/abs/2505.21523)
    Düşünen modeller Qwen2.5-VL tabanlarına göre görüntüye daha az, talimata daha çok dikkat ediyor; uzun muhakemede halüsinasyon artıyor. RH-AUC ölçütü. **AS2'nin en yakın çalışması; biz termalde ve mekanistik/nedensel olarak test ediyoruz.**
12. **Qwen Look Again** — [arXiv 2505.23558](https://arxiv.org/pdf/2505.23558) — üretim uzadıkça görsel dikkatin azaldığına dair teorik ve deneysel sonuç.
13. **VisRef: Visual Refocusing while Thinking** — [arXiv 2603.00207](https://arxiv.org/pdf/2603.00207) — muhakeme sırasında görsel tokenları yeniden enjekte eden eğitimsiz yöntem. AS5 için aday.
14. **See It, Say It, Sorted** — [arXiv 2602.21497](https://arxiv.org/html/2602.21497)
15. **Addressing Overthinking in LVLMs via Gated Perception-Reasoning Optimization** — [arXiv 2601.04442](https://arxiv.org/pdf/2601.04442)
16. **Adaptive Attention Calibration** — [arXiv 2505.21472](https://arxiv.org/pdf/2505.21472) · **Conscious Gaze** — [arXiv 2512.05546](https://arxiv.org/html/2512.05546) · **SAVAA** — [arXiv 2602.13600](https://arxiv.org/pdf/2602.13600) · **VisFlow** — [arXiv 2506.12609](https://arxiv.org/pdf/2506.12609)
    Görsel dikkati güçlendiren eğitimsiz müdahaleler. Çoğu korelasyonel; AS5'te termalde denenebilir.

## C. İç sinyal ve muhakeme verimliliği

17. **Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification** — [arXiv 2504.05419](https://huggingface.co/papers/2504.05419)
    Gizli durum probe'u ile erken çıkış; tokenlarda %24'e varan azalma. **AS4'ün metin karşılığı.**
18. **Temporal Predictors of Outcome in Reasoning Language Models** — [arXiv 2511.14773](https://arxiv.org/html/2511.14773) — sonuç ilk birkaç tokendan sonra doğrusal olarak tahmin edilebiliyor.
19. **Token Budget Saturation and Mechanistic Early-Detection of Reasoning** — [arXiv 2607.21433](https://arxiv.org/html/2607.21433v1)
20. **SpecExit** — [arXiv 2509.24248](https://arxiv.org/pdf/2509.24248) — probe'un ek gecikme getirdiği ve görevler arası zayıf genellediği eleştirisi. Tartışma bölümünde ele alınmalı.
21. **The Zero-Step Thinking** — [arXiv 2510.19176](https://arxiv.org/pdf/2510.19176)
22. Davranışsal erken çıkış ve bütçe: **DEER** [2504.15895](https://arxiv.org/pdf/2504.15895), **FlashThink** [2505.13949](https://arxiv.org/pdf/2505.13949), **DTSR** [2604.06787](https://arxiv.org/pdf/2604.06787), **AVIS** (VLM'de zorluğa göre bütçe) [2606.11576](https://arxiv.org/abs/2606.11576), derleme **Don't Overthink It** [2508.02120](https://arxiv.org/pdf/2508.02120).

## D. Muhakeme sadakati (CoT faithfulness)

23. **On the Faithfulness of Visual Thinking** — [arXiv 2510.23482](https://arxiv.org/pdf/2510.23482)
24. **SPD-Faith Bench** — [arXiv 2602.07833](https://arxiv.org/pdf/2602.07833) — "algısal körlük" ve "algı-muhakeme ayrışması" hata türleri.
25. **Evaluating Reasoning Faithfulness in Medical VLMs using Multimodal Perturbations** — [arXiv 2510.11196](https://arxiv.org/abs/2510.11196)

## E. VLM'ler ve termal görüntü

26. **ThermEval** — [arXiv 2602.14989](https://arxiv.org/html/2602.14989v1) — 25 VLM termalde sıcaklık temelli muhakemede başarısız, renk haritası değişince bozuluyor, dil önyargısına düşüyor; prompt ve SFT az kazanç sağlıyor. **Motivasyonun ana kaynağı.**
27. **RGB-Th-Bench** — [arXiv 2503.19654](https://arxiv.org/html/2503.19654v2)
28. **Right Answer, Wrong Heat** — [arXiv 2608.09145](https://arxiv.org/html/2608.09145v1) — doğru IR cevaplarının önemli kısmı termal kanıta dayanmıyor; çok ajanlı eğitimsiz düzeltme (maliyet ölçülmemiş).
29. **IF-Bench** — [arXiv 2512.09663](https://arxiv.org/html/2512.09663)
30. **Thermo-VL** — [arXiv 2605.21882](https://arxiv.org/pdf/2605.21882) — Molmo-7B'ye eğitilebilir termal kodlayıcı ekliyor. İç işleyişi incelemiyor.

## F. Küçük nesne algısı

31. **MLLMs Know Where to Look (ViCrop)** — ICLR 2025. [arXiv 2502.17422](https://arxiv.org/pdf/2502.17422) — nesne küçüldükçe doğruluk düşüyor; modelin kendi dikkatiyle kırpma doğruluğu artırıyor. **Model "nereye bakacağını biliyor ama göremiyor" bulgusu bizim drone-bölgesi dikkat ölçümümüzle karşılaştırılmalı.**

## H. Düşünmeye müdahale (steering)

36. **Exploring and Exploiting the Inherent Efficiency within Large Reasoning Models (Efficiency Steering)** — [arXiv 2506.15647](https://arxiv.org/pdf/2506.15647) — verimli ve uzun muhakeme aktivasyonlarını ayıran yön; yedi modelde muhakemeyi kısaltıp doğruluğu korumuş. **Durma yönümüzün temel referansı.**
37. **Mitigating Overthinking via Manifold Steering** — [arXiv 2505.22411](https://arxiv.org/html/2505.22411v1)
38. **The Halt Vector: Internalizing a Causal Steering Intervention for Efficient Reasoning** — [arXiv 2608.28859](https://arxiv.org/pdf/2608.28859)
39. **ReBalance: Efficient Reasoning with Balanced Thinking** — [arXiv 2603.12372](https://arxiv.org/abs/2603.12372) — sabit tek yönlü müdahalenin yetersiz düşünmeye yol açabileceği uyarısı; güvene göre ayarlanan güç. Tartışmada ele alınmalı.
40. **EasySteer** — [arXiv 2509.25175](https://arxiv.org/html/2509.25175v1) — steering çerçevesi; SEAL yöntemini de içeriyor.
41. **The Ebb and Flow of Multimodal Focus: Visual Relay Windows** — [arXiv 2607.11436](https://arxiv.org/html/2607.11436) — muhakeme sırasında görsel dikkatin zamanlaması.
42. **MAST: Medoid-Anchored Subspace Steering for Visual Grounding** — [arXiv 2611.15168](https://arxiv.org/pdf/2611.15168)
43. **Viscot-steer** — [arXiv 2611.11209](https://arxiv.org/pdf/2611.11209) — görsel düşünce zincirinde steering (yalnızca başlık doğrulandı, tam metin okunmalı).
44. Görsel dikkat güçlendirme uygulaması: Hugging Face [AttentionInterface](https://huggingface.co/docs/transformers/attention_interface) belgesi.

## G. Veri ve model

32. **Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking** — [arXiv 2101.08466](https://arxiv.org/abs/2101.08466) · [repo](https://github.com/ZhaoJ9014/Anti-UAV) (MIT). 300'den fazla RGB-IR video çifti.
33. **Anti-UAV410** — IEEE TPAMI 2023. [ACM](https://dl.acm.org/doi/abs/10.1109/TPAMI.2023.3335338) — yalnızca IR.
34. **Securing the Skies: A Comprehensive Survey on Anti-UAV Methods** — CVPRW 2025. [arXiv 2504.11967](https://arxiv.org/pdf/2504.11967)
35. **Qwen3-VL Technical Report** — [arXiv 2511.21631](https://arxiv.org/pdf/2511.21631) · [Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) (Apache-2.0)
