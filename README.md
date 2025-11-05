# Yüz Görsellerinden Otizm Tespiti

Bu proje, otizmli ve tipik gelişim gösteren çocukların yüz görsellerinden sınıflandırma yapabilen bir derin öğrenme modeli ve Grad-CAM tabanlı açıklanabilirlik arayüzü içerir. Eğitim bileşenleri PyTorch ile, görselleştirme arayüzü ise Streamlit ile geliştirilmiştir.

## Depo Yapısı
- `src/`: Eğitim, veri işleme ve çıkarım (inference) modülleri.
- `scripts/`: Komut satırından çalıştırılabilir yardımcı betikler (`train_model.py` vb.).
- `app/`: Streamlit tabanlı kullanıcı arayüzü.
- `requirements.txt`: Projenin Python bağımlılıkları.
- `artifacts/`: Eğitim çıktıları ve modeller (git tarafından yok sayılır).

## Kurulum
1. Python 3.10+ sürümü kurulu olduğundan emin olun.
2. Sanal ortam oluşturun ve etkinleştirin (Windows örneği):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## Veri Yerleşimi
Tüm veri dosyaları `data/` klasöründe gruplanmıştır:
```
data/
├── processed/
│   └── asd_faces/
│       ├── Train/
│       │   ├── autistic/
│       │   └── non_autistic/
│       ├── valid/
│       │   ├── autistic/
│       │   └── non_autistic/
│       └── Test/          # (varsa)
│           ├── autistic/
│           └── non_autistic/
├── raw_sources/          # Orijinal kaynak veri setleri (opsiyonel)
├── archives/             # Deduplicate arşivleri & eski sonuçlar
└── reports/              # Deduplicate raporu (git ignore altında)
```
Eğitim pipeline’ı `processed/asd_faces` dizinine ihtiyaç duyar. `.gitignore`, bu klasörlerin GitHub’a taşınmasını engeller.

### Veri Kaynağı
- Ana veri seti: [Autism Spectrum Detection from Kaggle + Zenodo](https://www.kaggle.com/datasets/ronakp004/autism-spectrum-detection-from-kaggle-zenodo) (Ronak Patel). Ham klasörler `data/raw_sources/` altında referans amaçlı saklanır, temizlenmiş sürüm `data/processed/asd_faces/` dizinine yerleştirilmiştir.

## Model ve Eğitim
- **Mimari:** ImageNet üzerinde önceden eğitilmiş ResNet-18 omurgası, son katmanda `dropout (p=0.3)` ve iki sınıflı linear sınıflandırıcı.
- **Veri büyüklüğü:** `Train=2423`, `valid=97`, `Test=281` görsel (yalnızca .jpg/.png vb. RGB dosyalar).
- **Ön işleme:** 224×224 orta kesim, eğitimde yatay çevirme, renk titreme, Gaussian blur ve rastgele silme (RandomErasing).
- **Optimizasyon:** AdamW (`lr=3e-4`, `weight_decay=1e-4`), sınıf ağırlıklı `CrossEntropyLoss`, `ReduceLROnPlateau` planlayıcı, CUDA varsa karma hassasiyet (AMP).

### Performans (Validasyon)
| Epoch | Doğruluk | Kayıp | Not |
|-------|----------|-------|-----|
| 21    | **85.57%** | 0.507 | Son eğitim oturumundaki en yüksek doğruluk (`artifacts/training_history.json`) |

> Not: Test kümesi değerlendirmesi isteğe bağlıdır; modeli `scripts/train_model.py` çıktısı ile yükleyip `training.evaluate` fonksiyonu üzerinden çalıştırabilirsiniz.

## Model Eğitimi
20 epoch boyunca eğitim yapmak için Windows Terminal üzerinden aşağıdaki komutu çalıştırabilirsiniz:
```powershell
python scripts/train_model.py --epochs 20
```
Ek parametreler:
- `--batch-size` ve `--num-workers` ile veri yükleme ayarlarını özelleştirebilirsiniz.
- `--freeze-backbone` ile yalnızca sınıflandırıcı katmanını eğitebilirsiniz.
- `--cpu` zorunlu CPU eğitimini etkinleştirir.

Eğitim sürecinde en iyi doğrulama sonucu `artifacts/resnet18_autism_classifier.pth` dosyasına kaydedilir ve epoch bazlı istatistikler `artifacts/training_history.json` dosyasında tutulur.

## Streamlit Arayüzü
Eğitilmiş modeli kullanarak görsel yüklemek ve Grad-CAM çıktısını görmek için:
```bash
streamlit run app/streamlit_app.py
```

Arayüz:
- Yüklenen görseli ve model sonuçlarını tek ekranda sunar.
- Sınıf olasılıklarını çubuk grafik ve tablo halinde gösterir.
- Grad-CAM ısı haritası ve bindirilmiş (overlay) çıktıları sabit çözünürlükte sunar.
