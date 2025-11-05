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

#### İndirme ve Yerleştirme Adımları
1. Veri setini Kaggle üzerinden indirin (veya Kaggle CLI kullanıyorsanız):
   ```bash
   kaggle datasets download -d ronakp004/autism-spectrum-detection-from-kaggle-zenodo
   ```
2. İndirilen `.zip` dosyasını projenin kök dizinine açın.
3. Aşağıdaki yapıdaki klasörleri oluşturun ve veri setindeki `Train/`, `valid/` ve varsa `Test/` dizinlerini birebir buraya taşıyın:
   ```
   data/
   └── processed/
       └── asd_faces/
           ├── Train/
           ├── valid/
           └── Test/    # veri setinde mevcutsa
   ```
4. İsteğe bağlı olarak ham arşivi `data/raw_sources/` altında saklayabilirsiniz; `.gitignore` bu klasörü depodan hariç tutar.

### Hazır Checkpoint
- Depoya **`artifacts/resnet18_autism_classifier.pth`** dosyası dahil edilmiştir. Bu dosya doğrulama setinde %85.57 doğruluk elde eden en iyi modeli içerir ve Streamlit arayüzü varsayılan olarak bu ağırlığı yükler.
- Eğer modeli yeniden eğitirseniz, aynı dosya üzerine yazılacaktır. Git geçmişine farklı bir versiyon eklemek isterseniz, mevcut dosyayı yedeklemeyi unutmayın.

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

Eğitim sürecinde en iyi doğrulama sonucu `artifacts/resnet18_autism_classifier.pth` dosyasına kaydedilir ve epoch bazlı istatistikler `artifacts/training_history.json` dosyasında tutulur. Streamlit arayüzünün çalışabilmesi için bu ağırlık dosyasının mevcut olması gerekir; aksi hâlde uygulama gerekli bilgilendirmeyi gösterecektir.

## Streamlit Arayüzü
Eğitilmiş modeli kullanarak görsel yüklemek ve Grad-CAM çıktısını görmek için:
```bash
streamlit run app/streamlit_app.py
```

Arayüz:
- Yüklenen görseli ve model sonuçlarını tek ekranda sunar.
- Sınıf olasılıklarını çubuk grafik ve tablo halinde gösterir.
- Grad-CAM ısı haritası ve bindirilmiş (overlay) çıktıları sabit çözünürlükte sunar.

> **Not:** İlk kez projeyi klonlayan kullanıcılar, `python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt` komutlarıyla ortamı hazırlayıp doğrudan Streamlit arayüzünü çalıştırabilir; ek eğitim gerektirmez.

