# Artifacts

Bu klasör, test edilebilmesi için depoya eklenmiş en iyi model ağırlığını içerir.

- `resnet18_autism_classifier.pth`: Validation doğruluğu %85.57 olan ResNet-18 tabanlı sınıflandırıcı checkpoint'i. Streamlit arayüzü bu dosyayı otomatik olarak yükler. Eğitim betiği (`scripts/train_model.py`) aynı dosyanın üzerine yazabilir; yeni bir model kaydederseniz mevcut dosyayı da güncellemeyi unutmayın.

Diğer geçici eğitim çıktıları (`*.pth`, `*.pt`, `*.onnx`, log dosyaları vb.) `.gitignore` tarafından hariç tutulur. Yeni bir checkpoint’i depoya eklemek isterseniz ilgili dosyayı ve gerekiyorsa kısa bir açıklama içeren README notunu burada güncelleyin.
