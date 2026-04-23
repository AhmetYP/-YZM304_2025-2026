# YZM304 Derin Öğrenme Projesi

## Banknote Authentication - MLP Modelleri Karşılaştırması

---

## Introduction

Bu projede, Banknote Authentication veri seti kullanılarak sahte ve gerçek banknotların sınıflandırılması problemi ele alınmıştır. Amaç, laboratuvar çalışmasında geliştirilen tek gizli katmanlı temel yapay sinir ağı modelini geliştirerek farklı mimariler ve farklı kütüphaneler (NumPy, scikit-learn, PyTorch) ile performans karşılaştırması yapmaktır.

Bu kapsamda, model karmaşıklığının (katman sayısı ve nöron sayısı) performansa etkisi incelenmiş, ayrıca overfitting ve underfitting davranışları analiz edilmiştir.

---

## Methods

### Veri Seti

Kullanılan veri seti 1372 örnekten ve 4 sayısal özellikten oluşmaktadır:

* variance
* skewness
* curtosis
* entropy

Hedef değişken ikili sınıflandırma (0 ve 1) şeklindedir. Veri setinde eksik değer bulunmamaktadır ve sınıf dağılımı dengelidir.

### Veri Ön İşleme

* Veri train (%60), validation (%20) ve test (%20) olarak bölünmüştür
* Stratified split kullanılmıştır
* Özellikler `StandardScaler` ile ölçeklendirilmiştir (yalnızca train set üzerinde fit edilmiştir)

---

### Model 1: NumPy - Tek Gizli Katman (Baseline)

* 1 hidden layer (6 nöron)
* Sigmoid aktivasyon
* Binary cross-entropy loss
* SGD optimizasyon

Bu model başlangıç (baseline) performansı için kullanılmıştır.

---

### Model 2: NumPy - Geliştirilmiş Tek Katman

* 1 hidden layer (16 ve 32 nöron denendi)
* Daha fazla nöron ile model kapasitesi artırıldı

Sonuç:
16 nöronlu model en iyi performansı vermiştir.

---

### Model 3: NumPy - İki Gizli Katman

İlk deneme:

* Sigmoid aktivasyon → başarısız (model çoğunluk sınıfa çöktü)

İyileştirme:

* ReLU aktivasyon
* He initialization (√(2/n))
* Daha düşük learning rate
* Daha fazla epoch

Sonuç:

* 2 hidden layer (16, 8) en iyi NumPy modeli olmuştur

---

### Model 4: scikit-learn MLPClassifier

* hidden_layer_sizes = (16, 8)
* activation = ReLU
* optimizer = SGD
* aynı veri bölmesi kullanıldı

---

### Model 5: PyTorch MLP

İlk deneme:

* SGD + Sigmoid → başarısız

İyileştirme:

* BCEWithLogitsLoss
* Adam optimizer
* ReLU aktivasyon

---

### Değerlendirme Metrikleri

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Ayrıca:

* Train / Validation loss eğrileri incelenmiştir

---

## Results

### Model Karşılaştırma Tablosu

| Model          | Yapı            | Train Acc | Val Acc | Test Acc | Precision | Recall |     F1 |
| -------------- | --------------- | --------: | ------: | -------: | --------: | -----: | -----: |
| NumPy Baseline | 1 hidden (6)    |    0.9696 |  0.9818 |   0.9673 |    0.9380 | 0.9918 | 0.9641 |
| NumPy Improved | 1 hidden (16)   |    0.9745 |  0.9855 |   0.9709 |    0.9385 | 1.0000 | 0.9683 |
| NumPy Deep     | 2 hidden (16,8) |    0.9939 |  0.9964 |   0.9964 |    0.9919 | 1.0000 | 0.9959 |
| sklearn MLP    | (16,8)          |    1.0000 |  1.0000 |   1.0000 |    1.0000 | 1.0000 | 1.0000 |
| PyTorch MLP    | (16,8)          |    1.0000 |  1.0000 |   1.0000 |    1.0000 | 1.0000 | 1.0000 |

---

### Gözlemler

* Tek katmanlı model zaten yüksek performans göstermektedir
* Nöron sayısı artırıldığında performans artmıştır
* İki katmanlı model uygun yapı ile en iyi NumPy sonucu vermiştir
* sklearn ve PyTorch modelleri test verisini tamamen doğru sınıflandırmıştır

---

### Loss Analizi

Train ve validation loss eğrileri birlikte incelenmiştir. Eğriler paralel şekilde azalmış ve belirgin bir ayrışma görülmemiştir. Bu durum modelin overfitting yapmadığını ve iyi genelleme yaptığını göstermektedir.

---

## Discussion

Bu projede model karmaşıklığının performansa etkisi açık şekilde gözlemlenmiştir.

* Basit model güçlü sonuçlar vermiştir
* Daha fazla nöron performansı artırmıştır
* İlk derin model başarısız olmuş, bu da uygun aktivasyon ve initialization'ın önemini göstermiştir
* ReLU ve doğru ağırlık başlatma ile derin model ciddi şekilde iyileşmiştir

Ayrıca:

* sklearn ve PyTorch modelleri optimize edilmiş implementasyonları sayesinde en yüksek performansı sağlamıştır
* veri setinin yüksek derecede ayrılabilir olduğu gözlemlenmiştir

### Sonuç

En iyi genel performans:

* sklearn ve PyTorch modelleri (%100 doğruluk)

En iyi manuel model:

* NumPy 2 hidden layer (16, 8)

---

## Future Work

* Farklı optimizer’lar (Adam, RMSProp)
* Dropout ve regularization
* Daha büyük veri setleri
* Cross-validation

---

## Requirements

```txt
numpy
pandas
matplotlib
scikit-learn
seaborn
torch
```

---
