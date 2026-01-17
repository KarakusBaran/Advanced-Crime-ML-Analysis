# 🕵️‍♂️ Cinayet Çözülme Durumu Tahmini (Homicide Clearance Prediction)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Kütüphane-Scikit--Learn-orange)
![Machine Learning](https://img.shields.io/badge/Tür-Sınıflandırma-green)
![Status](https://img.shields.io/badge/Durum-Tamamlandı-success)

## 📌 Proje Özeti
Bu proje, ABD'deki olay yeri verilerini analizerek bir cinayet vakasının çözülüp çözülemeyeceğini (**Solved vs. Unsolved**) tahmin etmeyi amaçlamaktadır. **Murder Accountability Project (MAP)** veri seti kullanılarak (~630.000 satır), kolluk kuvvetlerinin kaynak planlamasına yardımcı olabilecek bir makine öğrenmesi hattı (pipeline) geliştirilmiştir.

Proje, **Doğrusal Modellerin** (Linear Models) sınırlarını test ederek, suç verisinin karmaşıklığını çözmek için **Stacking Ensemble (Topluluk Öğrenmesi)** mimarisinin üstünlüğünü ortaya koyan karşılaştırmalı bir çalışmadır.

## 🚀 Öne Çıkan Stratejiler & Özellikler
* **Büyük Veri Yönetimi:** 600.000'den fazla gerçek hayat verisi (Real-world data) başarıyla işlendi ve analiz edildi.
* **Veri Sızıntısı (Leakage) Önlemi:** Modelin "cevap anahtarını görmemesi" için Fail (Perpetrator) ile ilgili yaş, ırk, cinsiyet gibi sütunlar titizlikle temizlendi.
* **Dengesiz Veri (Imbalanced Data) Yönetimi:** %70-%30 dengesizliği yönetmek için sentetik veri (SMOTE) üretmek yerine, **Maliyete Duyarlı Öğrenme** (`class_weight='balanced'`) yöntemi tercih edildi.
* **Boyut İndirgeme:** "Boyut Laneti"ni (Curse of Dimensionality) önlemek için One-Hot Encoding yerine **Label Encoding** kullanılarak özellik sayısı 24'ten 14'e düşürüldü.
* **Stacking Mimarisi:** Ağaç tabanlı modeller ve Sinir Ağları birleştirilerek 2 katmanlı hibrit bir yapı kuruldu.

## 📊 Veri Seti & Ön İşleme
Veri seti, 1976-2020 yılları arasındaki ABD cinayet raporlarını içerir.
* **Hedef Değişken (Target):** `Crime Solved` (1: Çözüldü, 0: Çözülemedi).
* **Öznitelik Mühendisliği (Feature Engineering):**
    * **Yaş Gruplandırma:** Sayısal yaş verileri sosyolojik gruplara (*Çocuk, Genç, Yetişkin, Yaşlı*) ayrılarak modelin gürültüden (noise) etkilenmesi engellendi.
    * **Korelasyon Analizi:** Çoklu bağlantı (Multicollinearity) sorunu yaratan sütunlar elendi.

## 🧠 Model Mimarisi

### 1. Aşama: Karmaşıklık Testi (Lineer Yaklaşım)
Verinin basit bir düzlemle ayrılıp ayrılamayacağını test etmek için hızlı lineer modeller kullanıldı.
* **Modeller:** `LinearSVC`, `SGDClassifier` (log_loss ile)
* **Sonuç:** ROC-AUC ~0.61
* **Çıkarım:** Düşük skor, cinayet verisinin **Doğrusal Olmadığını (Non-Linear)** ve basit sınırlarla ayrılamayacağını kanıtladı.

### 2. Aşama: Çözüm (Stacking Ensemble)
Karmaşık desenleri yakalamak için Lojistik Regresyon meta-öğrenicisine sahip bir Stacking Classifier kuruldu.

| Katman | Kullanılan Modeller | Görevi |
| :--- | :--- | :--- |
| **Katman 0 (Uzmanlar)** | `ExtraTreesClassifier` | Varyansı ve Aşırı Öğrenmeyi (Overfitting) düşürür |
| | `HistGradientBoosting` | Hatayı (Bias) optimize eder, büyük veride hızlıdır |
| | `MLPClassifier` (YSA) | Doğrusal olmayan karmaşık ilişkileri yakalar |
| **Katman 1 (Yönetici)** | `LogisticRegression` | Alt modellerin tahminlerini ağırlıklandırarak nihai kararı verir |

## 📈 Sonuçlar

Veri dengesiz olduğu için yanıltıcı olabilen "Accuracy" yerine, modelin ayırt etme gücünü gösteren **ROC-AUC** ve **F1-Score** metriklerine odaklanıldı.

| Model | ROC-AUC Skoru | Yorum |
| :--- | :--- | :--- |
| **LinearSVC / SGD** | 0.61 | Underfitting (Veri modele göre fazla karmaşık) |
| **Stacking Classifier** | **0.74** | **En İyi Performans / Başarılı Ayrım** |

> **Not:** 0.74 ROC-AUC skoru ile modelimiz, çözülebilir ve çözülemez vakaları %74 başarı oranıyla birbirinden ayırt edebilmektedir.

## 🛠️ Kurulum ve Kullanım

1.  **Projeyi klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADINIZ/homicide-prediction.git](https://github.com/KULLANICI_ADINIZ/homicide-prediction.git)
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Jupyter Notebook'ları çalıştırın:**
    * Lineer analiz ve veri temizliği için: `Baran_Karakus_Linear_SGD.ipynb`
    * Final model ve değerlendirme için: `Group_Project_Stacking.ipynb`

## 📂 Proje Yapısı
