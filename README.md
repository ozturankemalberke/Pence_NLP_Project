# 🦅 Pençe — Türkçe Toksik Dil Tespiti ve Suç Kategorisi Sınıflandırması

**Pençe**, Türkçe sosyal medya metinlerindeki saldırgan/toksik dili tespit eden ve bunları
Türk Ceza Kanunu (TCK) bağlamındaki suç kategorilerine ayıran bir Doğal Dil İşleme (NLP)
projesidir. Proje, aynı problemi **iki farklı yaklaşımla** çözen iki "motor" içerir ve bunları
bilimsel olarak karşılaştırır.

> Bu proje bir **Doğal Dil İşleme dersi** kapsamında geliştirilmiştir.

---

## Problem

Bir metni şu altı kategoriden birine sınıflandırmak:

| Kategori | İlgili TCK Maddesi |
|---|---|
| Tehdit | TCK 106 |
| Şantaj | TCK 107 |
| Cinsel Taciz | TCK 105 |
| Hakaret | TCK 125 |
| Suç Övgüsü | TCK 215 |
| Temiz | — |

TCK madde/ceza eşleştirmesi sabit bir tablodan gelir; modeller yalnızca **kategori** tahmin eder.

---

## İki Motor

- **Motor A — Klasik Makine Öğrenmesi**: TF-IDF özellik çıkarımı + çok katmanlı yapay sinir
  ağı (MLP). Dengesiz veri sorununu çözmek için nadir sınıflarda oversampling kullanılır.
  Geliştirilmiş sürüm (`v2`) kelime bigramları ve karakter n-gramları ekler.
- **Motor B — Büyük Dil Modeli (LLM)**: Anthropic Claude ile aynı test setini sınıflandırır.
  Her iki motor da **birebir aynı test verisi** üzerinde değerlendirilerek adil kıyas sağlanır.

Ayrıca bir **baseline** (TF-IDF + Lojistik Regresyon) kıyas çıtası olarak eğitilir.

---

## Veri Setleri

Proje, iki açık Türkçe veri setini temel alır ve bunları LLM ile yeniden etiketleyerek
özgün, çok sınıflı bir veri seti üretir:

- [`Overfit-GM/turkish-toxic-language`](https://huggingface.co/datasets/Overfit-GM/turkish-toxic-language)
- [`Toygar/turkish-offensive-language-detection`](https://huggingface.co/datasets/Toygar/turkish-offensive-language-detection)

---

## Proje Akışı (Pipeline)

Scriptler numara sırasıyla çalıştırılacak şekilde tasarlanmıştır:

| Script | Açıklama |
|---|---|
| `01_veri_inceleme.py` | HuggingFace veri setlerini indirip yapılarını inceler |
| `02_veri_birlestirme.py` | İki seti birleştirip temiz, tekil bir havuz oluşturur → `birlesik_havuz.csv` |
| `03_api_test.py` | Anthropic API bağlantısını test eder |
| `04_etiketleme.py` | Claude ile çok sınıflı nihai veri setini üretir → `etiketli_veri.csv` |
| `05_veri_hazirlama.py` | Etiketli veriyi eğitim/test olarak böler |
| `06_baseline.py` | TF-IDF + Lojistik Regresyon kıyas modeli |
| `07_motor_a_lstm.py` | Motor A — MLP + oversampling |
| `08_motor_b_llm.py` | Motor B — Claude ile sınıflandırma → `motor_b_sonuclar.csv` |
| `09_karsilastirma.py` | Üç modelin bilimsel karşılaştırması (F1, doğruluk, confusion matrix) |
| `10_arayuz.py` | Streamlit web arayüzü |
| `11_motor_a_v2.py` | Motor A geliştirilmiş sürüm (bigram + karakter n-gram) |

---

## Kurulum

```bash
# 1. Depoyu klonlayın
git clone <REPO_URL>
cd NLP_Proje

# 2. Sanal ortam oluşturup etkinleştirin
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### API Anahtarı

Motor B ve etiketleme adımları için bir Anthropic API anahtarı gerekir. Proje kök
dizininde bir `.env` dosyası oluşturun:

```
ANTHROPIC_API_KEY=sk-ant-...
```

> ⚠️ `.env` dosyası `.gitignore` ile hariç tutulmuştur ve **asla** depoya yüklenmemelidir.

---

## Kullanım

```bash
# Pipeline'ı sırayla çalıştırın
python 01_veri_inceleme.py
python 02_veri_birlestirme.py
# ... 03 ... 11

# Web arayüzünü başlatın
streamlit run 10_arayuz.py
```

---

## Sonuçlar

Karşılaştırma çıktıları grafik olarak kaydedilir: `grafik_dogruluk.png`, `grafik_hiz.png`
ve her motor için confusion matrix (`cm_baseline.png`, `cm_motor_a.png`, `cm_motor_b.png`).
Eğitilmiş modeller `.pkl` dosyaları olarak depoda mevcuttur.

---

## Teknolojiler

scikit-learn · imbalanced-learn · Anthropic Claude · Streamlit · pandas · NumPy ·
matplotlib · seaborn · HuggingFace `datasets`
