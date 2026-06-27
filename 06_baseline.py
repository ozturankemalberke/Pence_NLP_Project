# 06_baseline.py
# Amac: Basit temel model (TF-IDF + Lojistik Regresyon) egitmek.
#       Bu, LSTM ile karsilastirilacak "kiyas citasi"dir.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# --- 1. VERIYI YUKLE ve BOL (05'teki ile ayni mantik) ---
print("Veri yukleniyor...")
df = pd.read_csv("etiketli_veri.csv").dropna(subset=["text", "kategori"])
df = df[df["text"].astype(str).str.len() >= 3]

X = df["text"]
y = df["kategori"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Egitim: {len(X_train)} | Test: {len(X_test)}\n")

# --- 2. METNI SAYIYA CEVIR (TF-IDF) ---
print("TF-IDF vektorlestirme...")
vectorizer = TfidfVectorizer(
    max_features=5000,    # en onemli 5000 kelimeyi kullan
    ngram_range=(1, 2)    # tek kelimeler + ikili kelime gruplari
)
X_train_vec = vectorizer.fit_transform(X_train)  # egitimden ogren + cevir
X_test_vec = vectorizer.transform(X_test)         # testi ayni sozlukle cevir

# --- 3. MODELI EGIT (Lojistik Regresyon) ---
print("Model egitiliyor...")
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"   # nadir siniflara ekstra dikkat (Santaj vb.)
)
model.fit(X_train_vec, y_train)

# --- 4. TEST ET ve OLC ---
print("Test ediliyor...\n")
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")  # tum siniflarin ortalamasi

print("="*55)
print("BASELINE MODEL SONUCLARI (TF-IDF + Lojistik Regresyon)")
print("="*55)
print(f"Accuracy (Dogruluk): {acc:.3f}")
print(f"F1-Score (Macro):    {f1:.3f}")
print("\nSinif bazli detayli rapor:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- 5. MODELI KAYDET (sonra arayuzde kullanacagiz) ---
joblib.dump(model, "model_baseline.pkl")
joblib.dump(vectorizer, "vectorizer_baseline.pkl")
print("Model kaydedildi: model_baseline.pkl + vectorizer_baseline.pkl")