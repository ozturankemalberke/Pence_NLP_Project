# 07_motor_a_lstm.py
# Amac: Motor A - Cok katmanli yapay sinir agi (MLP) + Oversampling.
#       Dengesiz veri sorununu cozmek icin nadir siniflar egitimde cogaltilir.
#       Baseline (F1=0.638) ile karsilastirilacak.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import joblib

np.random.seed(42)

# ----------------------------------------------------------------------
# 1. VERIYI YUKLE ve BOL
# ----------------------------------------------------------------------
print("Veri yukleniyor...")
df = pd.read_csv("etiketli_veri.csv").dropna(subset=["text", "kategori"])
df = df[df["text"].astype(str).str.len() >= 3]

X = df["text"].astype(str)
y = df["kategori"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Egitim: {len(X_train)} | Test: {len(X_test)}\n")

# ----------------------------------------------------------------------
# 2. METNI SAYIYA CEVIR (TF-IDF)
# ----------------------------------------------------------------------
print("TF-IDF vektorlestirme...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------------------------------------------------
# 3. OVERSAMPLING - SADECE EGITIM SETINE
#    Nadir siniflarin orneklerini cogaltarak dengeli hale getir.
#    Test setine DOKUNULMAZ (gercek dunyayi temsil etmeli).
# ----------------------------------------------------------------------
print("\nEgitim seti dengeleniyor (oversampling)...")
print("Once (egitim):")
print(y_train.value_counts())

ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_vec, y_train)

print("\nSonra (dengelenmis egitim):")
print(pd.Series(y_train_bal).value_counts())

# ----------------------------------------------------------------------
# 4. SINIR AGINI EGIT (MLP)
# ----------------------------------------------------------------------
print("\nSinir agi (MLP) egitiliyor...\n")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True,
    verbose=True
)
model.fit(X_train_bal, y_train_bal)

# ----------------------------------------------------------------------
# 5. TEST ET ve OLC (orijinal, dokunulmamis test setiyle)
# ----------------------------------------------------------------------
print("\nTest ediliyor...")
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("\n" + "="*55)
print("MOTOR A (Sinir Agi MLP + Oversampling) SONUCLARI")
print("="*55)
print(f"Accuracy (Dogruluk): {acc:.3f}")
print(f"F1-Score (Macro):    {f1:.3f}")
print("\nSinif bazli rapor:")
print(classification_report(y_test, y_pred, zero_division=0))

print(f"\n--- KIYAS ---")
print(f"Baseline (Lojistik Reg.) F1-Macro: 0.638")
print(f"Motor A  (Sinir Agi MLP) F1-Macro: {f1:.3f}")

# ----------------------------------------------------------------------
# 6. MODELI KAYDET
# ----------------------------------------------------------------------
joblib.dump(model, "model_motor_a.pkl")
joblib.dump(vectorizer, "vectorizer_motor_a.pkl")
print("\nModel kaydedildi: model_motor_a.pkl + vectorizer_motor_a.pkl")