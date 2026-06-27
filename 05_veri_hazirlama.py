# 05_veri_hazirla.py
# Amac: Etiketli veriyi yukle, egitim/test olarak bol, durumu kontrol et.

import pandas as pd
from sklearn.model_selection import train_test_split

print("Etiketli veri yukleniyor...")
df = pd.read_csv("etiketli_veri.csv")
df = df.dropna(subset=["text", "kategori"])      # bos satir varsa at
df = df[df["text"].astype(str).str.len() >= 3]   # cok kisa metinleri at

print(f"Toplam etiketli tweet: {len(df)}\n")
print("Kategori dagilimi:")
print(df["kategori"].value_counts())

# --- Egitim / Test bolmesi ---
X = df["text"]
y = df["kategori"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # %20 test
    random_state=42,
    stratify=y            # her kategori egitim ve testte ayni oranda
)

print(f"\n--- BOLME TAMAMLANDI ---")
print(f"Egitim seti: {len(X_train)} tweet")
print(f"Test seti:   {len(X_test)} tweet")

print("\nEgitim setindeki kategori dagilimi:")
print(y_train.value_counts())
print("\nTest setindeki kategori dagilimi:")
print(y_test.value_counts())