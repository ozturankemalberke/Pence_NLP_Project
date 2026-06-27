# 11_motor_a_v2.py
# Motor A - GELISTIRILMIS SURUM
# Fark: TF-IDF artik (1) kelime ikililerine (bigram) ve
#       (2) karakter n-gramlarina da bakiyor. Siniflandirici ayni (MLP).
# Amac: Ayni veri + ayni bolme ile adil kiyas -> F1 artti mi?

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler

print("=" * 60)
print("MOTOR A v2 - Gelistirilmis ozellik cikarimi")
print("=" * 60)

# ----------------------------------------------------------------------
# 1) VERIYI OKU ve AYNI SEKILDE BOL (adil kiyas icin random_state=42)
# ----------------------------------------------------------------------
df = pd.read_csv("etiketli_veri.csv")
df = df.dropna(subset=["text", "kategori"])
X = df["text"].astype(str)
y = df["kategori"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Egitim: {len(X_train)} | Test: {len(X_test)}")

# ----------------------------------------------------------------------
# 2) YENI OZELLIK CIKARIMI
#    word: tek kelime + kelime ikilileri (1,2)
#    char: kelime icindeki 2-5 harflik diziler -> Turkce ek sorununu cozer
# ----------------------------------------------------------------------
kelime_vec = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2),
    min_df=2, max_features=15000, sublinear_tf=True
)
karakter_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(2, 5),
    min_df=2, max_features=15000, sublinear_tf=True
)
# Ikisini birlestir: metin hem kelime hem karakter gozuyle okunur
vectorizer = FeatureUnion([
    ("kelime", kelime_vec),
    ("karakter", karakter_vec),
])

print("Ozellikler cikariliyor (kelime ikilileri + karakter n-gram)...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Toplam ozellik sayisi: {X_train_vec.shape[1]}")  # eskisinden cok daha fazla

# ----------------------------------------------------------------------
# 3) DENGESIZLIGI COZ (nadir siniflari cogalt) - eskisiyle ayni mantik
# ----------------------------------------------------------------------
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_vec, y_train)
print(f"Oversampling sonrasi egitim ornegi: {X_train_bal.shape[0]}")

# ----------------------------------------------------------------------
# 4) SINIFLANDIRICI - eskisiyle AYNI (tek degisken: ozellikler)
# ----------------------------------------------------------------------
print("Sinir agi egitiliyor (biraz surebilir)...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=300, random_state=42, early_stopping=True
)
model.fit(X_train_bal, y_train_bal)

# ----------------------------------------------------------------------
# 5) DEGERLENDIR ve ESKI SKORLA KIYASLA
# ----------------------------------------------------------------------
y_pred = model.predict(X_test_vec)
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n" + "=" * 60)
print("SONUCLAR")
print("=" * 60)
print(classification_report(y_test, y_pred, zero_division=0))
print("-" * 60)
print(f"ESKI Motor A (F1-Macro):  0.648")
print(f"YENI Motor A v2 (F1-Macro): {f1_macro:.3f}")
fark = f1_macro - 0.648
print(f"FARK: {fark:+.3f}  ({'IYILESTI' if fark > 0 else 'DEGISMEDI/DUSTU'})")
print("=" * 60)

# ----------------------------------------------------------------------
# 6) O MESHUR CUMLEYI TEST ET
# ----------------------------------------------------------------------
test_cumleleri = [
    "seni bulup geberteceğim haberin olsun",
    "paranı vermezsen o fotoğrafları herkese atarım",
    "günaydın millet bugün hava çok güzel",
    "salak gerizekalı defol git buradan",
]
print("\nCANLI TEST (yeni model ne diyor?):")
for c in test_cumleleri:
    tahmin = model.predict(vectorizer.transform([c]))[0]
    print(f"  '{c[:45]}...' -> {tahmin}")

# ------------------------------lm----------------------------------------
# 7) AYRI ISIMLE KAYDET (eski model bozulmasin)
# ----------------------------------------------------------------------
joblib.dump(model, "model_motor_a_v2.pkl")
joblib.dump(vectorizer, "vectorizer_motor_a_v2.pkl")
print("\nKaydedildi: model_motor_a_v2.pkl + vectorizer_motor_a_v2.pkl")
print("(Eski model_motor_a.pkl dokunuadan duruyor.)")