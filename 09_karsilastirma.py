# 09_karsilastirma.py
# Amac: Uc modeli (Baseline, Motor A v2, Motor B) bilimsel olarak karsilastirmak.
#       - Karsilastirma bar grafigi (F1/Accuracy)
#       - Her motor icin Confusion Matrix (hata haritasi)
#       Ciktilar: grafik PNG dosyalari olarak kaydedilir.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import joblib

# Turkce karakter ve stil
plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")

KATEGORILER = ["Cinsel Taciz", "Hakaret", "Santaj", "Suc Ovgusu", "Tehdit", "Temiz"]

# ----------------------------------------------------------------------
# 1. AYNI TEST SETINI YENIDEN OLUSTUR (tum motorlar icin ayni)
# ----------------------------------------------------------------------
print("Veri ve test seti hazirlaniyor...")
df = pd.read_csv("etiketli_veri.csv").dropna(subset=["text", "kategori"])
df = df[df["text"].astype(str).str.len() >= 3]

X = df["text"].astype(str)
y = df["kategori"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 2. MOTOR A (v2) ve BASELINE TAHMINLERINI URET (kayitli modellerden)
# ----------------------------------------------------------------------
print("Baseline ve Motor A (v2) tahminleri uretiliyor...")

# Baseline
model_bl = joblib.load("model_baseline.pkl")
vec_bl = joblib.load("vectorizer_baseline.pkl")
pred_bl = model_bl.predict(vec_bl.transform(X_test))

# Motor A (v2 - gelistirilmis ozellik cikarimi)
model_a = joblib.load("model_motor_a_v2.pkl")
vec_a = joblib.load("vectorizer_motor_a_v2.pkl")
pred_a = model_a.predict(vec_a.transform(X_test))

# ----------------------------------------------------------------------
# 3. MOTOR B TAHMINLERINI YUKLE (kayitli CSV'den)
# ----------------------------------------------------------------------
print("Motor B tahminleri yukleniyor...")
mb = pd.read_csv("motor_b_sonuclar.csv")
pred_b = mb["motor_b_tahmin"].tolist()
gercek_b = mb["gercek"].tolist()
ort_gecikme_b = mb["gecikme_sn"].mean()

# ----------------------------------------------------------------------
# 4. METRIKLERI HESAPLA
# ----------------------------------------------------------------------
sonuc = {
    "Baseline\n(Lojistik Reg.)": {
        "acc": accuracy_score(y_test, pred_bl),
        "f1": f1_score(y_test, pred_bl, average="macro"),
        "gecikme": 0.001  # yaklasik anlik
    },
    "Motor A\n(Sinir Agi v2)": {
        "acc": accuracy_score(y_test, pred_a),
        "f1": f1_score(y_test, pred_a, average="macro"),
        "gecikme": 0.001
    },
    "Motor B\n(Claude LLM)": {
        "acc": accuracy_score(gercek_b, pred_b),
        "f1": f1_score(gercek_b, pred_b, average="macro"),
        "gecikme": ort_gecikme_b
    },
}

print("\n--- OZET ---")
for ad, m in sonuc.items():
    print(f"{ad.replace(chr(10),' ')}: Acc={m['acc']:.3f} F1={m['f1']:.3f} Gecikme={m['gecikme']:.2f}sn")

# ----------------------------------------------------------------------
# 5. GRAFIK 1: F1 ve Accuracy karsilastirma bar grafigi
# ----------------------------------------------------------------------
adlar = list(sonuc.keys())
f1ler = [sonuc[a]["f1"] for a in adlar]
accler = [sonuc[a]["acc"] for a in adlar]

x = np.arange(len(adlar))
genislik = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - genislik/2, accler, genislik, label="Accuracy", color="#4C72B0")
b2 = ax.bar(x + genislik/2, f1ler, genislik, label="F1-Macro", color="#DD8452")

ax.set_ylabel("Skor")
ax.set_title("PENCE - Uc Motor Karsilastirmasi (Dogruluk)")
ax.set_xticks(x)
ax.set_xticklabels(adlar)
ax.set_ylim(0, 1)
ax.legend()
ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
plt.tight_layout()
plt.savefig("grafik_dogruluk.png", bbox_inches="tight")
print("\nKaydedildi: grafik_dogruluk.png")

# ----------------------------------------------------------------------
# 6. GRAFIK 2: Hiz (gecikme) karsilastirmasi - log olcek
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
gecikmeler = [sonuc[a]["gecikme"] for a in adlar]
renkler = ["#55A868", "#55A868", "#C44E52"]
bars = ax.bar(adlar, gecikmeler, color=renkler)
ax.set_ylabel("Tweet basina gecikme (saniye, log olcek)")
ax.set_title("PENCE - Hiz Karsilastirmasi (dusuk = hizli)")
ax.set_yscale("log")
ax.bar_label(bars, fmt="%.3f sn", padding=3, fontsize=9)
plt.tight_layout()
plt.savefig("grafik_hiz.png", bbox_inches="tight")
print("Kaydedildi: grafik_hiz.png")

# ----------------------------------------------------------------------
# 7. GRAFIK 3: Confusion Matrix (her motor icin)
# ----------------------------------------------------------------------
def confusion_ciz(gercek, tahmin, baslik, dosya):
    cm = confusion_matrix(gercek, tahmin, labels=KATEGORILER)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=KATEGORILER, yticklabels=KATEGORILER, cbar=False)
    plt.xlabel("Tahmin edilen")
    plt.ylabel("Gercek")
    plt.title(baslik)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(dosya, bbox_inches="tight")
    print(f"Kaydedildi: {dosya}")

confusion_ciz(y_test, pred_bl, "Confusion Matrix - Baseline", "cm_baseline.png")
confusion_ciz(y_test, pred_a,  "Confusion Matrix - Motor A (Sinir Agi v2)", "cm_motor_a.png")
confusion_ciz(gercek_b, pred_b, "Confusion Matrix - Motor B (Claude)", "cm_motor_b.png")

print("\nTum grafikler proje klasorune kaydedildi.")
plt.show()  # grafikleri ekranda da goster