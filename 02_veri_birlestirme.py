# 02_veri_birlestir.py
# Amac: Iki HuggingFace setini birlestirip temiz, tekil bir havuz olusturmak.
#       Cikti: birlesik_havuz.csv (sutunlar: text, is_toxic)

from datasets import load_dataset
import pandas as pd

print("Veri setleri yukleniyor (onbellekten gelecek, hizli)...")

# --- Set 1: Overfit-GM ---
ds1 = load_dataset("Overfit-GM/turkish-toxic-language")["train"].to_pandas()
# Sadece ihtiyacimiz olan sutunlar; ortak isme cevir
set1 = ds1[["text", "is_toxic"]].copy()
set1.columns = ["text", "is_toxic"]
set1["kaynak"] = "overfit-gm"

# --- Set 2: Toygar (3 bolumu de birlestir) ---
ds2_dict = load_dataset("Toygar/turkish-offensive-language-detection")
parcalar = []
for bolum in ["train", "validation", "test"]:
    p = ds2_dict[bolum].to_pandas()[["text", "label"]].copy()
    parcalar.append(p)
ds2 = pd.concat(parcalar, ignore_index=True)
set2 = ds2.rename(columns={"label": "is_toxic"})[["text", "is_toxic"]].copy()
set2["kaynak"] = "toygar"

# --- Iki seti alt alta birlestir ---
havuz = pd.concat([set1, set2], ignore_index=True)
print(f"\nHam birlesik satir sayisi: {len(havuz)}")

# --- TEMIZLIK ---
# 1) is_toxic'i tam sayiya cevir, gecersizleri at
havuz["is_toxic"] = pd.to_numeric(havuz["is_toxic"], errors="coerce")
havuz = havuz.dropna(subset=["is_toxic"])
havuz["is_toxic"] = havuz["is_toxic"].astype(int)

# 2) Bos veya cok kisa metinleri at
havuz["text"] = havuz["text"].astype(str).str.strip()
havuz = havuz[havuz["text"].str.len() >= 3]

# 3) Ayni tweet'i tekrar tekrar saymayalim (kopyalari sil)
oncesi = len(havuz)
havuz = havuz.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Kopya temizligi: {oncesi} -> {len(havuz)} satir")

# --- OZET ---
print("\n--- BIRLESIK HAVUZ OZETI ---")
print("Toplam satir:", len(havuz))
print("\nEtiket dagilimi (0=temiz, 1=zararli):")
print(havuz["is_toxic"].value_counts())
print("\nKaynak dagilimi:")
print(havuz["kaynak"].value_counts())

# --- KAYDET ---
havuz.to_csv("birlesik_havuz.csv", index=False)
print("\n'birlesik_havuz.csv' kaydedildi. Proje klasorunde gorebilirsin.")