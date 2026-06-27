# 01_veri_incele.py
# Amac: Iki HuggingFace Turkce veri setini indirip yapilarini incelemek.

from datasets import load_dataset
import pandas as pd

pd.set_option("display.max_colwidth", 80)  # Tweet metinleri kisalmadan gorunsun

print("=" * 60)
print("VERI SETI 1: Overfit-GM/turkish-toxic-language indiriliyor...")
print("=" * 60)
ds1 = load_dataset("Overfit-GM/turkish-toxic-language")
print("\nVeri setinin yapisi (hangi bolumler/sutunlar var):")
print(ds1)

# Egitim bolumunu pandas tablosuna cevirip ilk 5 satira bakalim
df1 = ds1["train"].to_pandas()
print("\nIlk 5 satir:")
print(df1.head())
print("\nSatir sayisi:", len(df1))
print("Sutunlar:", list(df1.columns))


print("\n" + "=" * 60)
print("VERI SETI 2: Toygar/turkish-offensive-language-detection indiriliyor...")
print("=" * 60)
ds2 = load_dataset("Toygar/turkish-offensive-language-detection")
print("\nVeri setinin yapisi:")
print(ds2)

df2 = ds2["train"].to_pandas()
print("\nIlk 5 satir:")
print(df2.head())
print("\nSatir sayisi:", len(df2))
print("Sutunlar:", list(df2.columns))

print("\n\nINCELEME TAMAMLANDI.")