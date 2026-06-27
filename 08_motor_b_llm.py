# 08_motor_b_llm.py
# Amac: Motor B - Claude (LLM) ile ayni test setini siniflandirmak.
#       Motor A ile birebir ayni 1123 tweet kullanilir (adil kiyas).
#       Claude SADECE kategori verir; TCK/ceza sabit tablodan gelir (sonraki adim).
#       Her tweet icin gecikme (latency) olculur.
# Cikti: motor_b_sonuclar.csv

import os
import time
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

MODEL = "claude-haiku-4-5-20251001"
GECERLI = ["Tehdit", "Santaj", "Cinsel Taciz", "Hakaret", "Suc Ovgusu", "Temiz"]

# ----------------------------------------------------------------------
# 1. AYNI TEST SETINI OLUSTUR (Motor A ile birebir ayni bolme)
# ----------------------------------------------------------------------
print("Veri yukleniyor...")
df = pd.read_csv("etiketli_veri.csv").dropna(subset=["text", "kategori"])
df = df[df["text"].astype(str).str.len() >= 3]

X = df["text"].astype(str)
y = df["kategori"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Test seti: {len(X_test)} tweet (Motor A ile ayni)\n")

# ----------------------------------------------------------------------
# 2. CLAUDE ile SINIFLANDIRMA FONKSIYONU
# ----------------------------------------------------------------------
load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def claude_siniflandir(metin):
    prompt = f"""Sen bir icerik moderasyon uzmanisisin. Asagidaki Turkce metni
TEK BIR kategoriye ayir:

- Tehdit: Fiziksel zarar verme, oldurme, dovme niyeti.
- Santaj: Ifsa/zarar tehdidiyle menfaat veya itaat isteme.
- Cinsel Taciz: Cinsel icerikli, istenmeyen, rahatsiz edici ifadeler.
- Hakaret: Kufur, asagilama, etnik/dini/irksal asagilama.
- Suc Ovgusu: Sucu, siddeti, teroru oven/tesvik eden.
- Temiz: Yukaridakilerden hicbiri degil.

Metin: "{metin}"

SADECE kategori adini yaz. Baska aciklama yazma."""

    cevap = client.messages.create(
        model=MODEL, max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )
    sonuc = cevap.content[0].text.strip()
    for k in GECERLI:
        if k.lower() in sonuc.lower():
            return k
    return "Hakaret"  # eslesmezse en genel zararliya ata

# ----------------------------------------------------------------------
# 3. TUM TEST SETINI CLAUDE ile SINIFLANDIR (+ latency olc)
# ----------------------------------------------------------------------
print("Claude (Motor B) test setini siniflandiriyor...")
print("(Bu ~10-15 dakika surebilir, sabirla bekle)\n")

tahminler = []
gecikmeler = []
gercekler = list(y_test)
metinler = list(X_test)

for i, metin in enumerate(metinler, start=1):
    try:
        t0 = time.time()
        tahmin = claude_siniflandir(metin)
        gecikme = time.time() - t0
    except Exception as e:
        print(f"  Hata ({i}): {e} -- 'Temiz' atandi")
        tahmin = "Temiz"
        gecikme = 0.0
    tahminler.append(tahmin)
    gecikmeler.append(gecikme)
    if i % 50 == 0 or i == len(metinler):
        ort = sum(gecikmeler) / len(gecikmeler)
        print(f"  {i}/{len(metinler)} tamam | ort. gecikme: {ort:.2f}sn")
    time.sleep(0.1)

# ----------------------------------------------------------------------
# 4. SONUCLARI OLC
# ----------------------------------------------------------------------
acc = accuracy_score(gercekler, tahminler)
f1 = f1_score(gercekler, tahminler, average="macro")
ort_gecikme = sum(gecikmeler) / len(gecikmeler)

print("\n" + "="*55)
print("MOTOR B (Claude LLM) SONUCLARI")
print("="*55)
print(f"Accuracy (Dogruluk):    {acc:.3f}")
print(f"F1-Score (Macro):       {f1:.3f}")
print(f"Ortalama gecikme/tweet: {ort_gecikme:.2f} saniye")
print(f"Toplam sure (yaklasik): {sum(gecikmeler):.0f} saniye")
print("\nSinif bazli rapor:")
print(classification_report(gercekler, tahminler, zero_division=0))

print(f"\n--- UC YONLU KIYAS (F1-Macro) ---")
print(f"Baseline (Lojistik Reg.): 0.638")
print(f"Motor A  (Sinir Agi MLP): 0.648")
print(f"Motor B  (Claude LLM):    {f1:.3f}")

# ----------------------------------------------------------------------
# 5. SONUCLARI KAYDET (sonra grafik/arayuz icin)
# ----------------------------------------------------------------------
sonuc_df = pd.DataFrame({
    "text": metinler,
    "gercek": gercekler,
    "motor_b_tahmin": tahminler,
    "gecikme_sn": gecikmeler
})
sonuc_df.to_csv("motor_b_sonuclar.csv", index=False)
print("\nSonuclar kaydedildi: motor_b_sonuclar.csv")