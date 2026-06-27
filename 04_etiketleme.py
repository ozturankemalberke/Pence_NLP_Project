# 04_etiketle.py
# Amac: NIHAI ozgun veri setini uretmek.
#   - Nadir sinif adaylarini (Tehdit/Santaj/Suc Ovgusu) hedefli topla + etiketle
#   - Genel zararli ornem ekle + etiketle
#   - Temiz tweet'leri dogrudan ekle
# Cikti: etiketli_veri.csv  (sutunlar: text, kategori)

import os
import time
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

# ----------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------
TEST_MODU = False          # False = TAM TUR (gercek set uretilir)
GENEL_ZARARLI = 2000       # Rastgele genel zararli ornem
TEMIZ_SAYISI = 3000        # Dogrudan Temiz (API'ye gitmez)
MODEL = "claude-haiku-4-5-20251001"

GECERLI = ["Tehdit", "Santaj", "Cinsel Taciz", "Hakaret", "Suc Ovgusu", "Temiz"]

# Nadir sinif aday anahtar kelimeleri (sayim adiminda dogruladigimiz listeler)
NADIR_ANAHTAR = {
    "Tehdit": ["oldurece", "oldurur", "gebert", "geberteceg", "parcalar",
               "kafani kop", "ayagini kir", "bedel odeyecek", "pisman edece",
               "seni bulur", "yakacag", "doverim", "ezece", "patlatir",
               "sokar", "keser", "mahvedece"],
    "Santaj": ["ifsa", "yayinlar", "herkese soyle", "anlatirim", "elimde",
               "fotograflar", "ekran goruntu", "duyururum", "rezil ede",
               "vermezsen", "yapmazsan", "soylersen", "sustur"],
    "Suc Ovgusu": ["iyi ki vur", "iyi ki old", "olmeliydi", "hak etti",
                   "asilmali", "asilsin", "gebersin", "helal olsun",
                   "daha cok ols", "yok edilmeli", "temizlenmeli", "soykirim"],
}

# ----------------------------------------------------------------------
load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def kategori_belirle(metin):
    prompt = f"""Sen bir icerik moderasyon uzmanisisin. Asagidaki Turkce metni
TEK BIR kategoriye ayir. Kategorileri dikkatlice ayirt et:

- Tehdit: Birine fiziksel zarar verme, oldurme, dovme, yaralama niyeti.
  Ornek: "seni gebertecegim", "evini yakariz",
- Santaj: Bir seyi ifsa etme/zarar verme tehdidiyle menfaat veya itaat isteme.
  Ornek: "vermezsen fotograflarini yayinlarim", "yapmazsan herkese soylerim"
- Cinsel Taciz: Cinsel icerikli, istenmeyen, rahatsiz edici, mustehcen ifadeler.
- Hakaret: Kufur, asagilama, onur kirici sozler. BUNA DAHIL: etnik/dini/irksal
  asagilamalar, cinsiyetci asagilamalar. Tehdit/cinsellik icermeyen tum kufurler.
- Suc Ovgusu: Sucu, siddeti, teroru veya suclulari oven, mesrulastiran, tesvik eden.
  Ornek: "iyi ki vurmuslar", "hepsi gebersin", "hak etti olmeyi"
- Temiz: Yukaridakilerden HICBIRI degil. Suc icermeyen siyasi yorum, normal
  tartisma veya alakasiz icerik. ANCAK kufur/hakaret varsa Temiz DEGILDIR.

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
    return "Hakaret"  # tam turda belirsizleri en genel zararliya ata


def etiketle_liste(df, baslik):
    """Bir tweet listesini Claude ile etiketler, sonuc listesi dondurur."""
    sonuc = []
    print(f"\n>>> {baslik}: {len(df)} tweet etiketleniyor...")
    for i, (_, satir) in enumerate(df.iterrows(), start=1):
        metin = str(satir["text"])
        try:
            k = kategori_belirle(metin)
        except Exception as e:
            print(f"    Hata ({i}): {e} -- atlandi")
            continue
        sonuc.append({"text": metin, "kategori": k})
        if i % 50 == 0 or i == len(df):
            print(f"    {i}/{len(df)} tamam")
        time.sleep(0.2)
    return sonuc


# ----------------------------------------------------------------------
# VERIYI HAZIRLA
# ----------------------------------------------------------------------
print("Havuz yukleniyor...")
havuz = pd.read_csv("birlesik_havuz.csv")
zararli = havuz[havuz["is_toxic"] == 1].copy()
temiz_havuz = havuz[havuz["is_toxic"] == 0].copy()

if TEST_MODU:
    GENEL_ZARARLI, TEMIZ_SAYISI = 20, 20
    print(">>> TEST MODU AKTIF (kucuk ornem)")

# 1) Nadir sinif adaylarini topla (tekrarlari engellemek icin index takip et)
nadir_df_list = []
secilen_idx = set()
for kategori, kelimeler in NADIR_ANAHTAR.items():
    desen = "|".join(kelimeler)
    maske = zararli["text"].str.lower().str.contains(desen, na=False)
    adaylar = zararli[maske]
    adaylar = adaylar[~adaylar.index.isin(secilen_idx)]  # daha once alinmadiysa
    if TEST_MODU:
        adaylar = adaylar.head(5)
    secilen_idx.update(adaylar.index)
    nadir_df_list.append(adaylar)
    print(f"Nadir aday - {kategori}: {len(adaylar)} tweet")

nadir_adaylar = pd.concat(nadir_df_list) if nadir_df_list else pd.DataFrame()

# 2) Genel zararli ornem (nadir adaylarla cakismayan)
kalan_zararli = zararli[~zararli.index.isin(secilen_idx)]
genel_zararli = kalan_zararli.sample(min(GENEL_ZARARLI, len(kalan_zararli)),
                                     random_state=42)

# 3) Temiz ornem
temiz = temiz_havuz.sample(min(TEMIZ_SAYISI, len(temiz_havuz)), random_state=42)

# ----------------------------------------------------------------------
# ETIKETLE
# ----------------------------------------------------------------------
tum_sonuc = []
tum_sonuc += etiketle_liste(nadir_adaylar, "Nadir sinif adaylari")
tum_sonuc += etiketle_liste(genel_zararli, "Genel zararli ornem")

# Temiz tweet'ler dogrudan (API'siz)
for _, satir in temiz.iterrows():
    tum_sonuc.append({"text": str(satir["text"]), "kategori": "Temiz"})

# ----------------------------------------------------------------------
# KAYDET ve OZET
# ----------------------------------------------------------------------
df = pd.DataFrame(tum_sonuc).drop_duplicates(subset=["text"]).reset_index(drop=True)
cikti_adi = "etiketli_veri_TEST.csv" if TEST_MODU else "etiketli_veri.csv"
df.to_csv(cikti_adi, index=False)

print("\n" + "="*60)
print("NIHAI ETIKETLEME TAMAMLANDI")
print("="*60)
print(f"Toplam etiketli satir: {len(df)}")
print("\nKategori dagilimi:")
print(df["kategori"].value_counts())
print(f"\n'{cikti_adi}' kaydedildi.")