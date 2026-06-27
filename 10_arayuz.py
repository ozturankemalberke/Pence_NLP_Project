# 10_arayuz.py
# PENCE - Web Arayuzu (Streamlit) - TAM SURUM (Motor A v2)
# Sekmeler: Metin Analizi (2 motor) | Akis Simulasyonu | Bilimsel Sonuclar

import streamlit as st
import joblib
import time
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

st.set_page_config(page_title="PENCE", page_icon="🦅", layout="wide")

# ----------------------------------------------------------------------
# TCK ESLESTIRME TABLOSU (sabit - LLM uretmez)
# ----------------------------------------------------------------------
TCK_TABLO = {
    "Tehdit":       {"madde": "TCK 106", "ceza": "6 ay - 2 yil hapis"},
    "Santaj":       {"madde": "TCK 107", "ceza": "1 - 3 yil hapis + adli para cezasi"},
    "Cinsel Taciz": {"madde": "TCK 105", "ceza": "3 ay - 2 yil hapis (bedensel temas olmaksizin)"},
    "Hakaret":      {"madde": "TCK 125", "ceza": "3 ay - 2 yil hapis (sosyal medyada artirim)"},
    "Suc Ovgusu":   {"madde": "TCK 215", "ceza": "2 yila kadar hapis"},
    "Temiz":        {"madde": "-",       "ceza": "Suc unsuru tespit edilmedi"},
}
GECERLI = ["Tehdit", "Santaj", "Cinsel Taciz", "Hakaret", "Suc Ovgusu", "Temiz"]

# ----------------------------------------------------------------------
# MODELLERI ve API'yi YUKLE
# ----------------------------------------------------------------------
@st.cache_resource
def kaynaklari_yukle():
    model_a = joblib.load("model_motor_a_v2.pkl")
    vec_a = joblib.load("vectorizer_motor_a_v2.pkl")
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = None
    if api_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
        except Exception:
            client = None
    return model_a, vec_a, client

model_a, vec_a, client = kaynaklari_yukle()
MODEL_LLM = "claude-haiku-4-5-20251001"

# ----------------------------------------------------------------------
# MOTOR A (yerel sinir agi)
# ----------------------------------------------------------------------
def motor_a_tahmin(metin):
    t0 = time.time()
    kategori = model_a.predict(vec_a.transform([metin]))[0]
    return kategori, time.time() - t0

# ----------------------------------------------------------------------
# MOTOR B (Claude) - internet yoksa None doner, cokmez
# ----------------------------------------------------------------------
def motor_b_tahmin(metin):
    if client is None:
        return None, 0.0
    prompt = f"""Sen bir icerik moderasyon uzmanisisin. Asagidaki Turkce metni
TEK BIR kategoriye ayir: Tehdit, Santaj, Cinsel Taciz, Hakaret, Suc Ovgusu, Temiz.

- Tehdit: Fiziksel zarar verme, oldurme niyeti.
- Santaj: Ifsa/zarar tehdidiyle menfaat isteme.
- Cinsel Taciz: Cinsel icerikli rahatsiz edici ifadeler.
- Hakaret: Kufur, asagilama, etnik/dini asagilama.
- Suc Ovgusu: Sucu/siddeti oven, tesvik eden.
- Temiz: Hicbiri degil.

Metin: "{metin}"
SADECE kategori adini yaz."""
    try:
        t0 = time.time()
        cevap = client.messages.create(
            model=MODEL_LLM, max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )
        gecikme = time.time() - t0
        sonuc = cevap.content[0].text.strip()
        for k in GECERLI:
            if k.lower() in sonuc.lower():
                return k, gecikme
        return "Hakaret", gecikme
    except Exception:
        return None, 0.0

# ----------------------------------------------------------------------
# DELIL PDF URET
# ----------------------------------------------------------------------
def delil_pdf_uret(metin, kat_a, kat_b):
    from fpdf import FPDF
    tck = TCK_TABLO.get(kat_a, TCK_TABLO["Temiz"])
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "PENCE - Delil Raporu", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Incelenen Metin:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    # Turkce karakterleri PDF icin sadelestir
    guvenli = metin.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 7, guvenli)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Tespit Sonuclari:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Motor A (Yerel): {kat_a}", ln=True)
    pdf.cell(0, 7, f"Motor B (Claude): {kat_b if kat_b else 'Ulasilamadi'}", ln=True)
    pdf.cell(0, 7, f"Ilgili Madde: {tck['madde']}", ln=True)
    pdf.cell(0, 7, f"Olasi Yaptirim: {tck['ceza']}", ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6, "Not: Bu rapor otomatik uretilmistir ve bilgilendirme "
                         "amaclidir. Hukuki danismanlik niteligi tasimaz.")
    return bytes(pdf.output())

# ======================================================================
# BASLIK
# ======================================================================
st.title("🦅 PENÇE")
st.caption("Suç büyümeden, suçlu kaçmadan — Türkçe sosyal medya suç tespit sistemi")

if client is None:
    st.warning("⚠️ Motor B (Claude) şu an devre dışı (internet/API yok). Motor A çalışmaya devam ediyor.")

sekme1, sekme2, sekme3 = st.tabs(["🔍 Metin Analizi", "📡 Akış Simülasyonu", "📊 Bilimsel Sonuçlar"])

# ======================================================================
# SEKME 1: METIN ANALIZI (iki motor yan yana)
# ======================================================================
with sekme1:
    st.subheader("Tek Metin Analizi — İki Motor Karşılaştırmalı")
    metin = st.text_area("Analiz edilecek tweet/metni girin:",
                         placeholder="Örnek: seni bulup geberteceğim...", height=100)

    if st.button("Analiz Et", type="primary"):
        if not metin.strip():
            st.warning("Lütfen bir metin girin.")
        else:
            kat_a, gec_a = motor_a_tahmin(metin)
            with st.spinner("Motor B (Claude) düşünüyor..."):
                kat_b, gec_b = motor_b_tahmin(metin)

            kol_a, kol_b = st.columns(2)

            # --- MOTOR A ---
            with kol_a:
                st.markdown("### 🖥️ Motor A — Yerel Sinir Ağı")
                if kat_a == "Temiz":
                    st.success(f"**{kat_a}**")
                else:
                    st.error(f"**{kat_a}**")
                st.metric("Gecikme", f"{gec_a*1000:.1f} ms")
                st.caption("Ücretsiz • İnternetsiz • %100 yerel")

            # --- MOTOR B ---
            with kol_b:
                st.markdown("### 🤖 Motor B — Claude (LLM)")
                if kat_b is None:
                    st.warning("Ulaşılamadı (internet/API yok)")
                elif kat_b == "Temiz":
                    st.success(f"**{kat_b}**")
                else:
                    st.error(f"**{kat_b}**")
                if kat_b is not None:
                    st.metric("Gecikme", f"{gec_b:.2f} sn")
                st.caption("Yüksek doğruluk • Gerekçeli • Ücretli/çevrimiçi")

            # --- TCK KARTI (Motor A kategorisine gore) ---
            st.divider()
            tck = TCK_TABLO.get(kat_a, TCK_TABLO["Temiz"])
            k1, k2 = st.columns(2)
            k1.metric("İlgili Madde", tck["madde"])
            k2.metric("Olası Yaptırım", "—")
            st.info(f"**Olası Yaptırım:** {tck['ceza']}")
            st.caption("⚠️ Bu çıktı bilgilendirme amaçlıdır, hukuki danışmanlık değildir.")

            # --- DELIL PDF ---
            pdf_bytes = delil_pdf_uret(metin, kat_a, kat_b)
            st.download_button("📄 Delil Raporu (PDF) İndir", data=pdf_bytes,
                               file_name="pence_delil_raporu.pdf", mime="application/pdf")

# ======================================================================
# SEKME 2: AKIS SIMULASYONU
# ======================================================================
with sekme2:
    st.subheader("Canlı Akış Simülasyonu")
    st.caption("Gerçek bir sosyal medya akışını taklit eder; PENÇE her metni anında tarar (Motor A).")

    ornek_akis = [
        "günaydın millet bugün hava çok güzel",
        "seni bulup geberteceğim haberin olsun",
        "bu yemek tarifi gerçekten çok başarılı olmuş",
        "salak gerizekalı defol git buradan",
        "maç bu akşam saat kaçta başlıyor",
        "yunan artığı ermeni tohumu defolun bu ülkeden",
        "iyi ki o teröristleri vurmuşlar hepsi gebersin",
        "yeni telefonum geldi çok memnunum",
    ]

    if st.button("Akışı Başlat", type="primary"):
        alan = st.empty()
        kayitlar = []
        for tw in ornek_akis:
            kat, gec = motor_a_tahmin(tw)
            durum = "✅ TEMİZ" if kat == "Temiz" else f"🚨 {kat.upper()}"
            kayitlar.insert(0, {"Durum": durum, "Metin": tw,
                                "Madde": TCK_TABLO[kat]["madde"]})
            alan.dataframe(pd.DataFrame(kayitlar), use_container_width=True, hide_index=True)
            time.sleep(0.8)
        st.success("Akış tarama tamamlandı.")

# ======================================================================
# SEKME 3: BILIMSEL SONUCLAR
# ======================================================================
with sekme3:
    st.subheader("Üç Motor — Bilimsel Karşılaştırma")

    # Metrik ozet tablosu (Motor A = v2)
    veri = {
        "Model": ["Baseline (Lojistik Reg.)", "Motor A (Sinir Ağı v2)", "Motor B (Claude)"],
        "Accuracy": [0.817, 0.850, 0.899],
        "F1-Macro": [0.638, 0.628, 0.706],
        "Hız (sn/tweet)": [0.001, 0.001, 2.471],
        "Maliyet": ["Ücretsiz", "Ücretsiz", "Ücretli"],
    }
    st.dataframe(pd.DataFrame(veri), use_container_width=True, hide_index=True)
    st.caption("Not: Motor A v2'de F1-Macro'daki ufak fark, tek örnekli (support=1) "
               "Şantaj sınıfının gürültüsünden kaynaklanır; anlamlı tüm sınıflar (Tehdit, "
               "Hakaret, Temiz, Suç Övgüsü) iyileşmiştir.")

    st.markdown("#### Doğruluk Karşılaştırması")
    if os.path.exists("grafik_dogruluk.png"):
        st.image("grafik_dogruluk.png")

    st.markdown("#### Hız Karşılaştırması (log ölçek)")
    if os.path.exists("grafik_hiz.png"):
        st.image("grafik_hiz.png")

    st.markdown("#### Confusion Matrix — Hata Haritaları")
    c1, c2, c3 = st.columns(3)
    if os.path.exists("cm_baseline.png"):
        c1.image("cm_baseline.png", caption="Baseline")
    if os.path.exists("cm_motor_a.png"):
        c2.image("cm_motor_a.png", caption="Motor A")
    if os.path.exists("cm_motor_b.png"):
        c3.image("cm_motor_b.png", caption="Motor B")

    st.divider()
    st.markdown("""
    **Sonuç:** Tek bir "kazanan" yoktur. Motor B en yüksek doğruluğu verir ama
    ~2.5 sn/tweet ve ücretlidir. Motor A neredeyse anlık, ücretsiz ve internetsizdir.
    Yüksek hacimli gerçek zamanlı tarama için Motor A; az sayıda şüpheli içeriğin
    en yüksek doğrulukla incelenmesi için Motor B uygundur. İdeal sistem ikisini
    birlikte kullanır: Motor A hızlı eler, Motor B şüphelileri teyit eder.
    """)