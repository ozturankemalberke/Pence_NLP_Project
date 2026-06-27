# 03_api_test.py
# Amac: API anahtarinin dogru okundugunu ve baglantinin calistigini dogrulamak.

import os
from dotenv import load_dotenv
from anthropic import Anthropic

# .env dosyasindaki anahtari yukle
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("HATA: Anahtar bulunamadi. .env dosyasini kontrol et.")
    exit()

print("Anahtar okundu (ilk 15 karakter):", api_key[:15], "...")

# Baglantiyi test et: kucuk bir istek gonder
client = Anthropic(api_key=api_key)

print("\nClaude'a test mesaji gonderiliyor...")
mesaj = client.messages.create(
    model="claude-haiku-4-5-20251001",   # Etiketleme icin hizli ve ucuz model
    max_tokens=50,
    messages=[
        {"role": "user", "content": "Sadece 'Baglanti basarili' yaz, baska hicbir sey yazma."}
    ]
)

print("Claude'un cevabi:", mesaj.content[0].text)
print("\nAPI baglantisi CALISIYOR. Etiketlemeye hazir.")