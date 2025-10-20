# pipeline/translator.py

import logging
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def translate_text(text: str, source_lang="auto", target_lang="en") -> str:
    """
    Menerjemahkan satu kalimat menggunakan Google Translate.
    Args:
        text (str): teks input (biasanya dalam bahasa Indonesia)
        source_lang (str): bahasa sumber (default: 'auto')
        target_lang (str): bahasa tujuan (default: 'en')
    Returns:
        str: hasil terjemahan
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated.strip()
    except Exception as e:
        logging.error(f"Gagal menerjemahkan teks: {e}")
        return text  # fallback: kembalikan teks asli jika gagal


def translate_batch(texts: list[str], source_lang="auto", target_lang="en") -> list[str]:
    """
    Menerjemahkan list teks secara batch.
    """
    translated_texts = []
    for t in texts:
        translated_texts.append(translate_text(t, source_lang, target_lang))
    return translated_texts


if __name__ == "__main__":
    # Contoh uji cepat
    samples = [
        "Saya sangat suka produk ini!",
        "Pelayanannya buruk sekali.",
        "Lumayan lah, tidak terlalu bagus tapi tidak jelek juga."
    ]
    translated = translate_batch(samples)
    print("\n".join(translated))
