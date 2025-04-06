"""
train.py

Komut satırından çalıştırılabilen model eğitim betiğidir.
Veriyi yükler, işler ve modeli eğitir.
"""

import sys
import os
from pathlib import Path

# Üst dizini sys.path'e ekle (app modüllerini içeri aktarabilmek için)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from cvmatcher.trainer import train_model
from cvmatcher.config import PROCESSED_DATA_DIR


def main():
    training_path = PROCESSED_DATA_DIR / "training_data.csv"

    if not training_path.exists():
        print(f"[HATA] Eğitim verisi bulunamadı: {training_path}")
        return

    print("📥 Eğitim verisi yükleniyor...")
    df = pd.read_csv(training_path)

    # Gerekli sütunlar var mı?
    if not {"cv_text", "job_text", "label"}.issubset(df.columns):
        print("❗ 'cv_text', 'job_text', 'label' sütunlarını içeren bir CSV dosyası olmalı.")
        return

    print(f"✅ {len(df)} satırlık veri ile eğitim başlıyor...")
    train_model(df)


if __name__ == "__main__":
    main()
