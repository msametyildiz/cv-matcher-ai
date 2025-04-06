"""
config.py

Uygulama genelinde kullanılacak sabitler, dosya yolları ve ayarlar burada tanımlanır.
Bu yapı, projenin farklı bileşenlerinde tekrarlı tanımlamaların önüne geçer.
"""

import os
from pathlib import Path

# Ana dizin (projenin çalıştırıldığı yerin kökü)
BASE_DIR = Path(__file__).resolve().parent.parent

# Veri dizinleri
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

# Model dizini
MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pt"

# Çıktılar (grafikler, PDF raporlar vs.)
OUTPUT_DIR = BASE_DIR / "outputs"

# PDF rapor dosya adı
PDF_REPORT_PATH = OUTPUT_DIR / "cv_job_report.pdf"

# Max karakter limiti gösterim için
TEXT_PREVIEW_LIMIT = 500

# Varsayılan benzerlik türü
DEFAULT_SIMILARITY_METHOD = "BERT"

# Eğitim parametreleri
TRAINING_PARAMS = {
    "batch_size": 4,
    "epochs": 2,
    "learning_rate": 2e-5,
    "max_seq_length": 128
}

# Model ve tokenizer adı (HuggingFace için)
MODEL_NAME = "bert-base-uncased"

# Rastgelelik için sabit seed
SEED = 42
