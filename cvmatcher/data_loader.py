"""
data_loader.py

Ham verilerin ve etiketli örneklerin yüklenmesini sağlar.
Bu modül CSV, JSON veya TXT formatındaki veri dosyalarını okuyarak
model eğitimi veya tahmin süreci için hazır hale getirir.
"""

import pandas as pd
from pathlib import Path
from cvmatcher.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DIR


def load_csv_data(file_path: Path) -> pd.DataFrame:
    """
    CSV formatında veri yükler.

    Args:
        file_path (Path): CSV dosyasının yolu

    Returns:
        pd.DataFrame: Yüklenen veriler
    """
    return pd.read_csv(file_path)


def load_sample_pairs() -> pd.DataFrame:
    """
    Örnek veri çifti (CV-Job Description) yükler.

    Returns:
        pd.DataFrame: Örnek eşleşmeler içeren dataframe
    """
    sample_file = SAMPLES_DIR / "sample_pairs.csv"
    return load_csv_data(sample_file)


def load_training_data() -> pd.DataFrame:
    """
    Model eğitimi için etiketli veri setini yükler.

    Returns:
        pd.DataFrame: Etiketli eğitim verisi
    """
    training_file = PROCESSED_DATA_DIR / "training_data.csv"
    return load_csv_data(training_file)


def save_dataframe(df: pd.DataFrame, file_path: Path):
    """
    DataFrame'i CSV dosyasına kaydeder.

    Args:
        df (pd.DataFrame): Kayıt edilecek veri
        file_path (Path): Hedef dosya yolu
    """
    df.to_csv(file_path, index=False)
