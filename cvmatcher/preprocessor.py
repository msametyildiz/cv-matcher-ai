"""
preprocessor.py

Bu modül, gelen metin verilerini temizlemek, normalize etmek ve 
model eğitimine hazır hale getirmek için kullanılır.
Lemmatization, stopword kaldırma, özel karakter temizliği gibi işlemleri içerir.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

# NLTK veri setlerini indir
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

# Stopwords listesi (Türkçe + İngilizce)
STOPWORDS = set(stopwords.words("english") + stopwords.words("turkish"))

# Lemmatizer (İngilizce için)
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Metinden özel karakterleri, fazla boşlukları ve sayıları temizler.

    Args:
        text (str): Girdi metni

    Returns:
        str: Temizlenmiş metin
    """
    text = text.lower()
    text = re.sub(r"\S+@\S+", "", text)  # e-posta
    text = re.sub(r"\b\d{10,11}\b", "", text)  # telefon numarası
    text = re.sub(r"http\S+", "", text)  # URL
    text = re.sub(r"[^\w\s]", "", text)  # noktalama
    text = re.sub(r"\d+", "", text)  # sayılar
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """
    Stopwords listesindeki kelimeleri metinden kaldırır.

    Args:
        text (str): Temiz metin

    Returns:
        str: Stopword'süz metin
    """
    return " ".join([word for word in text.split() if word not in STOPWORDS])


def lemmatize_text(text: str) -> str:
    """
    İngilizce metni lemmatize eder.

    Args:
        text (str): Girdi metni

    Returns:
        str: Lemmatize edilmiş metin
    """
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def preprocess(text: str) -> str:
    """
    Tüm ön işleme adımlarını tek fonksiyonda birleştirir.

    Args:
        text (str): Girdi metni

    Returns:
        str: İşlenmiş metin
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def preprocess_batch(texts: List[str]) -> List[str]:
    """
    Birden fazla metni toplu olarak işler.

    Args:
        texts (List[str]): Metin listesi

    Returns:
        List[str]: Temizlenmiş metin listesi
    """
    return [preprocess(t) for t in texts]
