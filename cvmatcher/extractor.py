"""
extractor.py

Bu modül, farklı dosya formatlarından (PDF, DOCX, TXT) metin çıkarmayı sağlar.
İleride OCR destekli resim dosyaları (.jpg, .png) da eklenebilir.
"""

import os
from pathlib import Path

try:
    import pdfplumber
    from docx import Document
except ImportError:
    raise ImportError("Lütfen pdfplumber ve python-docx paketlerini yükleyin: pip install pdfplumber python-docx")


def extract_text_from_pdf(file_path: str) -> str:
    """PDF dosyasından metin çıkarır."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"[HATA] PDF okuma hatası: {file_path} → {e}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """DOCX dosyasından metin çıkarır."""
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"[HATA] DOCX okuma hatası: {file_path} → {e}")
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """TXT dosyasından metin çıkarır."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[HATA] TXT okuma hatası: {file_path} → {e}")
        return ""


def extract_text(file_path: str) -> str:
    """Dosya türüne göre uygun çıkarıcıyı seçer."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        print(f"[UYARI] Desteklenmeyen dosya türü: {file_path}")
        return ""
