"""
tokenizer.py

Hugging Face modelleri için tokenizer yükleme işlemleri bu modülde yapılır.
Ayrıca model girişleri için encode işlemi (tokenization) burada gerçekleştirilir.
"""

from transformers import AutoTokenizer
from cvmatcher.config import MODEL_NAME

# Tokenizer yükle (örneğin: 'bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_pair(text1: str, text2: str, max_length: int = 256):
    """
    İki metin parçasını tokenize eder ve BERT uyumlu hale getirir.

    Args:
        text1 (str): İlk metin (CV)
        text2 (str): İkinci metin (Job Description)
        max_length (int): Maksimum sekans uzunluğu

    Returns:
        dict: Tokenizer çıktısı (input_ids, attention_mask vb.)
    """
    return tokenizer(
        text=text1,
        text_pair=text2,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
