# predictor.py

# Doğrudan model.py'den import edin
from cvmatcher.model import CVMatcherModel  

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cvmatcher.config import BEST_MODEL_PATH, TRAINING_PARAMS
from cvmatcher.tokenizer import tokenizer
from cvmatcher.preprocessor import preprocess

def load_model():
    """
    Modeli yükler ve değerlendirme modunda hazır hale getirir.
    """
    model = CVMatcherModel()  # Modeli burada oluşturuyorsunuz
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
    model.eval()  # Modeli değerlendirme moduna alıyoruz
    return model

class PredictionDataset(Dataset):
    def __init__(self, cv_texts, job_text):
        """
        Eğitim verilerini uygun formata dönüştürmek için.
        
        Args:
            cv_texts (List[str]): CV metinleri
            job_text (str): İş tanımı metni
        """
        self.cv_texts = [preprocess(t) for t in cv_texts]
        self.job_text = preprocess(job_text)

    def __len__(self):
        """
        Dataset uzunluğunu döndürür.
        """
        return len(self.cv_texts)

    def __getitem__(self, idx):
        """
        Belirli bir indeks için giriş verisi döndürür.

        Args:
            idx (int): İndeks numarası

        Returns:
            Dict: Model için gerekli veriler (input_ids, attention_mask)
        """
        encoding = tokenizer(
            text=self.cv_texts[idx],
            text_pair=self.job_text,
            padding="max_length",
            truncation=True,
            max_length=TRAINING_PARAMS["max_seq_length"],
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

def predict_scores(cv_texts, job_text):
    """
    CV listesine karşılık, tek bir iş tanımına uygunluk puanı verir.

    Args:
        cv_texts (List[str]): CV metinleri
        job_text (str): İş tanımı metni

    Returns:
        List[Tuple[int, float]]: (cv_index, skor)
    """
    dataset = PredictionDataset(cv_texts, job_text)
    dataloader = DataLoader(dataset, batch_size=8)

    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scores = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Tahmin Ediliyor...")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # ✅ Skorları güvenli şekilde ekle
            for score in outputs.squeeze().cpu().numpy().reshape(-1):
                scores.append(float(score))


    # Skorları %'ye çevir ve indeksle
    result = [(i, round(score * 100, 2)) for i, score in enumerate(scores)]
    return sorted(result, key=lambda x: x[1], reverse=True)
