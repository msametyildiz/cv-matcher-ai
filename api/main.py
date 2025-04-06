"""
main.py

FastAPI ile oluşturulmuş REST servisidir.
Bir iş tanımı ve CV metinleri alır, her biri için uygunluk skoru döner.
"""

from fastapi import FastAPI, UploadFile, Form
from typing import List
from cvmatcher.predictor import predict_scores
from cvmatcher.extractor import extract_text

app = FastAPI(title="CV Matcher API", description="Yapay Zeka Destekli CV - İş Tanımı Eşleştirme Servisi", version="1.0")


@app.post("/match/")
async def match_cv_to_job(
    job_description: str = Form(...),
    cv_files: List[UploadFile] = []
):
    """
    📌 API Girdisi:
    - job_description: İş tanımı metni
    - cv_files: Birden fazla .pdf/.txt/.docx dosya yüklenebilir

    📤 API Çıktısı:
    - CV sırası ve uygunluk skorlarını içeren liste
    """
    cv_texts = []
    for file in cv_files:
        contents = await file.read()
        tmp_path = f"temp_{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(contents)

        text = extract_text(tmp_path)
        cv_texts.append(text)

    results = predict_scores(cv_texts, job_description)

    return {"results": [{"cv_index": i+1, "score": s} for i, s in results]}
