import pandas as pd
from pathlib import Path

# Eğitim için örnek veri
data = {
    "cv_text": [
        "Experienced software engineer with 5 years in Python and NLP. Built models using BERT and HuggingFace.",
        "Graduated in economics. Background in financial modeling and accounting. No software experience.",
        "NLP researcher skilled in machine learning, PyTorch, and large language models. Published papers in ACL.",
        "Sales representative with strong communication skills. No programming knowledge.",
        "Interned at a tech company working on computer vision and Python scripts. Basic NLP experience.",
    ],
    "job_text": [
        "We are hiring an NLP engineer with strong Python skills and experience in transformer models."
    ] * 5,
    "label": [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Hedef CSV dosyasını oluştur
output_path = Path("data/processed/training_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)
print("✅ training_data.csv başarıyla oluşturuldu!")
