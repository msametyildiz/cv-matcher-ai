import pandas as pd
import os

# Giriş ve çıkış dosya yolları
input_path = "data/raw/kaggle_resumes/UpdatedResumeDataSet.csv"
output_dir = "data/processed"
output_path = os.path.join(output_dir, "training_data.csv")

# Klasör yoksa oluştur
os.makedirs(output_dir, exist_ok=True)

# CSV'yi oku
df = pd.read_csv(input_path)

# Temel filtreleme
df = df[["Resume", "Category"]].dropna()

# İş tanımı (job_text)
job_text = "Looking for a Data Scientist with experience in Python, Machine Learning, and NLP."

# Etiket: sadece 'Data Science' uygun
df["label"] = df["Category"].apply(lambda x: 1 if "data" in x.lower() else 0)

# Sütunları yeniden adlandır
df_out = df.rename(columns={"Resume": "cv_text"})
df_out["job_text"] = job_text
df_out = df_out[["cv_text", "job_text", "label"]]

# CSV'ye kaydet
df_out.to_csv(output_path, index=False)

print(f"✅ {len(df_out)} satır veri 'training_data.csv' olarak kaydedildi.")
