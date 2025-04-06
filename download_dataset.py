from datasets import load_dataset
import pandas as pd
import os

# Hugging Face'ten geçerli veri setini indir
print("📥 Hugging Face veri seti indiriliyor...")
dataset = load_dataset("smasala/jobmatcher-corpus")

# Eğitim verisini pandas DataFrame'e çevir
df = dataset["train"].to_pandas()

# İlk 5 satırı göster
print("🔍 İlk 5 satır:")
print(df.head())

# Hedef klasörü oluştur
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# CSV dosyasına kaydet
output_path = os.path.join(output_dir, "training_data.csv")
df.to_csv(output_path, index=False)

print(f"✅ Veri başarıyla kaydedildi: {output_path}")
