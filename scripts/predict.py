from cvmatcher.predictor import predict_scores
from cvmatcher.preprocessor import preprocess

# Ortak CV ve iş tanımı
cv_texts = [
    preprocess("Experienced NLP engineer skilled in Python, Transformers, and deep learning."),
    preprocess("Financial expert with 10 years of experience in Excel and budgeting."),
    preprocess("Salesperson with strong communication and CRM skills.")
]

job_text = preprocess("We are hiring an NLP Engineer with experience in Transformers and deep learning.")

# Skorları al
results = predict_scores(cv_texts, job_text)

# Sonuçları yazdır
print("\n🔍 Terminal Model Tahminleri:\n")
for idx, (i, score) in enumerate(results):
    print(f"CV {i + 1}: {score}% uyum")
