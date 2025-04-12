import sys
import os
print("Current Working Directory:", os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from pathlib import Path
from cvmatcher.predictor import predict_scores
from cvmatcher.extractor import extract_text
from cvmatcher.config import RAW_DATA_DIR
from cvmatcher.preprocessor import preprocess
import matplotlib.pyplot as plt

# Sayfa yapılandırması
st.set_page_config(page_title="CV Matcher AI", layout="wide")

st.title("📄 CV - İş Tanımı Eşleşme Aracı")
st.markdown("Bu uygulama, yüklediğiniz iş tanımına göre klasördeki tüm CV'leri analiz eder ve **uygunluk skorlarına** göre sıralar.")

# 🔹 İş Tanımı Yükle
st.subheader("📝 İş Tanımını Yükle")
job_text = st.text_area("İş ilanı metnini bu alana yapıştırın:", height=200)

# 🔹 CV Klasörü Seçimi
st.subheader("📁 CV Klasörü")
cv_folder = st.text_input("CV klasörünün yolu (örneğin: data/raw/cvs):", value=str(RAW_DATA_DIR))

# 🔍 Skor Hesapla Butonu
if st.button("🔍 Uygunluk Skorlarını Hesapla") and job_text and cv_folder:
    cv_folder_path = Path(cv_folder)
    cv_texts = []
    file_names = []

    for file in os.listdir(cv_folder_path):
        file_path = cv_folder_path / file
        if file_path.suffix.lower() in [".txt", ".pdf", ".docx"]:
            try:
                text = extract_text(str(file_path))
                if text.strip():
                    preprocessed_text = preprocess(text)
                    cv_texts.append(preprocessed_text)
                    file_names.append(file)
            except Exception as e:
                st.warning(f"Dosya {file} okunamadı: {str(e)}")

    if not cv_texts:
        st.warning("Desteklenen formatta CV bulunamadı. Lütfen .txt, .pdf veya .docx yükleyin.")
    else:
        with st.spinner("Tahminler yapılıyor..."):
            preprocessed_job_text = preprocess(job_text)
            scores = predict_scores(cv_texts, preprocessed_job_text)

        st.success(f"{len(scores)} adet CV analiz edildi.")
        st.subheader("📊 Uygunluk Skorları")

        score_table = [
            {"Dosya Adı": file_names[idx], "Skor (%)": score}
            for idx, score in scores
        ]
        st.dataframe(score_table)

        # 🏆 En iyi eşleşme
        best_idx, best_score = scores[0]
        st.markdown(f"🏆 **En Uygun CV: `{file_names[best_idx]}` → Skor: `{best_score}%`**")

        # 📊 Bar chart
        sorted_scores = sorted(zip(file_names, [score for _, score in scores]), key=lambda x: x[1], reverse=True)
        labels = [name for name, _ in sorted_scores]
        values = [score for _, score in sorted_scores]

        fig, ax = plt.subplots()
        ax.barh(labels, values, color='skyblue')
        ax.invert_yaxis()
        ax.set_xlabel("Uygunluk Skoru (%)")
        ax.set_title("CV - İş Tanımı Eşleşme Skorları")
        st.pyplot(fig)

else:
    st.info("Lütfen bir iş tanımı ve CV klasör yolu giriniz.")
