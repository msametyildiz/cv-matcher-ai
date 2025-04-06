import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
from urllib.parse import quote_plus

# ------------------ Ayarlar ------------------
LANGUAGES = {
    "en": {
        "base_url": "https://www.indeed.com/jobs?q=",
        "job_titles": [
            "Data Scientist", "Software Engineer", "Machine Learning Engineer",
            "Backend Developer", "Frontend Developer", "Project Manager",
            "Product Manager", "DevOps Engineer", "IT Support Specialist",
            "Sales Executive", "Business Analyst", "Graphic Designer",
            "Marketing Manager", "Financial Analyst", "Human Resources Specialist",
            "Mobile App Developer", "Cybersecurity Analyst", "System Administrator",
            "UX UI Designer", "Content Writer"
        ]
    },
    "tr": {
        "base_url": "https://tr.indeed.com/i%C5%9F-ilanlar?q=",
        "job_titles": [
            "Veri Bilimci", "Yazılım Mühendisi", "Makine Öğrenmesi Mühendisi",
            "Backend Geliştirici", "Frontend Geliştirici", "Proje Yöneticisi",
            "Ürün Yöneticisi", "DevOps Mühendisi", "BT Destek Uzmanı",
            "Satış Temsilcisi", "İş Analisti", "Grafik Tasarımcı",
            "Pazarlama Yöneticisi", "Finansal Analist", "İnsan Kaynakları Uzmanı",
            "Mobil Uygulama Geliştirici", "Siber Güvenlik Uzmanı", "Sistem Yöneticisi",
            "UX UI Tasarımcı", "İçerik Yazarlığı"
        ]
    }
}

HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_POSTS_PER_ROLE = 15

# ------------------ Fonksiyonlar ------------------
def scrape_indeed_jobs(lang):
    config = LANGUAGES[lang]
    records = []

    for title in tqdm(config["job_titles"], desc=f"[{lang.upper()}] Meslekler taranıyor"):
        url = config["base_url"] + quote_plus(title)
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        jobs = soup.select("a.tapItem")[:MAX_POSTS_PER_ROLE]

        for job in jobs:
            job_title = job.select_one("h2.jobTitle span")
            company = job.select_one("span.companyName")
            location = job.select_one("div.companyLocation")
            summary = job.select_one("div.job-snippet")
            link = "https://www.indeed.com" + job.get("href") if job.get("href") else None

            records.append({
                "searched_title": title,
                "job_title": job_title.text.strip() if job_title else None,
                "company": company.text.strip() if company else None,
                "location": location.text.strip() if location else None,
                "description": summary.text.strip().replace("\n", " ") if summary else None,
                "link": link
            })

        time.sleep(1)

    return pd.DataFrame(records)

# ------------------ Ana Akış ------------------
if __name__ == "__main__":
    for lang in ["en", "tr"]:
        df = scrape_indeed_jobs(lang)
        output_path = f"data/raw/job_postings_{lang}.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ {lang.upper()} için {len(df)} ilan kaydedildi: {output_path}")

