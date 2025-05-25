# 🔍 JobHunt: Smart Job Discovery & Alerts

A Streamlit web app that uses Machine Learning and Conversational AI to intelligently cluster job listings based on skills, helping users discover jobs that match their interests — faster and smarter.

---

## 📌 Features

- ✅ Skill-based job filtering
- 🤖 AI-powered clustering using TF-IDF and KMeans
- 📊 Automatic optimal cluster detection with Elbow and Silhouette methods
- 🌐 Live job scraping from Karkidi.com
- 💬 Multilingual-ready chatbot-based donor/job screening (customizable)
- 📁 CSV data input and model persistence

---

├── app.py                 # Streamlit app
├── model.py               # ML logic (TF-IDF, clustering)
├── scrape.py              # Web scraper for job listings
├── data/
│   ├── jobs.csv           # Raw job data (required for app startup)
│   ├── jobs_clustered.csv # Output file after clustering
│   └── model.pkl / vectorizer.pkl # Saved ML models


---

```bash
pip install -r requirements.txt

---

